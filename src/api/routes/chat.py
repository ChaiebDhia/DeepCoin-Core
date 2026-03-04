"""
src/api/routes/chat.py
=======================
AI numismatic knowledge chat endpoint.

WHAT: Accepts a natural-language question about ancient coins (coin types,
      dynasties, materials, historical figures, mint cities, iconography)
      and returns a grounded answer drawn from the Corpus Nummorum knowledge
      base — 9,541 coin types with 47,705 semantic chunks.

WHY this endpoint exists:
    DeepCoin already has an 80%-accurate CNN classifier and a 9,541-type
    knowledge base. Most platforms stop there. The chat endpoint makes the
    KB CONVERSATIONAL: instead of needing the exact CN type ID, a museum
    curator can ask "tell me about silver tetradrachms from Athens" and get
    an immediate, structured answer grounded in real numismatic data.

    This is the "AI librarian" feature — the KB data was there all along;
    the chat layer makes it accessible to non-technical stakeholders.

WHY grounded (not open-ended):
    LLMs tend to fabricate plausible-sounding historical details.  In
    numismatics (museum documentation, auction house provenance), a
    hallucinated fact means a wrongly attributed artefact.  Every claim
    in the answer must cite a [CONTEXT N] block drawn from the RAG search.
    The LLM's job is natural language quality, not historical invention.

ARCHITECTURE (per request):
    1. RAG search: query → BM25 + vector hybrid → top-5 coin type chunks
    2. Context injection: 5 chunks formatted as [CONTEXT 1]…[CONTEXT 5]
    3. LLM call (blocking, run in asyncio.to_thread):
       provider priority: Ollama (gemma3:4b) → GitHub Models → Google AI Studio
    4. Fallback: if no LLM available, return structured context as prose
    5. Return: {answer, sources, provider}

RATE LIMITING:
    Chat is CPU/GPU bound (LLM inference).  In production, this should be
    gated behind the same 10/min rate limit as /api/classify.  Left open
    in development for testing convenience.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi        import APIRouter, HTTPException
from pydantic       import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chat"])


# ── request / response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Chat query body.

    Fields:
        query — The natural language question (e.g. "What are silver coins
                from Maroneia?"). Required, max 500 characters.
        n_sources — How many RAG context chunks to inject (1–10). A larger
                    value gives richer context at the cost of longer prompts.
    """
    query:    str = Field(..., min_length=1, max_length=500, description="Numismatic question")
    n_sources: int = Field(5, ge=1, le=10, description="Number of KB chunks to retrieve")


class ChatSource(BaseModel):
    """One knowledge-base source chunk cited in the answer."""
    type_id:    str
    chunk_type: str
    snippet:    str
    score:      float


class ChatResponse(BaseModel):
    """
    Chat answer with cited sources.

    Fields:
        answer    — LLM-generated (or fallback) answer, grounded in context.
        sources   — The KB chunks used as context, for transparency.
        provider  — Which LLM provider generated the answer.
    """
    answer:   str
    sources:  list[ChatSource]
    provider: str


# ── RAG + LLM pipeline (blocking — runs in to_thread) ────────────────────────

def _run_chat(query: str, n_sources: int) -> dict[str, Any]:
    """
    Execute a complete RAG → LLM chat pipeline synchronously.

    This function is blocking (LLM inference, ChromaDB reads) so it is
    always called via asyncio.to_thread() from the async endpoint handler.

    Steps:
        1. Load the singleton RAG engine (already warm from gatekeeper startup).
        2. Hybrid search: BM25 keyword + vector cosine → top-n_sources chunks.
        3. Format context blocks [CONTEXT 1]…[CONTEXT n].
        4. Build the grounded prompt: inject contexts + strict instruction.
        5. Call LLM (Ollama / GitHub / Google); fall back to structured prose.
        6. Return {answer, sources, provider}.
    """
    # ── 1. Load RAG engine ─────────────────────────────────────────────────
    from src.core.rag_engine import get_rag_engine
    rag = get_rag_engine()

    # ── 2. Hybrid search ───────────────────────────────────────────────────
    hits = rag.search(query, n=n_sources)
    if not hits:
        return {
            "answer":   (
                "No relevant coin types found in the Corpus Nummorum knowledge base "
                f"for your query: \"{query}\". Try a different search term, such as "
                "the denomination, dynasty name, region, or material."
            ),
            "sources":  [],
            "provider": "no-results",
        }

    # ── 3. Format context blocks ────────────────────────────────────────────
    context_lines: list[str] = []
    sources: list[ChatSource] = []
    # WHY 'hit' directly (not hit.get('metadata', {})):
    #   rag_engine.search() returns flat dicts with type_id, chunk_type, etc.
    #   at the top level — there is no nested 'metadata' key. Using hit directly
    #   prevents all sources from showing '? / ?' in the frontend SourceChip.
    for i, hit in enumerate(hits, 1):
        snippet = hit.get("document", hit.get("text", ""))[:300]
        type_id    = str(hit.get("type_id", hit.get("id", "?")))
        chunk_type = str(hit.get("chunk_type", hit.get("chunk_types", ["?"])[0] if hit.get("chunk_types") else "?"))
        context_lines.append(f"[CONTEXT {i}] Type {type_id} ({chunk_type}):\n{snippet}")
        sources.append(ChatSource(
            type_id    = type_id,
            chunk_type = chunk_type,
            snippet    = snippet,
            score      = float(hit.get("rrf_score", hit.get("score", 0.0))),
        ))

    context_block = "\n\n".join(context_lines)

    # ── 4. Build grounded prompt ────────────────────────────────────────────
    prompt = (
        f"{context_block}\n\n"
        "INSTRUCTION: You are an expert numismatist. "
        "Answer the user's question using ONLY the context blocks above. "
        "Cite each fact with [CONTEXT N]. "
        "Do NOT add information that is not present in the context. "
        "If the context does not contain enough information, say so explicitly. "
        "Write 2–3 concise paragraphs in professional English.\n\n"
        f"USER QUESTION: {query}"
    )

    # ── 5. Call LLM (reuse historian's provider chain) ─────────────────────
    from src.agents.historian import _get_llm
    client, model = _get_llm("text")
    provider = model if model else "fallback"

    if client is None:
        # Structured fallback: stitch context snippets into prose
        answer = (
            f"Based on the Corpus Nummorum knowledge base, here are the most relevant "
            f"coin types for your query \"{query}\":\n\n"
        )
        for src in sources:
            answer += f"• CN Type {src.type_id} ({src.chunk_type}): {src.snippet[:150]}…\n"
        provider = "structured-fallback"
        return {"answer": answer, "sources": [s.model_dump() for s in sources], "provider": provider}

    try:
        response = client.chat.completions.create(
            model       = model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 600,
            temperature = 0.3,
        )
        answer = response.choices[0].message.content or ""
        # Strip any thinking artifacts from reasoning models
        if "<think>" in answer:
            parts = answer.split("</think>", 1)
            answer = parts[-1].strip()
    except Exception as exc:
        logger.warning("Chat LLM call failed (%s): %s", provider, exc)
        # Graceful degradation — return structured context instead
        answer = (
            f"LLM unavailable ({exc.__class__.__name__}). "
            f"Top results from the knowledge base for \"{query}\":\n\n"
        )
        for src in sources:
            answer += f"• CN Type {src.type_id} ({src.chunk_type}): {src.snippet[:200]}…\n"
        provider = "fallback-on-error"

    return {
        "answer":   answer,
        "sources":  [s.model_dump() for s in sources],
        "provider": provider,
    }


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model = ChatResponse,
    summary        = "Ask the DeepCoin AI about Corpus Nummorum coin types",
)
async def chat(
    body: ChatRequest,
) -> ChatResponse:
    """
    Natural language Q&A over the Corpus Nummorum 9,541-type knowledge base.

    NO authentication required — the knowledge base content is public
    numismatic data from the Corpus Nummorum project.

    INPUT:
        query — e.g. "What are the silver coins of Alexander the Great?"
        n_sources — how many KB chunks to retrieve (default 5)

    OUTPUT:
        answer — a 2–3 paragraph grounded response, citing [CONTEXT N] blocks
        sources — the KB chunks used (type_id, chunk_type, snippet, score)
        provider — which LLM was used (or "structured-fallback")

    LATENCY:
        With Ollama gemma3:4b on RTX 3050 Ti: ~8–15 s
        With GitHub Models Gemini 2.5 Flash: ~3–8 s
        With structured fallback (no LLM): < 100 ms
    """
    try:
        result = await asyncio.to_thread(_run_chat, body.query, body.n_sources)
    except Exception as exc:
        logger.error("chat endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat pipeline error: {exc}")

    return ChatResponse(**result)
