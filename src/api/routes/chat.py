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
        1. Detect a specific CN type ID in the query (e.g. "CN 4776").
           If found, call get_context_blocks() to pull ALL 5 structured
           chunks for that exact type as the primary grounded context.
        2. Hybrid BM25+vector search for supplementary related chunks.
        3. Build a system+user message pair:
           - System: expert numismatist persona + explicit permission to use
             general numismatic knowledge when KB data is sparse.
           - User:   primary CN record + supplementary hits + question.
        4. Call LLM (Ollama → GitHub → Google); fall back to structured prose.
        5. Return {answer, sources, provider}.

    WHY system+user messages (not single prompt):
        The older single-prompt approach told the LLM "only use context, say
        insufficient information if missing." This produced empty responses for
        coin types with sparse KB data. The new system message gives the LLM an
        expert persona and explicitly requires it to supplement with general
        numismatic knowledge — matching standard practice in museum documentation
        where an expert contextualises sparse records with period/dynasty knowledge.

    WHY get_context_blocks() for specific type queries:
        Semantic search on "CN 4776 ancient coin numismatics" matches chunks
        whose TEXT is similar to that phrase — not necessarily the chunks FOR
        type 4776.  In particular, chunk_type labels like "identity" and "context"
        can score high on semantic similarity to the query itself, returning
        scaffold text instead of substantive coin data.  Direct type lookup via
        get_context_blocks() guarantees all 5 structured fields for the requested
        type are injected as primary context before the semantic results.
    """
    import re as _re

    from src.core.rag_engine import get_rag_engine
    rag = get_rag_engine()

    # ── 1. Detect specific CN type ID in the query ──────────────────────────
    # Matches: "CN 4776", "CN4776", "type 4776", "Type-4776", bare "4776" etc.
    cn_id_match = _re.search(r'(?:CN|cn|Type|type)\s*[-#]?\s*(\d{3,6})\b', query)
    primary_context   = ""
    primary_sources: list[ChatSource] = []

    if cn_id_match:
        detected_id = cn_id_match.group(1)
        try:
            context_str = rag.get_context_blocks(int(detected_id))
            # get_context_blocks() returns "(no identity data for type N)" when
            # the type is not in the corpus — skip those as primary context.
            if f"(no identity data for type {detected_id})" not in context_str:
                primary_context = (
                    f"=== CORPUS NUMMORUM COMPLETE RECORD: CN TYPE {detected_id} ===\n"
                    f"{context_str}\n"
                )
                record = rag.get_by_id(int(detected_id))
                if record:
                    primary_sources.append(ChatSource(
                        type_id    = detected_id,
                        chunk_type = "full_record",
                        snippet    = context_str[:300],
                        score      = 1.0,
                    ))
            else:
                logger.debug("CN type %s not in RAG corpus — falling back to search", detected_id)
        except Exception as exc:
            logger.debug("get_context_blocks(%s) failed: %s", detected_id, exc)

    # ── 2. Hybrid search for supplementary context ──────────────────────────
    hits = rag.search(query, n=n_sources)

    if not hits and not primary_context:
        return {
            "answer": (
                f"No relevant coin types found in the Corpus Nummorum knowledge base "
                f"for your query: \"{query}\". "
                "Try searching by denomination (tetradrachm, denarius, drachm), "
                "dynasty (Seleucid, Ptolemaic, Roman Imperial), region (Thrace, Athens, "
                "Alexandria), material (silver, bronze, gold electrum), or a specific CN "
                "type ID such as \"CN 1015\"."
            ),
            "sources":  [],
            "provider": "no-results",
        }

    # ── 3. Format supplementary context blocks ──────────────────────────────
    # WHY 'hit' directly (not hit.get('metadata', {})):
    #   rag_engine.search() returns flat dicts — no nested 'metadata' key.
    supplementary_lines: list[str] = []
    supplementary_sources: list[ChatSource] = []
    seen_type_ids: set[str] = {cn_id_match.group(1)} if cn_id_match else set()

    for hit in hits:
        type_id    = str(hit.get("type_id", hit.get("id", "?")))
        if type_id in seen_type_ids:
            continue   # deduplicate — don't repeat the primary type's chunks
        seen_type_ids.add(type_id)
        chunk_type = str(hit.get("chunk_type", "?"))
        snippet    = hit.get("document", hit.get("text", ""))[:300]
        n          = len(supplementary_lines) + 1
        supplementary_lines.append(
            f"[RELATED CONTEXT {n}] CN Type {type_id} ({chunk_type}):\n{snippet}"
        )
        supplementary_sources.append(ChatSource(
            type_id    = type_id,
            chunk_type = chunk_type,
            snippet    = snippet,
            score      = float(hit.get("rrf_score", hit.get("score", 0.0))),
        ))

    all_sources = primary_sources + supplementary_sources

    # ── 4. Build system + user messages ─────────────────────────────────────
    system_message = (
        "You are DeepCoin AI, an expert numismatist and ancient historian specializing "
        "in the Corpus Nummorum (CN) database — a DFG-funded scholarly catalogue of "
        "9,716 ancient coin types spanning Greek, Hellenistic, Roman provincial, and "
        "imperial mints from the 7th century BC to the 4th century AD.\n\n"
        "YOUR RESPONSE MUST:\n"
        "1. GROUND specific facts (dates, denominations, weights, mint cities, legends, "
        "   iconography) in the provided context blocks, citing them as [CONTEXT N] or "
        "   [CN Type XXXX].\n"
        "2. SUPPLEMENT with your expert numismatic knowledge when context data is sparse "
        "   — this is REQUIRED, not optional. Mark supplemented facts explicitly as "
        "   'Historically,' or '[General numismatic knowledge]'. A sparse KB record "
        "   for a well-known coin type (e.g. Athenian tetradrachm, Roman denarius) is "
        "   not a reason to withhold expertise.\n"
        "3. STRUCTURE your answer clearly:\n"
        "   • Coin identity (type, denomination, issuing authority, date range)\n"
        "   • Physical / iconographic description (obverse, reverse, metal, weight)\n"
        "   • Historical and numismatic significance (mint city, ruler, dynasty, context)\n"
        "4. NEVER respond with 'insufficient information' — a senior numismatist always "
        "   provides the most complete assessment possible from available evidence.\n"
        "5. Write in precise, professional English suitable for museum documentation "
        "   (2–4 paragraphs)."
    )

    context_section = ""
    if primary_context:
        context_section += primary_context + "\n"
    if supplementary_lines:
        context_section += (
            "=== RELATED COIN TYPES FROM THE CORPUS NUMMORUM ===\n"
            + "\n\n".join(supplementary_lines) + "\n"
        )

    user_message = (
        f"{context_section}\n"
        f"NUMISMATIC QUESTION: {query}\n\n"
        "Provide a complete, authoritative numismatic analysis using the context above "
        "and your expert knowledge."
    )

    # ── 5. Call LLM (reuse historian's provider chain) ─────────────────────
    from src.agents.historian import _get_llm
    client, model = _get_llm("text")
    provider = model if model else "fallback"

    if client is None:
        # Structured fallback: assemble context snippets into readable prose
        answer = f"**Corpus Nummorum results for: \"{query}\"**\n\n"
        if primary_context:
            answer += primary_context + "\n"
        if supplementary_sources:
            answer += "**Related types:**\n"
            for src in supplementary_sources[:4]:
                answer += f"• CN Type {src.type_id} ({src.chunk_type}): {src.snippet[:150]}…\n"
        provider = "structured-fallback"
        return {"answer": answer, "sources": [s.model_dump() for s in all_sources], "provider": provider}

    try:
        response = client.chat.completions.create(
            model       = model,
            messages    = [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 1000,
            temperature = 0.4,
        )
        answer = response.choices[0].message.content or ""
        # Strip thinking artifacts from reasoning models (DeepSeek-R1, o1, etc.)
        if "<think>" in answer:
            parts = answer.split("</think>", 1)
            answer = parts[-1].strip()
    except Exception as exc:
        logger.warning("Chat LLM call failed (%s): %s", provider, exc)
        answer = (
            f"LLM unavailable ({exc.__class__.__name__}). "
            f"Top Corpus Nummorum results for \"{query}\":\n\n"
        )
        for src in all_sources[:4]:
            answer += f"• CN Type {src.type_id} ({src.chunk_type}): {src.snippet[:200]}…\n"
        provider = "fallback-on-error"

    return {
        "answer":   answer,
        "sources":  [s.model_dump() for s in all_sources],
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
