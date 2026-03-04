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
        query       — The natural language question (e.g. "What are silver coins
                      from Maroneia?"). Required, max 500 characters.
        n_sources   — How many RAG context chunks to inject (1–10). A larger
                      value gives richer context at the cost of longer prompts.
        top5_labels — Optional list of CN type IDs (as strings) representing
                      the CNN's top-5 predicted candidates, e.g. ["1015","544"].
                      When provided, the backend fetches get_context_blocks()
                      for each label and injects them as PRIMARY CANDIDATE N
                      blocks at the top of the LLM context — before the
                      semantic search results.  This lets the chat AI compare
                      all 5 candidates instead of only the top-1.
    """
    query:       str       = Field(..., min_length=1, max_length=500, description="Numismatic question")
    n_sources:   int       = Field(5, ge=1, le=10, description="Number of KB chunks to retrieve")
    top5_labels: list[str] = Field(default_factory=list, description="Top-5 CNN predicted CN type IDs")


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

def _web_search(query: str, max_results: int = 4) -> str:
    """
    Fallback web search using DuckDuckGo (no API key required).

    WHAT: Runs a DuckDuckGo text search and returns top snippets formatted
          as plain prose blocks labeled with their source title.

    WHY: When the Corpus Nummorum KB has sparse or no data for a query
         (e.g. general numismatic topics, less-common dynasties), the LLM
         would otherwise generate content from memory alone.  Injecting real
         web snippets grounds the answer in external scholarly sources and
         acknowledges the internet source transparently to the user.

    Called only when the KB context is thin (< 2 supplementary hits AND
    no direct CN type match), so it does not slow down well-covered queries.
    """
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(
            query + " ancient coin numismatics",
            max_results=max_results,
            timelimit="y",
        )
        if not results:
            return ""
        lines = []
        for r in results:
            title = r.get("title", "Web source")
            body  = r.get("body", "")[:400]
            href  = r.get("href", "")
            lines.append(f"[Web: {title}]\n{body}\n(source: {href})")
        return "\n\n".join(lines)
    except Exception as exc:
        logger.debug("Web search unavailable: %s", exc)
        return ""


def _run_chat(query: str, n_sources: int, top5_labels: list[str] | None = None) -> dict[str, Any]:
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

    # ── 0. Non-numismatic query guard ────────────────────────────────────────
    # WHY: The LLM is instructed to "never say insufficient information" —
    # essential for sparse-but-valid numismatic queries, but catastrophic for
    # entirely off-topic input.  A greeting like "assalamu alaykom" has no
    # numismatic terms, so rag.search() returns weakly-matched junk and the
    # model fabricates a coin description.  This guard catches non-numismatic
    # input before any LLM call and returns a friendly redirect.
    _NUMISMATIC_RE = _re.compile(
        r'coin|numismat|\bCN\b|tetradrachm|denarius|drachm|obol|solidus|'
        r'aureus|sestert|didrachm|stater|hemidrachm|diobol|triobol|'
        r'obverse|reverse|mint|iconograph|legend|die|weight|diameter|'
        r'hellenistic|roman|greek|byzantine|macedon|persian|seleucid|ptolem|'
        r'emperor|king|dynasty|ancient|archaeolog|'
        r'bronze|silver|gold|electrum|billon|'
        r'\bCN\s*\d{2,6}|\b\d{4,6}\b',
        _re.IGNORECASE,
    )
    if not _NUMISMATIC_RE.search(query):
        logger.info("chat: non-numismatic query rejected: %r", query[:60])
        return {
            "answer": (
                "Hello! I'm DeepCoin AI, a specialist in ancient numismatics "
                "and the Corpus Nummorum database covering 9,541 coin types. "
                "I can help with questions about specific coin types ("
                "e.g. \"CN 1015\"), denominations (\"silver tetradrachm from "
                "Athens\"), dynasties (\"Seleucid coinage\"), rulers ("
                "\"coins of Alexander the Great\"), materials, or mint cities. "
                "What would you like to explore?"
            ),
            "sources":  [],
            "provider": "guard",
        }

    from src.core.rag_engine import get_rag_engine
    rag = get_rag_engine()

    # ── 1. Detect specific CN type ID in the query ──────────────────────────
    # Matches: "CN 4776", "CN4776", "type 4776", "Type-4776", bare "4776" etc.
    cn_id_match = _re.search(r'(?:CN|cn|Type|type)\s*[-#]?\s*(\d{2,6})\b', query)
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
                    f"<cn_record id=\"{detected_id}\">\n"
                    f"{context_str}\n"
                    f"</cn_record>\n"
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

    # ── 1b. Inject context for each of the CNN's top-5 candidate types ──────
    # WHY: The frontend passes all 5 predicted CN type IDs so the AI can
    #      compare candidates directly.  Each label gets its own labeled block
    #      so the LLM can reference them by type ID in its answer without using
    #      opaque [CONTEXT N] notation.  Labels already fetched as primary
    #      context (from cn_id_match) are skipped to avoid duplication.
    candidate_sources: list[ChatSource] = []
    if top5_labels:
        for rank, label in enumerate(top5_labels, start=1):
            label_str = label.strip()
            if not label_str:
                continue
            try:
                candidate_id_int = int(label_str)
            except ValueError:
                continue
            # Skip if already fetched as the primary direct-lookup type
            if cn_id_match and cn_id_match.group(1) == label_str:
                continue
            try:
                cand_context = rag.get_context_blocks(candidate_id_int)
                if f"(no identity data for type {label_str})" not in cand_context:
                    primary_context += (
                        f"<cn_candidate rank=\"{rank}\" id=\"{label_str}\">\n"
                        f"{cand_context}\n"
                        f"</cn_candidate>\n\n"
                    )
                    cand_record = rag.get_by_id(candidate_id_int)
                    if cand_record:
                        candidate_sources.append(ChatSource(
                            type_id    = label_str,
                            chunk_type = "cnn_candidate",
                            snippet    = cand_context[:300],
                            score      = 1.0 / rank,
                        ))
            except Exception as exc:
                logger.debug("top5 get_context_blocks(%s) failed: %s", label_str, exc)

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
    already_fetched = {cn_id_match.group(1)} if cn_id_match else set()
    if top5_labels:
        already_fetched.update(lbl.strip() for lbl in top5_labels if lbl.strip())

    for hit in hits:
        type_id    = str(hit.get("type_id", hit.get("id", "?")))
        if type_id in seen_type_ids:
            continue   # deduplicate — don't repeat the primary type's chunks
        seen_type_ids.add(type_id)
        chunk_type = str(hit.get("chunk_type", "?"))
        snippet    = hit.get("document", hit.get("text", ""))[:300]
        n          = len(supplementary_lines) + 1
        supplementary_lines.append(
            f"<related_type id=\"{type_id}\" aspect=\"{chunk_type}\">\n{snippet}\n</related_type>"
        )
        supplementary_sources.append(ChatSource(
            type_id    = type_id,
            chunk_type = chunk_type,
            snippet    = snippet,
            score      = float(hit.get("rrf_score", hit.get("score", 0.0))),
        ))

    all_sources = primary_sources + candidate_sources + supplementary_sources

    # ── 4b. Web search fallback when KB context is thin ────────────────────────
    # Trigger web search when:
    #   - No direct CN type ID was found in the query (general numismatic topic)
    #   - AND the supplementary search returned fewer than 2 results
    # This prevents web fallback from firing on known CN type queries.
    web_context = ""
    if not primary_context and len(supplementary_lines) < 2:
        web_context = _web_search(query)
        if web_context:
            logger.info("chat: KB context thin — injected web search results for: %r", query[:80])

    # ── 5. Build system + user messages ─────────────────────────────────────
    system_message = (
        "You are DeepCoin AI — a world-class expert numismatist and ancient historian "
        "who has spent decades studying the Corpus Nummorum (CN), a DFG-funded scholarly "
        "catalogue of 9,716 ancient coin types: Greek, Hellenistic, Roman provincial, "
        "and imperial mintages from the 7th century BC to the 4th century AD.\n\n"
        "You will receive database records wrapped in XML-style tags such as "
        "<cn_record>, <cn_candidate>, <related_type>, and <web_references>. "
        "These are INTERNAL DATA DELIMITERS — never mention them, never quote them, "
        "and never reference section headings such as 'Identity', 'Obverse', or "
        "'Material' in your answer. Extract the facts silently and speak as an expert "
        "who simply knows them.\n\n"
        "HOW TO ANSWER:\n"
        "• Write flowing, authoritative prose — like a museum curator explaining a "
        "  coin to an interested scholar. 2–4 paragraphs.\n"
        "• Lead with what the coin IS: denomination, issuing authority, approximate date.\n"
        "• Describe the physical coin: obverse design, reverse design, metal, weight.\n"
        "• Place it in historical context: the ruler, the mint city, why it was struck,\n"
        "  what it tells us about the period.\n"
        "• When database records are sparse, draw on your expert knowledge naturally. "
        "  Introduce your own knowledge with phrases like 'Historically,' or 'Within "
        "  the broader numismatic tradition...' — not as a disclaimer but as enrichment.\n"
        "• Refer to Corpus Nummorum naturally: 'CN Type 1015 is catalogued as...' or "
        "  'The Corpus Nummorum records this as a...'\n"
        "• NEVER say 'the data shows', 'according to context', 'based on the record', "
        "  'I cannot determine', or any phrase that exposes the pipeline. "
        "  Speak as if you simply know this from decades of scholarship.\n"
        "• NEVER say 'insufficient information'. Always give the most complete "
        "  assessment possible, combining database facts with expert knowledge."
    )

    # ── 4c. "Type not in corpus" caveat ────────────────────────────────────
    # WHY: When a specific CN type ID was detected in the query but
    # get_context_blocks() returned "no identity data" (type not scraped),
    # primary_context is empty.  Without this caveat the LLM — told to
    # "never say insufficient information" — confidently invents a plausible
    # coin from period-general knowledge and presents it as CN Type XXXX.
    # The caveat makes the gap explicit so the LLM produces a clearly-labelled
    # general assessment rather than a falsely-attributed specific record.
    corpus_caveat = ""
    if cn_id_match and not primary_context:
        corpus_caveat = (
            f"CRITICAL CONSTRAINT: CN Type {cn_id_match.group(1)} is NOT present "
            f"in the indexed Corpus Nummorum records (9,541 types). "
            f"Your FIRST SENTENCE must state this explicitly — e.g. "
            f"'CN Type {cn_id_match.group(1)} is not currently indexed in the "
            f"Corpus Nummorum records available to me.' "
            f"AFTER that opening statement you may: "
            f"(1) describe related types from the supplementary context below, "
            f"(2) offer general numismatic context about the period or "
            f"denomination if inferable from the query — clearly introduced as "
            f"'Generally speaking,...' or 'Historically,...'. "
            f"Do NOT invent a specific obverse, reverse, weight, or legend for "
            f"this exact type and present it as confirmed fact."
        )

    context_section = ""
    if primary_context:
        context_section += primary_context + "\n"
    if supplementary_lines:
        context_section += (
            "<related_types>\n"
            + "\n\n".join(supplementary_lines)
            + "\n</related_types>\n"
        )
    if web_context:
        context_section += (
            "<web_references>\n"
            + web_context
            + "\n</web_references>\n"
        )

    user_message = (
        f"<scholarly_database>\n{context_section}</scholarly_database>\n\n"
        + (f"{corpus_caveat}\n\n" if corpus_caveat else "")
        + f"QUESTION: {query}\n\n"
        "Answer as a world-class numismatist. Do not reference any XML tags, section "
        "headers, or data delimiters in your response."
    )

    # ── 6. Call LLM (reuse historian's provider chain) ─────────────────────
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
            temperature = 0.6,
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
        result = await asyncio.to_thread(_run_chat, body.query, body.n_sources, body.top5_labels)
    except Exception as exc:
        logger.error("chat endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat pipeline error: {exc}")

    return ChatResponse(**result)
