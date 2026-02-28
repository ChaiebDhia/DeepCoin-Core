"""
src/agents/investigator.py
===========================
Layer 3 — Investigator Agent

Triggered when CNN confidence < 0.40 (model is unsure).
Pipeline:
  1. Load coin image → base64 encode
  2. Send to a vision-capable LLM with a structured numismatic prompt:
       a. qwen3-vl:4b via Ollama  (offline, OLLAMA_HOST set)
       b. Gemini 2.5 Flash Vision (cloud, GITHUB_TOKEN or GOOGLE_API_KEY)
       c. OpenCV local fallback   (always available, no LLM required)
  3. Parse response into structured attributes
  4. Cross-reference the description against the full RAG corpus
     (9,541 types) to surface the closest matching coin type

Engineering notes:
  - Vision model selected via _get_llm(capability='vision') — same priority
    chain as Historian, but routes to qwen3-vl:4b instead of gemma3:4b
  - Image is sent as base64 data URL (OpenAI multimodal message format)
  - KB cross-reference uses the RAG engine (9,541 types) not the legacy KB
    (438 types) — critical for unknown coins outside the CNN training set
  - OpenCV fallback: HSV histogram + Sobel edge analysis — always runs when
    no vision LLM is configured, ensuring the pipeline never returns empty
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from src.core.knowledge_base import get_knowledge_base
from src.core.rag_engine     import get_rag_engine

# Reuse the same LLM loader as Historian (already handles text/vision split)
from src.agents.historian import _get_llm


# ══════════════════════════════════════════════════════════════════════════════

class Investigator:
    """
    Visual Investigator — Gemini Vision analysis for low-confidence coins.

    Input  (from Gatekeeper state):
        image_path : str   — path to the coin image
        class_id   : int   — CNN best guess (low confidence)
        confidence : float — CNN score (< 0.40)
        top5       : list  — CNN top-5 predictions

    Output dict:
        {
          "visual_description" : str,    # Gemini free-text description
          "detected_features"  : {
              "metal_color"        : str,
              "profile_direction"  : str,
              "inscriptions"       : list[str],
              "symbols"            : list[str],
              "condition"          : str,
          },
          "kb_matches"         : list[dict],  # top-3 KB hits from Gemini description
          "suggested_type_id"  : int | None,  # best KB match type_id
          "llm_used"           : bool,
        }
    """

    def __init__(self) -> None:
        self._kb  = get_knowledge_base()   # legacy KB (438 types) — fallback only
        self._rag = get_rag_engine()        # full corpus (9,541 types) — primary

    # ── public ────────────────────────────────────────────────────────────────

    def investigate(self, image_path: str, cnn_prediction: dict) -> dict:
        """
        Main entry point called by Gatekeeper.

        Parameters
        ----------
        image_path     : absolute path to coin image
        cnn_prediction : dict with keys class_id, label, confidence, top5
        """
        confidence = float(cnn_prediction["confidence"])
        top5       = cnn_prediction.get("top5", [])

        # 1. Gemini Vision description
        description, features, llm_used = self._describe_image(image_path, confidence, top5)

        # 2. KB cross-reference via RAG engine (9,541 types, hybrid BM25+vector)
        #    WHY RAG not legacy KB:
        #        The Investigator runs on LOW-confidence coins — coins the CNN
        #        struggled to classify.  These are often types outside the 438
        #        CNN training classes.  The legacy KB covers only those 438.
        #        The RAG engine covers all 9,541 scraped types, so an unknown
        #        coin from any part of the CN corpus can still surface a match.
        kb_matches: list[dict] = []
        suggested_type_id: int | None = None
        if description:
            raw_matches = self._rag.search(description, n=3)
            # RAG search returns chunk-level hits; normalise to record-level dicts
            kb_matches = []
            seen_ids: set = set()
            for hit in raw_matches:
                tid = hit.get("type_id")
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                record = self._rag.get_by_id(tid) or hit
                kb_matches.append({
                    "type_id":      record.get("type_id", tid),
                    "denomination": record.get("denomination", ""),
                    "mint":         record.get("mint", ""),
                    "region":       record.get("region", ""),
                    "date":         record.get("date", ""),
                    "material":     record.get("material", ""),
                    "score":        hit.get("rrf_score", hit.get("score", 0.0)),
                    "in_training_set": record.get("in_training_set", False),
                })
            if kb_matches:
                suggested_type_id = kb_matches[0]["type_id"]

        return {
            "visual_description":  description,
            "detected_features":   features,
            "kb_matches":          kb_matches,
            "suggested_type_id":   suggested_type_id,
            "llm_used":            llm_used,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _describe_image(self, image_path: str, confidence: float, top5: list) -> tuple:
        """
        Encode image as base64 and send to the vision-capable LLM.

        Uses _get_llm(capability="vision") so the correct model is selected:
          - Gemini 2.5 Flash (if GITHUB_TOKEN or GOOGLE_API_KEY set)
          - qwen3-vl:4b via Ollama (if OLLAMA_HOST set, fully offline)
          - Fallback description (if nothing configured)

        WHY capability="vision" matters:
            gemma3:4b (the text model) is text-only — it cannot accept images.
            qwen3-vl:4b is a Vision-Language model — it processes image+text.
            Requesting the wrong model would raise an API error.
            The capability parameter routes to the right model automatically.

        Returns (description_str, features_dict, llm_was_used).
        """
        client, model = _get_llm(capability="vision")
        if client is None:
            # No vision LLM configured — run local OpenCV analysis instead.
            # WHY not just return an error string:
            #   The Investigator's whole purpose is to describe unknown coins.
            #   Even without an LLM, OpenCV can extract metal color (HSV),
            #   edge sharpness (Sobel), and coin region statistics.
            #   The description is coarser than an LLM's, but it is real data
            #   that the RAG cross-reference can still query meaningfully.
            desc, feats = _opencv_fallback(image_path)
            return desc, feats, False

        # Read and base64-encode the image
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            # Detect MIME type from extension
            ext = Path(image_path).suffix.lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png",  "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
        except Exception as e:
            return f"Could not read image: {e}", _empty_features(), False

        # Build the CNN context hints
        cnn_hint = f"CNN confidence is low ({confidence:.1%})."
        if top5:
            candidates = ", ".join(
                f"{t['label']} ({t['confidence']:.1%})" for t in top5[:3]
            )
            cnn_hint += f" Top CNN candidates: {candidates}."

        prompt = (
            "You are an expert numismatist and archaeologist specialising in ancient Greek and Roman coins.\n"
            f"{cnn_hint}\n\n"
            "Carefully examine this coin image and provide a structured analysis.\n"
            "Use ONLY plain prose sentences. No Markdown. No asterisks, no bullet points, "
            "no dashes used as bullets, no numbered items, no bold, no headers.\n\n"
            "Structure your answer as exactly six labeled paragraphs, each starting with the "
            "label followed by a colon:\n\n"
            "METAL: State whether the coin is silver, bronze, gold, or unknown, with brief reasoning.\n"
            "OBVERSE: Describe the front face — portrait, deity, animal, symbols, and any visible legend text.\n"
            "REVERSE: Describe the reverse face — symbols, design, and any visible legend text.\n"
            "INSCRIPTIONS: Quote any readable letter sequences visible on either side. "
            "If none are readable, write None.\n"
            "CONDITION: One sentence assessment — well-preserved, worn, or corroded.\n"
            "IDENTIFICATION: Period, issuing authority, and region if determinable. "
            "If uncertain, state the most likely candidate with brief reasoning.\n\n"
            "Use numismatic terminology. If something is not visible, say so plainly."
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }],
                max_tokens=3000,   # reasoning models (qwen3-vl) output <think>...</think>
                                   # blocks before the answer — needs extra headroom.
                                   # 1500 was too tight; bumped to 3000 so the
                                   # structured answer is never cut off.
                temperature=0.3,
            )
            description = resp.choices[0].message.content.strip()
            # Some reasoning models return empty content + populated reasoning field
            # when max_tokens is too low. Guard against that edge case.
            if not description:
                description = getattr(resp.choices[0].message, "reasoning", "") or ""
            # qwen3-vl (and other thinking models) prefix the actual answer with
            # their internal chain-of-thought wrapped in <think>...</think> tags.
            # Strip those blocks so only the final structured response is stored.
            # WHY: The thinking text contains speculative reasoning, not numismatic
            # facts. Passing it to RAG search would dilute the query signal.
            description = _strip_think_tags(description)
            # Apply the same Markdown / citation cleanup used by the historian
            # WHY: qwen3-vl sometimes emits Markdown even with plain-text instructions.
            # _clean_narrative() strips **, *, ##, [CONTEXT N], curly quotes, etc.
            from src.agents.historian import _clean_narrative
            description = _clean_narrative(description)
            # Strip verbose reasoning preamble BEFORE the first section label
            description = _trim_preamble(description)
            # Strip 'Wait, ...' reasoning loops that appear INSIDE sections
            # (llama/gemma reasoning models sometimes second-guess mid-section)
            description = _strip_wait_loops(description)
            # Cap each section to 350 chars so a looping section cannot fill pages
            description = _cap_sections(description, max_chars=350)
            features = _parse_features(description)
            return description, features, True
        except Exception as e:
            return f"Vision LLM unavailable: {e}", _empty_features(), False


# ── helpers ──────────────────────────────────────────────────────────────────────

def _empty_features() -> dict:
    return {
        "metal_color":       "unknown",
        "profile_direction": "unknown",
        "inscriptions":      [],
        "symbols":           [],
        "condition":         "unknown",
    }


def _strip_wait_loops(text: str) -> str:
    """
    Remove 'Wait, ...' sentences emitted by reasoning models when they
    second-guess their previous answer mid-section.

    WHAT: Strips every clause/sentence starting with 'Wait,' from the text,
          then collapses the resulting excess whitespace.

    WHY: gemma3:4b and qwen3-vl:4b sometimes freeze in a self-correction loop
         inside the REVERSE section, repeating variants of 'Wait, the reverse has
         a square with a cross' dozens of times before continuing.  These
         repetitions carry zero numismatic content and bloat the PDF to 4 pages.
    """
    import re
    # Each 'Wait, ' clause runs to end of the same line or next period
    text = re.sub(r'[ \t]*\bWait,\s[^\n]*', '', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _cap_sections(text: str, max_chars: int = 350) -> str:
    """
    Truncate each structured section (METAL:, OBVERSE:, etc.) to max_chars.

    WHAT: Splits the VLM output on section labels, caps each body at max_chars
          (breaking at the last sentence boundary), and reassembles.

    WHY: Even after stripping 'Wait,' loops a runaway section can still be too
         long (e.g. a model that writes 10 sentences per section).  350 chars
         is enough for 2-3 sentences of archaeological description without
         truncating useful content.

    Graceful: if the split finds nothing the original text is returned unchanged.
    """
    import re
    _SEC = re.compile(
        r'^(METAL|OBVERSE|REVERSE|INSCRIPTIONS|CONDITION|IDENTIFICATION):',
        re.IGNORECASE | re.MULTILINE,
    )
    boundaries = [(m.start(), m.group(1).upper()) for m in _SEC.finditer(text)]
    if not boundaries:
        return text

    parts   = []
    for idx, (start, label) in enumerate(boundaries):
        end  = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
        # Body starts after 'LABEL:'
        colon_pos = text.index(':', start) + 1
        body = text[colon_pos:end].strip()
        if len(body) > max_chars:
            truncated = body[:max_chars]
            # Break at last full sentence
            m = re.search(r'[.!?](?=[^.!?]*$)', truncated)
            if m:
                truncated = truncated[:m.end()]
            body = truncated.strip() + '...'
        parts.append(f"{label}: {body}")
    return '\n'.join(parts)


def _trim_preamble(text: str) -> str:
    """
    Strip verbose reasoning text that precedes the structured answer sections.

    WHY this is needed:
        qwen3-vl:4b (and similar reasoning models run via Ollama) sometimes
        output their chain-of-thought as plain prose BEFORE the structured
        METAL: / OBVERSE: / etc. sections, without wrapping it in <think> tags.
        Example of problematic output:
            "Got it, let's tackle this step by step. First I need to look at...
             Starting with METAL/MATERIAL. The coin looks light-colored...
             [500+ words of stream-of-consciousness reasoning]
             METAL: The coin appears to be silver...
             OBVERSE: ..."
        The PDF should show only the six structured sections, not the reasoning.

    Strategy:
        Find the first section label at the start of a line (METAL:, OBVERSE:,
        etc.) and return everything from that point.  If no section label is
        found the full text is returned unchanged (safe fallback).
    """
    import re
    m = re.search(
        r"^(METAL|OBVERSE|REVERSE|INSCRIPTIONS|CONDITION|IDENTIFICATION):",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return text[m.start():].strip() if m else text


def _strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks emitted by reasoning models (qwen3-vl, etc.).

    WHAT: Uses a regex to strip everything between <think> and </think> tags,
          including the tags themselves, then strips surrounding whitespace.

    WHY: qwen3-vl in its default configuration outputs its internal chain-of-
         thought before the structured answer. Example output shape:
             <think>
             The coin appears to be... let me think step by step...
             </think>
             1. METAL/MATERIAL: silver...
         Only the part AFTER </think> is the actual numismatic analysis.
         If the tags are absent (e.g. Gemini, gemma3), this function is a no-op.

    WHY remove rather than extract the thinking section:
         The thinking content is speculative mid-reasoning text. Storing it in
         the report or passing it as a RAG query string would add noise.
         The structured answer (after </think>) is what the Synthesiser uses.
    """
    import re
    # Strip <think>...</think> blocks (non-greedy, handles multi-line)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def _opencv_fallback(image_path: str) -> tuple[str, dict]:
    """
    Local OpenCV coin analysis — runs when no vision LLM is configured.

    WHAT: Performs three analyses on the coin image:
      1. HSV histogram on the central 60% crop → infer metal color
         (silver: low saturation / gold: warm hue 15-35 / bronze: hue 5-25)
      2. Sobel edge density on greyscale → estimate condition
         (high edge density = well-preserved details; low = worn/corroded)
      3. Overall brightness and saturation stats → support metal inference

    WHY this ordering:
      Metal color is the most valuable attribute for KB cross-referencing.
      It immediately narrows the search space: silver vs bronze vs gold.
      Edge density tells us coin condition, which affects attribution.

    WHY central 60% crop:
      Coin edges are often dark (patina, mounting damage, photography shadow).
      The central region contains the primary design and most informative pixels.

    Returns (description_str, features_dict).
    """
    import cv2
    import numpy as np

    features = _empty_features()
    lines    = []

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cv2.imread returned None for {image_path}")

        h, w = img.shape[:2]
        # Central 60% crop
        y0, y1 = int(h * 0.20), int(h * 0.80)
        x0, x1 = int(w * 0.20), int(w * 0.80)
        crop = img[y0:y1, x0:x1]

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        H, S, V = hsv[:, 0].astype(float), hsv[:, 1].astype(float), hsv[:, 2].astype(float)

        mean_s = float(S.mean())
        mean_v = float(V.mean())
        mean_h = float(H.mean())

        # ── Metal detection ───────────────────────────────────────────────────
        gold_mask   = ((H >= 15) & (H <= 35) & (S >= 80)).sum() / len(H)
        bronze_mask = ((H >=  5) & (H <= 25) & (S >= 50)).sum() / len(H)
        silver_mask = (S < 40).sum() / len(H)

        THRESHOLD = 0.15   # at least 15% of pixels must satisfy the mask
        if gold_mask > THRESHOLD:
            metal = "gold"
        elif bronze_mask > THRESHOLD and bronze_mask > silver_mask:
            metal = "bronze"
        elif silver_mask > THRESHOLD:
            metal = "silver"
        else:
            metal = "unknown metal"

        features["metal_color"] = metal
        lines.append(
            f"METAL/MATERIAL: Pixel analysis (HSV) suggests this is a {metal} coin "
            f"(mean saturation {mean_s:.0f}/255, mean brightness {mean_v:.0f}/255)."
        )

        # ── Condition via Sobel edge density ──────────────────────────────────
        grey       = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sobelx     = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
        sobely     = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag   = np.sqrt(sobelx**2 + sobely**2)
        edge_score = float(edge_mag.mean())

        if edge_score > 25:
            condition = "well-preserved"
            cond_note = "high edge density suggests clear detail preservation"
        elif edge_score > 12:
            condition = "worn"
            cond_note = "moderate edge density suggests partial wear"
        else:
            condition = "corroded"
            cond_note = "low edge density suggests heavy wear or corrosion"

        features["condition"] = condition
        lines.append(
            f"CONDITION: {condition.capitalize()} "
            f"(Sobel edge score {edge_score:.1f} — {cond_note})."
        )

        lines.append(
            "NOTE: This description was generated by local OpenCV analysis (no vision LLM "
            "configured). For a detailed numismatic description, set OLLAMA_HOST and run "
            "'ollama pull qwen3-vl:4b'."
        )

    except Exception as e:
        lines.append(f"OpenCV analysis failed: {e}")

    return " ".join(lines), features

def _parse_features(description: str) -> dict:
    """
    Extract structured attributes from the VLM plain-prose response.

    WHY this approach (versus JSON from the LLM):
        Plain prose paragraphs labelled METAL:/INSCRIPTIONS: are more
        robust because the structure is in the label, not JSON syntax.

    WHY scope inscription extraction to the INSCRIPTIONS section:
        The old regex r"[A-ZA-O]{3,}" searched the ENTIRE description,
        so section labels like METAL, OBVERSE, BCE appeared as inscriptions.
        Restricting the search to after the INSCRIPTIONS: label eliminates
        false positives from the prose structure itself.
    """
    import re

    def _section(label: str) -> str:
        """Extract text of a labeled section up to the next label or end."""
        m = re.search(
            rf"^{label}:\s*(.+?)(?=^[A-Z]{{3,}}:|\Z)",
            description,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        return m.group(1).strip() if m else ""

    # Metal -- prefer section-scoped match, fall back to full-text keyword scan
    metal = "unknown"
    metal_text = _section("METAL") or description
    for m in ("bronze", "gold", "electrum", "billon", "copper", "silver"):
        if m in metal_text.lower():
            metal = m
            break

    # Inscriptions -- only parse within the INSCRIPTIONS section so that
    # section headers (METAL, OBVERSE) are never mis-classified as legends.
    inscriptions: list = []
    ins_section = _section("INSCRIPTIONS")
    if ins_section and ins_section.lower() not in ("none", "none visible", "n/a", "unknown"):
        raw = re.findall(r"[A-ZΑ-Ω]{2,}", ins_section)
        _STOP = {"THE", "AND", "FOR", "NOT", "BUT", "WITH", "FROM", "NONE",
                 "ANY", "ALL", "ARE", "HAS", "ITS", "CAN", "WAY", "MAY"}
        inscriptions = [w for w in dict.fromkeys(raw) if w not in _STOP][:6]

    # Condition
    condition = "unknown"
    cond_text = _section("CONDITION") or description
    for cond in ("well-preserved", "well preserved", "worn", "corroded", "good", "fair", "poor"):
        if cond in cond_text.lower():
            condition = cond
            break

    # Profile direction -- look in OBVERSE section
    profile = "unknown"
    obv_text = _section("OBVERSE")
    m_dir = re.search(r"(?:facing|portrait|head|turned)\s+(left|right)", obv_text, re.I)
    if m_dir:
        profile = m_dir.group(1).lower()

    return {
        "metal_color":       metal,
        "profile_direction": profile,
        "inscriptions":      inscriptions,
        "symbols":           [],
        "condition":         condition,
    }
