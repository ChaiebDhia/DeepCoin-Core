"""
src/agents/synthesis.py
========================
Layer 3 — Synthesis Agent

Produces two outputs from the LangGraph state:
  1. A structured plain-text string  (API / logs)
  2. A professional enterprise-grade PDF (fpdf2, direct draw — no Markdown)

PDF design:
  - No Markdown syntax characters (* # ` [ ] -)
  - No ASCII art
  - Proper bordered tables with alternating row shading
  - Branded navy header band + timestamp
  - Section titles as blue-ruled uppercase headings
  - Confidence as percentage only
  - Greek / non-latin chars replaced gracefully (latin-1 font limitation)
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime


# ── colour palette (R, G, B) ──────────────────────────────────────────────────
_C_BRAND_DARK  = (15,  40,  80)    # deep navy  — header/footer band
_C_BRAND_MID   = (30,  80, 160)    # mid blue   — section rule lines
_C_BRAND_LIGHT = (220, 230, 245)   # pale blue  — table header row bg
_C_ROW_ALT     = (245, 247, 250)   # near-white — alternating table rows
_C_TEXT        = (30,  30,  30)    # near-black — body text
_C_MUTED       = (110, 110, 110)   # grey       — secondary / label text
_C_WHITE       = (255, 255, 255)
_C_GREEN       = (30,  130,  80)   # high confidence
_C_AMBER       = (180, 120,   0)   # medium confidence
_C_RED_DK      = (160,  30,  30)   # low confidence
_C_ORANGE      = (200, 100,  20)   # forensic warning
_C_RULE        = (200, 210, 225)   # light border lines

# ── Route display labels ───────────────────────────────────────────────────────
_ROUTE_LABELS = {
    "historian":    "Historical Analysis",
    "validator":    "Forensic Validation",
    "investigator": "Visual Investigation",
}


# ── Text-cleaning patterns ───────────────────────────────────────────────────
# Matches [CONTEXT 1], [CONTEXT 2], [CONTEXT CNN], [CONTEXT N], etc.
# These are internal RAG citation markers injected into the LLM prompt so that
# Gemini can produce grounded answers.  They must NEVER appear in the final PDF.
_RE_CONTEXT = re.compile(r"\[CONTEXT\s*(?:\d+|CNN|N)?\s*(?:—[^\]]*)?\]", re.I)

# Markdown patterns produced by some LLM responses when the model ignores the
# "plain prose only" instruction.  Order matters — bold before italic.
_MD_PATTERNS = [
    (re.compile(r"\*{3}(.+?)\*{3}"),    r"\1"),   # ***bold-italic*** → text
    (re.compile(r"\*{2}(.+?)\*{2}"),    r"\1"),   # **bold** → text
    (re.compile(r"\*(.+?)\*"),          r"\1"),   # *italic* → text
    (re.compile(r"_{2}(.+?)_{2}"),      r"\1"),   # __bold__ → text
    (re.compile(r"_(.+?)_"),            r"\1"),   # _italic_ → text
    (re.compile(r"`{1,3}(.+?)`{1,3}"), r"\1"),   # `code` / ```code``` → text
    (re.compile(r"(?m)^\s*#{1,6}\s+"),  ""),     # ## Heading at line-start → stripped
    (re.compile(r"#{2,}\s*"),           ""),     # inline ## / #### (LLM artefact)
]

# Greek → Latin transliteration (dict-based)
_GREEK_MAP: dict = {
    "Α":"A",  "Β":"B",  "Γ":"G",  "Δ":"D",  "Ε":"E",  "Ζ":"Z",
    "Η":"E",  "Θ":"TH", "Ι":"I",  "Κ":"K",  "Λ":"L",  "Μ":"M",
    "Ν":"N",  "Ξ":"X",  "Ο":"O",  "Π":"P",  "Ρ":"R",  "Σ":"S",
    "Τ":"T",  "Υ":"Y",  "Φ":"PH", "Χ":"KH", "Ψ":"PS", "Ω":"O",
    "α":"a",  "β":"b",  "γ":"g",  "δ":"d",  "ε":"e",  "ζ":"z",
    "η":"e",  "θ":"th", "ι":"i",  "κ":"k",  "λ":"l",  "μ":"m",
    "ν":"n",  "ξ":"x",  "ο":"o",  "π":"p",  "ρ":"r",  "σ":"s",
    "ς":"s",  "τ":"t",  "υ":"y",  "φ":"ph", "χ":"kh", "ψ":"ps",
    "ω":"o",
}

# Typographic characters that NFD decomposition cannot resolve to ASCII.
# These appear frequently in LLM output (curly quotes, dashes, ligatures)
# and would silently become '?' via latin-1 encode("replace").
# Mapped explicitly BEFORE the NFD step so the result is always readable.
_TYPO_MAP: dict = {
    "\u2018": "'",   # LEFT  SINGLE QUOTATION MARK  '
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK  '  ← commonest source of '?'
    "\u201A": ",",   # SINGLE LOW-9 QUOTATION MARK  ‚
    "\u201C": '"',   # LEFT  DOUBLE QUOTATION MARK  "
    "\u201D": '"',   # RIGHT DOUBLE QUOTATION MARK  "
    "\u201E": '"',   # DOUBLE LOW-9 QUOTATION MARK  „
    "\u2013": "-",   # EN DASH                       –
    "\u2014": "-",   # EM DASH                       —
    "\u2015": "-",   # HORIZONTAL BAR                ―
    "\u2026": "...", # HORIZONTAL ELLIPSIS           …
    "\u00DF": "ss",  # LATIN SMALL LETTER SHARP S    ß  (German: Geiß → Geiss)
    "\u00C6": "AE",  # LATIN CAPITAL LIGATURE AE     Æ
    "\u00E6": "ae",  # LATIN SMALL   LIGATURE AE     æ
    "\u0152": "OE",  # LATIN CAPITAL LIGATURE OE     Œ
    "\u0153": "oe",  # LATIN SMALL   LIGATURE OE     œ
    "\u00D8": "O",   # LATIN CAPITAL LETTER O STROKE Ø
    "\u00F8": "o",   # LATIN SMALL   LETTER O STROKE ø
    "\u00D0": "D",   # LATIN CAPITAL LETTER ETH      Ð
    "\u00F0": "d",   # LATIN SMALL   LETTER ETH      ð
    "\u00DE": "TH",  # LATIN CAPITAL LETTER THORN    Þ
    "\u00FE": "th",  # LATIN SMALL   LETTER THORN    þ
}


def _s(text: str) -> str:
    """
    Sanitise a string for PDF output through six ordered steps.

    Step 1 — Strip RAG citation markers  ([CONTEXT 1], [CONTEXT CNN], …)
             These are internal prompt tokens that must never appear in print.
    Step 2 — Strip Markdown syntax  (**, *, ##, `code`, …)
             Some LLM responses contain inline Markdown even when instructed
             not to.  We extract just the visible text between the markers.
    Step 3 — Collapse excess whitespace left by stripping.
    Step 4 — Typographic character normalisation (_TYPO_MAP)
             Curly quotes, em/en dashes, ligatures, sharp-s, etc. that NFD
             decomposition cannot resolve to latin-1.  These are the most
             frequent source of '?' in LLM-generated text.
    Step 5 — Greek → Latin transliteration  (Σ→S, Α→A, …)
             fpdf2's built-in fonts use latin-1 encoding, which excludes the
             Greek Unicode block (U+0370–U+03FF).
    Step 6 — Unicode NFD decomposition + combining-mark removal
             Converts accented characters (ō, é, ñ, …) to their ASCII base
             letter by decomposing to NFD form and stripping combining marks
             (Unicode category 'Mn').  Handles the long tail of accented
             Latin chars not covered by _TYPO_MAP.
    Step 7 — latin-1 encode/decode  (final safety net for any remaining
             characters outside the font's supported range).
    """
    t = str(text)

    # Step 1 — remove [CONTEXT N] / [CONTEXT CNN] citation markers
    t = _RE_CONTEXT.sub("", t)

    # Step 2 — strip Markdown formatting, keep visible text
    for pattern, repl in _MD_PATTERNS:
        t = pattern.sub(repl, t)

    # Step 3a — strip Corpus Nummorum HTML navigation artefacts
    #   corpus-nummorum.eu embeds "go to the NLP result of this description"
    #   links and "| legend Design" / "| legend Legend" section labels as
    #   inline text in the scraped HTML.  These are never numismatic content.
    t = re.sub(r"go to the NLP result of this description", "", t, flags=re.I)
    t = re.sub(r"\|\s*legend\s+(Design|Legend)\s*", " / ", t, flags=re.I)
    t = re.sub(r"^legend\s+(Design|Legend)\s+", "", t, flags=re.I)  # at field start
    t = re.sub(r"\s*\|\s*Legend:\s*", " / Legend: ", t)  # | Legend: x -> / Legend: x
    t = re.sub(r"\s*\|\s*$", "", t)  # trailing " | " left after stripping

    # Step 3b — normalise German date notation from corpus-nummorum.eu
    #   The KB stores dates in German: "500-400 v.Chr." = "500-400 BC"
    t = re.sub(r"\bv\.\s*Chr\.", "BC",  t)
    t = re.sub(r"\bn\.\s*Chr\.", "AD",  t)

    # Step 3b — collapse runs of spaces / clean up leftover punctuation gaps
    t = re.sub(r"  +", " ", t).strip()

    # Step 4 — typographic character normalisation
    t = "".join(_TYPO_MAP.get(c, c) for c in t)

    # Step 5 — Greek transliteration
    t = "".join(_GREEK_MAP.get(c, c) for c in t)

    # Step 6 — decompose accented chars, strip combining diacritics
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")

    # Step 7 — final latin-1 safety net
    return t.encode("latin-1", "replace").decode("latin-1")


def _safe(text: str) -> str:
    """
    Minimal latin-1 safety encode for filenames, timestamps, and other
    system-generated strings that contain NO Markdown and NO Greek.

    WHY NOT use _s() here:
        _s() applies Markdown-stripping patterns including _italic_ which
        matches underscores in filenames like 'CN_type_1015_cn_coin.jpg',
        silently eating the underscores and producing 'CNtype1015cncoin.jpg'.
        _safe() skips all Markdown processing — it only ensures the string
        is latin-1 encodeable so fpdf2 can render it without crashing.
    """
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _conf_color(conf: float) -> tuple:
    """
    Return an RGB colour tuple for a confidence value.

    Thresholds mirror the Gatekeeper routing thresholds so the badge colour
    matches the routing decision intuitively:
        Green  (> 85%)  — high confidence, historian route
        Amber  (40-85%) — medium confidence, validator route
        Red    (< 40%)  — low confidence, investigator route
    """
    if conf > 0.85:
        return _C_GREEN
    if conf > 0.40:
        return _C_AMBER
    return _C_RED_DK


def _enrich_label(type_id, include_date: bool = False) -> str:
    """
    Return a user-friendly coin description for a CN type ID.

    Format (no date):   "Material Denomination - Mint"
    Format (with date): "Material Denomination - Mint, Date"
    Example: "Silver Drachm - Maroneia, c.365-330 BC"

    WHAT: Combines the material, denomination, mint (and optionally date) from
          the RAG knowledge base into a single readable string.

    WHY no date in the stripe label:
        The result stripe uses a 50-char truncation limit for the coin name.
        Adding the date would overflow it for most coins.  The date is only
        appended when include_date=True, which is used by the top-5 table to
        differentiate coins of the same denomination from the same mint
        (e.g. CN 1015, 1017, 864 are all Silver Drachm - Maroneia but with
        different date ranges).

    WHY: The raw CN type number (e.g. 532, 1015) is an opaque database key
         that means nothing to a museum visitor or researcher reading the PDF.
         An enriched string gives an immediate, self-explanatory description.

    HOW: Calls get_rag_engine().get_by_id() which is an in-memory dict lookup
         (zero I/O, sub-millisecond).  Falls back gracefully to "CN {type_id}".
    """
    try:
        from src.core.rag_engine import get_rag_engine
        rec = get_rag_engine().get_by_id(int(type_id))
        if not rec:
            return f"CN {type_id}"
        mat   = (rec.get("material",     "") or "").strip().title()
        denom = (rec.get("denomination", "") or "").strip().title()
        mint  = (rec.get("mint",         "") or "").strip()
        date  = (rec.get("date",         "") or "").strip()
        # Strip parenthetical qualifiers from denom: "Large Denomination (Bronze)" -> "Large Denomination"
        import re as _re
        denom = _re.sub(r'\s*\([^)]*\)', '', denom).strip()
        # Strip archaeological period appended to date: "c. 500-450 BC Archaic Period" -> "c. 500-450 BC"
        date = _re.sub(
            r'\s+(Archaic|Classical|Hellenistic|Roman|Byzantine|Early|Late|Middle)\b.*',
            '', date, flags=_re.IGNORECASE).strip()
        # Filter denominations that are scraped field names, not real values
        _BAD_DENOMS = {"material", "type", "region", "date", "mint",
                       "period", "denomination", "weight", "diameter",
                       "obverse", "reverse", "legend", "authority"}
        # Before filtering, rescue metal name hidden in compound bad-denom
        # e.g. denom="Material bronze" -> mat empty -> extract "Bronze" from denom
        _METAL_WORDS = {"bronze", "silver", "gold", "electrum", "billon", "copper", "lead"}
        _denom_words = denom.lower().split()
        if not mat and _denom_words:
            for _w in _denom_words:
                if _w in _METAL_WORDS:
                    mat = _w.title()
                    break
        # Also catch compound denominations whose first word is a bad key
        # e.g. "Material Bronze" -> split()[0] == "material" -> filter out
        if denom.lower() in _BAD_DENOMS or (_denom_words and _denom_words[0] in _BAD_DENOMS):
            denom = ""
        parts = " ".join(p for p in (mat, denom) if p)
        base  = f"{parts} - {mint}" if (parts and mint) else (parts or mint or f"CN {type_id}")
        if include_date and date and len(date) <= 30:
            return f"{base}, {date}"
        return base
    except Exception:
        return f"CN {type_id}"


def _basename(path: str) -> str:
    """
    Return just the original filename from a path, stripping the UUID prefix.

    WHY strip UUID:
        classify.py saves uploads as '{uuid}_{original_filename}' to prevent
        name collisions.  The UUID is internal bookkeeping — the header of the
        PDF should show the human-readable original filename, not a 36-char
        identifier that means nothing to a reader.

    UUID format: 8-4-4-4-12 hex chars + underscore = 37 leading characters.
    Example: '2240d431-f93c-4fc1-b8b9-96fece4bab9d_coin.jpg' -> 'coin.jpg'
    """
    import re as _re
    name = path.replace("\\", "/").split("/")[-1] if path else "N/A"
    # Strip UUID prefix: exactly 36 hex+dash chars followed by underscore
    name = _re.sub(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_", "", name)
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════════

class Synthesis:
    """
    Synthesis agent — assembles the final report from LangGraph state.

    State keys used (all optional):
        image_path          : str
        cnn_prediction      : dict  {label, confidence, top5, tta_used}
        route_taken         : str   historian | validator | investigator
        historian_result    : dict
        validator_result    : dict
        investigator_result : dict
    """

    # ── public ────────────────────────────────────────────────────────────────

    def synthesize(self, state: dict) -> str:
        """Returns a structured plain-text summary (no Markdown, no special chars)."""
        cnn   = state.get("cnn_prediction", {})
        route = state.get("route_taken", "unknown")
        h     = state.get("historian_result", {})
        ts    = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        img   = _basename(state.get("image_path", ""))
        label = cnn.get("label", "N/A")
        conf  = cnn.get("confidence", 0.0)

        lines = [
            "=" * 62,
            "  DeepCoin  —  Numismatic Classification Report",
            f"  Generated : {ts}",
            "=" * 62,
            "",
            f"  Image          : {img}",
            f"  CN Type        : {label}",
            f"  Confidence     : {conf:.1%}",
            f"  Pipeline Route : {route.upper()}",
            "",
        ]

        if h:
            lines += [
                "  HISTORICAL RECORD",
                "  " + "-" * 44,
                f"  Mint           : {h.get('mint', '')}",
                f"  Region         : {h.get('region', '')}",
                f"  Date           : {h.get('date', '')}",
                f"  Period         : {h.get('period', '')}",
                f"  Material       : {h.get('material', '')}",
                f"  Denomination   : {h.get('denomination', '')}",
                f"  Obverse        : {h.get('obverse', '')}",
                f"  Reverse        : {h.get('reverse', '')}",
                f"  Persons        : {h.get('persons', '')}",
                "",
            ]
            if h.get("narrative"):
                lines += ["  Expert Commentary:", f"  {h['narrative']}", ""]
            if h.get("source_url"):
                lines += [f"  Source: {h['source_url']}", ""]

        lines += [
            "=" * 62,
            "  DeepCoin-Core  |  ESPRIT School of Engineering  |  YEBNI",
            "  corpus-nummorum.eu",
            "=" * 62,
        ]
        return "\n".join(lines)

    def to_pdf(self, state: dict, output_path: str) -> None:
        """
        Build an enterprise-grade PDF directly from the LangGraph state dict.
        All drawing is explicit — zero Markdown parsing.
        """
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos

        cnn   = state.get("cnn_prediction", {})
        route = state.get("route_taken", "unknown")
        h     = state.get("historian_result", {})
        v     = state.get("validator_result", {})
        inv   = state.get("investigator_result", {})
        img   = _basename(state.get("image_path", ""))
        ts    = datetime.now().strftime("%d %B %Y    %H:%M")

        # ── page setup ────────────────────────────────────────────────────────
        # Use _PDF subclass so every page automatically gets the branded footer
        # band with page number — no manual footer call needed.
        f = _PDF()
        f.set_margins(20, 15, 20)
        f.set_auto_page_break(auto=True, margin=20)
        f.add_page()

        # ── header band ───────────────────────────────────────────────────────
        _draw_header_band(f, ts, img)

        # ── result summary stripe ─────────────────────────────────────────────
        _draw_result_stripe(f, cnn, route)
        f.ln(6)

        # ── CNN classification ────────────────────────────────────────────────
        _section_title(f, "CNN Classification")
        _kv_table(f, [
            ("Best Match",      _s(_enrich_label(str(cnn.get("label", "N/A"))))),
            ("Confidence",      f"{cnn.get('confidence', 0):.1%}"),
            ("Model",           "EfficientNet-B3  (438 classes)"),
            ("Analysis Route",  _ROUTE_LABELS.get(route, route.upper())),
            ("TTA Applied",     "Yes" if cnn.get("tta_used") else "No"),
        ])

        top5 = cnn.get("top5", [])
        if top5:
            f.ln(4)
            _subsection_title(f, "Top-5 Predictions")
            _confidence_table(f, top5)
        f.ln(7)

        # ── historical record ─────────────────────────────────────────────────
        if h:
            _section_title(f, "Historical Record")
            rows = [
                ("Mint",          h.get("mint",           "")),
                ("Region",        h.get("region",         "")),
                ("Date",          h.get("date",           "")),
                ("Period",        h.get("period",         "")),
                ("Material",      h.get("material",       "")),
                ("Denomination",  h.get("denomination",   "")),
                ("Obverse",       h.get("obverse",        "")),
                ("Reverse",       h.get("reverse",        "")),
                ("Persons",       h.get("persons",        "")),
                ("CN Reference",  str(h.get("type_id",    ""))),
            ]
            _kv_table(f, [(k, _s(val)) for k, val in rows if val])

            if h.get("narrative"):
                f.ln(4)
                _subsection_title(f, "Expert Commentary")
                _body_paragraph(f, h["narrative"])

            if h.get("source_url"):
                f.ln(2)
                _source_line(f, h["source_url"])
            f.ln(7)

        # ── forensic validation ───────────────────────────────────────────────
        if v:
            _section_title(f, "Forensic Validation")
            _status_badge(f, v.get("match", True))
            _kv_table(f, [
                ("Status",            _s(v.get("status", "").upper())),
                ("Detected Material", _s(v.get("detected_material", ""))),
                ("Expected Material", _s(v.get("expected_material", ""))),
            ])
            if v.get("warning"):
                f.ln(3)
                _warning_box(f, v["warning"])
            f.ln(7)

        # ── visual investigation ──────────────────────────────────────────────
        if inv:
            _section_title(f, "Visual Investigation")
            if inv.get("visual_description"):
                # Trim verbose pre-analysis preamble: show only the structured
                # section text (METAL: / OBVERSE: / etc.) without the LLM's
                # internal reasoning that precedes the actual answer.
                vis = _trim_to_sections(inv["visual_description"])
                _body_paragraph(f, vis)

            feats = inv.get("detected_features", {})
            if feats:
                f.ln(3)
                _subsection_title(f, "Detected Attributes")
                _kv_table(f, [
                    (k.replace("_", " ").title(), _s(str(val)))
                    for k, val in feats.items()
                    if val and val not in ("unknown", [], "")
                ])

            kb = inv.get("kb_matches", [])
            if kb:
                f.ln(3)
                _subsection_title(f, "Closest Knowledge Base Matches")
                _kb_table(f, kb[:3])

            if inv.get("suggested_type_id"):
                f.ln(3)
                tid  = inv['suggested_type_id']
                name = _s(_enrich_label(tid))
                _info_box(f, f"Best visual match: {name}  (CN {tid})")
            f.ln(5)

        # ── save ──────────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as fout:
            f.output(fout)


# ═══════════════════════════════════════════════════════════════════════════════
# FPDF subclass — auto branded footer on every page
# ═══════════════════════════════════════════════════════════════════════════════

class _PDF:
    """
    Thin wrapper that defers FPDF construction until first use so that the
    fpdf import stays inside to_pdf() (avoids import-time cost).

    WHY subclass rather than monkey-patch:
        fpdf2's FPDF.footer() is called automatically after every add_page().
        Overriding it in a subclass is the canonical fpdf2 pattern for
        auto-repeating page elements (headers, footers, page numbers).

    Footer design:
        Navy band (14 mm) at the bottom of every page.
        Left: DeepCoin branding
        Right: Page N / Total  (requires two-pass: AliasNbPages)
    """

    def __new__(cls):
        """Return a real FPDF subclass instance on construction."""
        from fpdf import FPDF

        class DeepCoinPDF(FPDF):
            def footer(self_pdf):
                from fpdf.enums import XPos, YPos
                self_pdf.set_y(-(14))
                self_pdf.set_fill_color(*_C_BRAND_DARK)
                self_pdf.rect(0, self_pdf.h - 14, self_pdf.w, 14, style="F")
                self_pdf.set_xy(20, self_pdf.h - 10)
                self_pdf.set_text_color(180, 200, 230)
                self_pdf.set_font("Helvetica", "", 8)
                self_pdf.cell(110, 5,
                    "DeepCoin-Core  |  Dhia Chaieb  |  ESPRIT School of Engineering  |  YEBNI")
                self_pdf.set_xy(self_pdf.w - 65, self_pdf.h - 10)
                # {nb} is replaced by fpdf2 with the total page count
                self_pdf.cell(45, 5,
                    f"Page {self_pdf.page_no()}/{{nb}}",
                    align="R")
                self_pdf.set_text_color(*_C_TEXT)

        pdf = DeepCoinPDF()
        pdf.alias_nb_pages("{nb}")   # two-pass total-page substitution
        return pdf


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ew(f) -> float:
    """Effective (printable) page width in mm."""
    return f.w - f.l_margin - f.r_margin


def _draw_header_band(f, ts: str, img: str) -> None:
    from fpdf.enums import XPos, YPos
    # Navy band
    f.set_fill_color(*_C_BRAND_DARK)
    f.rect(0, 0, f.w, 30, style="F")

    # Left: product name
    f.set_xy(20, 5)
    f.set_text_color(*_C_WHITE)
    f.set_font("Helvetica", "B", 18)
    f.cell(90, 10, "DeepCoin", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    f.set_xy(20, 16)
    f.set_font("Helvetica", "", 9)
    f.set_text_color(180, 200, 230)
    f.cell(90, 6, "Numismatic Intelligence System  |  ESPRIT / YEBNI")

    # Attribution line — author, email, internship context
    f.set_xy(20, 23)
    f.set_font("Helvetica", "I", 7)
    f.set_text_color(150, 175, 215)
    f.cell(80, 5, "Prepared by: Dhia Chaieb")

    # Right: timestamp + filename
    # WHY _safe() not _s() here:
    #   _s() applies _italic_ Markdown stripping which eats underscores in
    #   filenames like 'CN_type_1015.jpg' → 'CNtype1015.jpg'.  _safe() only
    #   ensures latin-1 compatibility without touching the content.
    f.set_font("Helvetica", "", 8)
    f.set_text_color(180, 200, 230)
    f.set_xy(f.w - 85, 8)
    f.cell(65, 5, _safe(ts), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_xy(f.w - 85, 14)
    f.cell(65, 5, _safe(img), align="R")

    f.set_text_color(*_C_TEXT)
    f.set_xy(f.l_margin, 36)


def _draw_result_stripe(f, cnn: dict, route: str) -> None:
    """
    Horizontal summary stripe beneath the header band.

    Layout (left → right):
        CN Type NNN             [Confidence pill]    Route label

    WHY a coloured pill for confidence:
        The colour immediately communicates the reliability tier —
        green (>85%), amber (40-85%), red (<40%) — matching the Gatekeeper
        routing thresholds so the badge is self-explanatory to a reader who
        knows the system.
    """
    from fpdf.enums import XPos, YPos
    label      = str(cnn.get("label", "N/A"))
    conf       = cnn.get("confidence", 0.0)
    conf_color = _conf_color(conf)
    route_lbl  = _safe(_ROUTE_LABELS.get(route, route.upper()))

    # Human-readable coin identity from KB  (e.g. "Silver Drachm — Maroneia")
    human_name = _safe(_enrich_label(label))

    y  = f.get_y()
    ew = _ew(f)

    # Background stripe — 22 mm tall to fit coin name + sub-label on two lines
    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_BRAND_LIGHT)
    f.rect(f.l_margin, y, ew, 22, style="FD")

    # Line 1 — coin name (or "Unclassified Specimen" for low-confidence route)
    f.set_xy(f.l_margin + 5, y + 3)
    f.set_font("Helvetica", "B", 11)
    f.set_text_color(*_C_BRAND_DARK)
    if conf < 0.40:
        f.cell(86, 8, "Unclassified Specimen")
    else:
        name = human_name[:50] + ("..." if len(human_name) > 50 else "")
        f.cell(86, 8, name)

    # Line 2 — scholarly reference sub-label
    f.set_xy(f.l_margin + 5, y + 12)
    f.set_font("Helvetica", "", 8)
    f.set_text_color(*_C_MUTED)
    if conf < 0.40:
        f.cell(86, 6, f"Best candidate: CN {label}")
    else:
        f.cell(86, 6, f"Corpus Nummorum \xb7 CN {label}")

    # Confidence pill — coloured filled rectangle + white text
    pill_w, pill_h = 46, 8
    pill_x = f.l_margin + 95
    pill_y = y + 7
    f.set_fill_color(*conf_color)
    f.set_draw_color(*conf_color)
    f.rect(pill_x, pill_y, pill_w, pill_h, style="FD")
    f.set_xy(pill_x, pill_y)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_WHITE)
    f.cell(pill_w, pill_h, f"  Confidence: {conf:.1%}", align="L")

    # Route label — right aligned
    f.set_xy(f.l_margin + 147, y + 8)
    f.set_font("Helvetica", "", 9)
    f.set_text_color(*_C_MUTED)
    f.cell(ew - 147, 7, route_lbl, align="R")

    f.set_text_color(*_C_TEXT)
    f.set_draw_color(*_C_RULE)
    f.set_xy(f.l_margin, y + 22)


def _section_title(f, title: str) -> None:
    from fpdf.enums import XPos, YPos
    f.set_x(f.l_margin)
    f.set_font("Helvetica", "B", 11)
    f.set_text_color(*_C_BRAND_DARK)
    f.cell(0, 7, _s(title.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # Blue rule
    y = f.get_y()
    f.set_draw_color(*_C_BRAND_MID)
    f.line(f.l_margin, y, f.l_margin + _ew(f), y)
    f.set_draw_color(*_C_RULE)
    f.set_text_color(*_C_TEXT)
    f.ln(3)


def _subsection_title(f, title: str) -> None:
    from fpdf.enums import XPos, YPos
    f.set_x(f.l_margin)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_MUTED)
    f.cell(0, 6, _s(title.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_text_color(*_C_TEXT)
    f.ln(1)


def _kv_table(f, rows: list) -> None:
    """
    Two-column key/value table with borders and alternating row shading.

    WHY the previous approach caused a staircase:
        The key column used `cell()` (fixed height = row_h), while the value
        column used `multi_cell()` (variable height for long descriptions).
        When a value wrapped to 3 lines the value column was 21 mm tall but
        the key column border was only 7 mm — leaving 14 mm unbordered.

    Fix strategy:
        1. Measure the value height *before* drawing using `dry_run=True`
           with `output="LINES"` — this gives the exact wrapped line count
           without touching the page.
        2. Compute total_h = max(key_lines, val_lines) × row_h.
        3. Draw the key column as a solid rectangle (background + border)
           spanning the full total_h, then draw the text on top.
        4. Do the same for the value column.
        5. Advance the cursor to start_y + total_h.

    Column widths:
        col_k = 38 mm  (~22 % of 170 mm effective width)
        col_v = 132 mm (~78 %)  — generous space for long descriptions.
    """
    from fpdf.enums import XPos, YPos
    if not rows:
        return

    col_k = 38                     # key column — was 52, narrowed to give value more room
    col_v = _ew(f) - col_k         # value column — typically ~132 mm on A4
    row_h = 6                      # line-height per text line inside a cell
    hdr_h = 7                      # header row is slightly taller

    # ── Column header ─────────────────────────────────────────────────────────
    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_RULE)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_BRAND_DARK)
    f.set_x(f.l_margin)
    f.cell(col_k, hdr_h, "  Field",  border=1, fill=True,
           new_x=XPos.RIGHT, new_y=YPos.TOP)
    f.cell(col_v, hdr_h, "  Value",  border=1, fill=True,
           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for i, (key, val) in enumerate(rows):
        fill_color = _C_ROW_ALT if i % 2 == 0 else _C_WHITE
        key_text = f"  {_s(key)}"
        val_text = f"  {_s(val)}"

        # ── Step 1: measure wrapped line counts (dry run — no page output) ───
        f.set_font("Helvetica", "B", 9)
        key_lines = f.multi_cell(col_k, row_h, key_text,
                                 dry_run=True, output="LINES")
        f.set_font("Helvetica", "", 9)
        val_lines = f.multi_cell(col_v, row_h, val_text,
                                 dry_run=True, output="LINES")

        n_lines  = max(len(key_lines), len(val_lines), 1)
        total_h  = n_lines * row_h
        # PREVENT mid-row page breaks: add a page NOW if the row won't fit.
        # 14 mm is reserved for the branded footer band at the bottom.
        if f.get_y() + total_h > f.h - f.b_margin - 14:
            f.add_page()
        start_y  = f.get_y()

        # ── Step 2: key column — full-height rectangle then text ──────────────
        f.set_fill_color(*fill_color)
        f.set_draw_color(*_C_RULE)
        # Draw filled, bordered rectangle spanning the full row height
        f.rect(f.l_margin, start_y, col_k, total_h, style="FD")
        # Overlay text (no border / fill — rect already covers it)
        f.set_font("Helvetica", "B", 9)
        f.set_text_color(*_C_MUTED)
        f.set_xy(f.l_margin, start_y)
        f.multi_cell(col_k, row_h, key_text, border=0, fill=False,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # ── Step 3: value column — same approach ──────────────────────────────
        f.set_fill_color(*fill_color)
        f.set_draw_color(*_C_RULE)
        f.rect(f.l_margin + col_k, start_y, col_v, total_h, style="FD")
        f.set_font("Helvetica", "", 9)
        f.set_text_color(*_C_TEXT)
        f.set_xy(f.l_margin + col_k, start_y)
        f.multi_cell(col_v, row_h, val_text, border=0, fill=False,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # ── Step 4: advance cursor to end of row ──────────────────────────────
        f.set_xy(f.l_margin, start_y + total_h)

    f.set_text_color(*_C_TEXT)
    f.set_draw_color(*_C_RULE)


def _confidence_table(f, top5: list) -> None:
    """
    Rank / CN Type / Coin Description / Confidence four-column table.

    WHY four columns vs the original three:
        The old "CN Type" column showed raw database numbers (532, 1015)
        that mean nothing to a reader.  Adding "Coin Description" — built
        from the KB material + denomination + mint — makes the table
        immediately interpretable (e.g. "Silver Drachm — Maroneia").
    """
    from fpdf.enums import XPos, YPos
    ew    = _ew(f)
    c1    = 14   # Rank
    c2    = 28   # CN Type (numeric ID)
    c4    = 30   # Confidence
    c3    = ew - c1 - c2 - c4  # Coin Description (fills remaining ~98 mm)
    row_h = 7

    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_RULE)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_BRAND_DARK)
    f.set_x(f.l_margin)
    for lbl, w in [("Rank", c1), ("CN Type", c2), ("Coin Description", c3), ("Confidence", c4)]:
        f.cell(w, row_h, f"  {lbl}", border=1, fill=True,
               new_x=XPos.RIGHT, new_y=YPos.TOP)
    f.set_xy(f.l_margin, f.get_y() + row_h)

    for i, t in enumerate(top5):
        f.set_fill_color(*(_C_ROW_ALT if i % 2 == 0 else _C_WHITE))
        f.set_font("Helvetica", "B" if i == 0 else "", 9)
        f.set_text_color(*(_C_BRAND_DARK if i == 0 else _C_TEXT))
        f.set_x(f.l_margin)
        raw_lbl = str(t.get("label", ""))
        _desc = _s(_enrich_label(raw_lbl, include_date=True))
        if len(_desc) > 48:
            _desc = _desc[:45] + "..."
        for val, w in [
            (str(i + 1),                                  c1),
            (_s(raw_lbl),                                  c2),
            (_desc,                                        c3),
            (f"{t.get('confidence', 0):.1%}",             c4),
        ]:
            f.cell(w, row_h, f"  {val}", border="LBR", fill=True,
                   new_x=XPos.RIGHT, new_y=YPos.TOP)
        f.set_xy(f.l_margin, f.get_y() + row_h)

    f.set_text_color(*_C_TEXT)


def _clean_kb_date(raw: str) -> str:
    """
    Normalise a KB date string for PDF display.

    Some KB records store the period name directly appended to the date field:
        'c. 500-450 BC Archaic Period'  -->  'c. 500-450 BC'
    This happens because the scraper concatenated the period sub-label into the
    date cell rather than the separate period field.

    WHY strip here rather than at scrape time:
        Fixing the scraper would require a full re-scrape of 9,541 records.
        A local display fix is faster and has zero risk of data corruption.
    """
    import re as _re
    cleaned = _re.sub(
        r'\s+(Archaic|Classical|Hellenistic|Roman|Byzantine|Early|Late|Middle)\b.*',
        '', raw, flags=_re.IGNORECASE).strip()
    return _s(cleaned)


def _kb_table(f, matches: list) -> None:
    """
    Match% / Coin Identity / Date three-column table for KB closest matches.

    WHY remove the raw CN Type column:
        Raw database IDs (21229, 20674) are opaque to all readers.
        The enriched "Coin Identity" column (Material + Denomination + Mint)
        gives the same information in human-readable form.
        The CN type number is appended in the identity cell as a reference.

    WHY "Match%" instead of raw RRF score:
        RRF scores are small fractions (0.016) meaningless to a numismatist.
        Normalising to 0-100% relative to the top hit gives an intuitive
        sense of relative similarity without implying false precision.
    """
    from fpdf.enums import XPos, YPos
    ew    = _ew(f)
    c1    = 22   # Match %
    c3    = 44   # Date
    c2    = ew - c1 - c3  # Coin Identity (fills remaining ~104 mm)
    row_h = 7

    # Normalise scores to 0-100 % relative to top hit
    raw_scores = [hit.get("score", 0) for hit in matches]
    max_s = max(raw_scores) if raw_scores and max(raw_scores) > 0 else 1.0

    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_RULE)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_BRAND_DARK)
    f.set_x(f.l_margin)
    for label, w in [("Match", c1), ("Coin Identity", c2), ("Date", c3)]:
        f.cell(w, row_h, f"  {label}", border=1, fill=True,
               new_x=XPos.RIGHT, new_y=YPos.TOP)
    f.set_xy(f.l_margin, f.get_y() + row_h)

    for i, hit in enumerate(matches):
        f.set_fill_color(*(_C_ROW_ALT if i % 2 == 0 else _C_WHITE))
        f.set_font("Helvetica", "", 9)
        f.set_text_color(*_C_TEXT)
        f.set_x(f.l_margin)
        sim     = hit.get("score", 0) / max_s * 100
        tid     = hit.get("type_id", "")
        # Coin identity: enriched description + CN number as reference
        identity = _s(_enrich_label(tid))
        identity_cell = f"{identity}  (CN {tid})" if identity != f"CN {tid}" else f"CN {tid}"
        for val, w in [
            (f"{sim:.0f}%",              c1),
            (_s(identity_cell),          c2),
            (_clean_kb_date(hit.get("date", "")),   c3),
        ]:
            f.cell(w, row_h, f"  {val}", border="LBR", fill=True,
                   new_x=XPos.RIGHT, new_y=YPos.TOP)
        f.set_xy(f.l_margin, f.get_y() + row_h)

    f.set_text_color(*_C_TEXT)


def _body_paragraph(f, text: str) -> None:
    """
    Render multi-paragraph body text with smart page-break handling.

    WHAT: Splits text on blank-line paragraph boundaries, renders each
          paragraph individually.  Before each paragraph, checks whether at
          least 25 mm of space remains — if not, adds a new page first.

    WHY: fpdf2's multi_cell() auto-paginates mid-sentence, producing jarring
         "...produced during the Hellenistic period [PAGE] between 148–90 BC"
         breaks.  By rendering paragraph-by-paragraph and adding a page before
         starting a new paragraph when space is tight, each paragraph begins
         with enough room to fit its first sentence without orphaning it.

    NOTE: A single very long paragraph (>page height) will still auto-paginate
          mid-sentence — that is unavoidable without full height pre-measurement
          of every line.  This fix handles the common case (3×100-word
          paragraphs) that fills Routes 1+2 expert commentary.
    """
    from fpdf.enums import XPos, YPos
    f.set_font("Helvetica", "", 10)
    f.set_text_color(*_C_TEXT)
    min_space = 25   # mm: start a new page if less than this remains
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text]
    for i, para in enumerate(paras):
        # Available space = distance to footer band (14 mm) from current Y
        available = f.h - f.b_margin - 14 - f.get_y()
        if available < min_space:
            f.add_page()
        f.set_x(f.l_margin)
        f.multi_cell(_ew(f), 6, _s(para), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if i < len(paras) - 1:
            f.ln(3)   # small gap between paragraphs


def _source_line(f, url: str) -> None:
    from fpdf.enums import XPos, YPos
    f.set_font("Helvetica", "I", 8)
    f.set_text_color(*_C_MUTED)
    f.set_x(f.l_margin)
    f.cell(0, 5, f"Source: {_s(url)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_text_color(*_C_TEXT)


def _status_badge(f, match: bool) -> None:
    from fpdf.enums import XPos, YPos
    color = _C_GREEN if match else _C_ORANGE
    label = "MATERIAL CONSISTENT" if match else "MATERIAL MISMATCH"
    f.set_fill_color(*color)
    f.set_text_color(*_C_WHITE)
    f.set_font("Helvetica", "B", 9)
    f.set_x(f.l_margin)
    f.cell(62, 8, f"  {label}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_text_color(*_C_TEXT)
    f.ln(3)


def _warning_box(f, text: str) -> None:
    from fpdf.enums import XPos, YPos
    f.set_fill_color(255, 243, 220)
    f.set_draw_color(*_C_ORANGE)
    f.set_font("Helvetica", "I", 9)
    f.set_text_color(*_C_ORANGE)
    f.set_x(f.l_margin)
    f.multi_cell(_ew(f), 6, f"  Note: {_s(text)}", border=1, fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_text_color(*_C_TEXT)
    f.set_draw_color(*_C_RULE)


def _info_box(f, text: str) -> None:
    from fpdf.enums import XPos, YPos
    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_BRAND_MID)
    f.set_font("Helvetica", "I", 9)
    f.set_text_color(*_C_BRAND_DARK)
    f.set_x(f.l_margin)
    f.multi_cell(_ew(f), 6, f"  {_s(text)}", border=1, fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_text_color(*_C_TEXT)
    f.set_draw_color(*_C_RULE)


def _trim_to_sections(text: str) -> str:
    """
    Strip verbose reasoning preamble from VLM responses that do not use
    <think>...</think> tags (e.g. qwen3-vl:4b in plain-text mode).

    WHY: Some Ollama reasoning models output their chain-of-thought as plain
    prose before the structured answer rather than wrapping it in think-tags.
    The PDF should show only the six structured sections (METAL:, OBVERSE:,
    etc.), not the model's internal deliberation.

    Strategy:
        Find the first occurrence of a section label at the start of a line.
        Return everything from that point onward.
        If no section label is found, return the original text (graceful).
    """
    m = re.search(
        r"^(METAL|OBVERSE|REVERSE|INSCRIPTIONS|CONDITION|IDENTIFICATION):",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return text[m.start():].strip() if m else text


def _draw_footer_band(f) -> None:
    """Legacy helper — kept for external callers; body is now in _PDF.footer()."""
    pass
