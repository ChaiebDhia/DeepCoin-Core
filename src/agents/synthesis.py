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
_C_GREEN       = (30,  130,  80)
_C_ORANGE      = (200, 100,  20)
_C_RULE        = (200, 210, 225)   # light border lines


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


def _s(text: str) -> str:
    """
    Sanitise a string for PDF output through five ordered steps.

    Step 1 — Strip RAG citation markers  ([CONTEXT 1], [CONTEXT CNN], …)
             These are internal prompt tokens that must never appear in print.
    Step 2 — Strip Markdown syntax  (**, *, ##, `code`, …)
             Some LLM responses contain inline Markdown even when instructed
             not to.  We extract just the visible text between the markers.
    Step 3 — Collapse excess whitespace left by stripping.
    Step 4 — Greek → Latin transliteration  (Σ→S, Α→A, …)
             fpdf2's built-in fonts use latin-1 encoding, which excludes the
             Greek Unicode block (U+0370–U+03FF).
    Step 5 — Unicode NFD decomposition + combining-mark removal
             Converts accented characters (ō, é, ñ, …) to their ASCII base
             letter by decomposing to NFD form and stripping combining marks
             (Unicode category 'Mn').  This preserves readability instead of
             emitting the latin-1 replacement character '?'.
    Step 6 — latin-1 encode/decode  (final safety net for any remaining
             characters outside the font's supported range).
    """
    t = str(text)

    # Step 1 — remove [CONTEXT N] / [CONTEXT CNN] citation markers
    t = _RE_CONTEXT.sub("", t)

    # Step 2 — strip Markdown formatting, keep visible text
    for pattern, repl in _MD_PATTERNS:
        t = pattern.sub(repl, t)

    # Step 3 — collapse runs of spaces / clean up leftover punctuation gaps
    t = re.sub(r"  +", " ", t).strip()

    # Step 4 — Greek transliteration
    t = "".join(_GREEK_MAP.get(c, c) for c in t)

    # Step 5 — decompose accented chars, strip combining diacritics
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")

    # Step 6 — final latin-1 safety net
    return t.encode("latin-1", "replace").decode("latin-1")


def _basename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1] if path else "N/A"


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
        f = FPDF()
        f.set_margins(20, 15, 20)
        f.set_auto_page_break(auto=True, margin=20)
        f.add_page()

        # ── header band ───────────────────────────────────────────────────────
        _draw_header_band(f, ts, img)

        # ── result summary stripe ─────────────────────────────────────────────
        _draw_result_stripe(f, cnn)
        f.ln(6)

        # ── CNN classification ────────────────────────────────────────────────
        _section_title(f, "CNN Classification")
        _kv_table(f, [
            ("Predicted Type",  _s(str(cnn.get("label", "N/A")))),
            ("Confidence",      f"{cnn.get('confidence', 0):.1%}"),
            ("Model",           "EfficientNet-B3  (438 classes)"),
            ("Pipeline Route",  route.upper()),
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
                ("CN Type ID",    str(h.get("type_id",    ""))),
                ("Mint",          h.get("mint",           "")),
                ("Region",        h.get("region",         "")),
                ("Date",          h.get("date",           "")),
                ("Period",        h.get("period",         "")),
                ("Material",      h.get("material",       "")),
                ("Denomination",  h.get("denomination",   "")),
                ("Obverse",       h.get("obverse",        "")),
                ("Reverse",       h.get("reverse",        "")),
                ("Persons",       h.get("persons",        "")),
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
            _section_title(f, "Visual Investigation  (Low-Confidence Route)")
            if inv.get("visual_description"):
                _body_paragraph(f, inv["visual_description"])

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
                _info_box(f, f"Suggested CN type: {inv['suggested_type_id']}  "
                             f"(visual description + semantic KB search)")
            f.ln(7)

        # ── save ──────────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as fout:
            f.output(fout)


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
    f.set_xy(20, 6)
    f.set_text_color(*_C_WHITE)
    f.set_font("Helvetica", "B", 18)
    f.cell(90, 10, "DeepCoin", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    f.set_xy(20, 17)
    f.set_font("Helvetica", "", 9)
    f.set_text_color(180, 200, 230)
    f.cell(90, 6, "Numismatic Intelligence System  |  ESPRIT / YEBNI")

    # Right: timestamp + filename
    f.set_font("Helvetica", "", 8)
    f.set_text_color(180, 200, 230)
    f.set_xy(f.w - 85, 8)
    f.cell(65, 5, _s(ts), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    f.set_xy(f.w - 85, 14)
    f.cell(65, 5, _s(img), align="R")

    f.set_text_color(*_C_TEXT)
    f.set_xy(f.l_margin, 36)


def _draw_result_stripe(f, cnn: dict) -> None:
    from fpdf.enums import XPos, YPos
    label = _s(str(cnn.get("label", "N/A")))
    conf  = cnn.get("confidence", 0.0)

    y = f.get_y()
    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_BRAND_LIGHT)
    f.rect(f.l_margin, y, _ew(f), 16, style="FD")

    f.set_xy(f.l_margin + 5, y + 3)
    f.set_font("Helvetica", "B", 13)
    f.set_text_color(*_C_BRAND_DARK)
    f.cell(60, 8, f"CN Type  {label}")

    f.set_xy(f.l_margin + 70, y + 5)
    f.set_font("Helvetica", "", 10)
    f.set_text_color(*_C_MUTED)
    f.cell(50, 6, f"Confidence:  {conf:.1%}")

    f.set_text_color(*_C_TEXT)
    f.set_draw_color(*_C_RULE)
    f.set_xy(f.l_margin, y + 16)


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
    """Rank / CN Type / Confidence three-column table."""
    from fpdf.enums import XPos, YPos
    ew = _ew(f)
    c1, c3 = 24, 38
    c2 = ew - c1 - c3
    row_h = 7

    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_RULE)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_BRAND_DARK)
    f.set_x(f.l_margin)
    for label, w in [("Rank", c1), ("CN Type", c2), ("Confidence", c3)]:
        f.cell(w, row_h, f"  {label}", border=1, fill=True,
               new_x=XPos.RIGHT, new_y=YPos.TOP)
    f.set_xy(f.l_margin, f.get_y() + row_h)

    for i, t in enumerate(top5):
        f.set_fill_color(*(_C_ROW_ALT if i % 2 == 0 else _C_WHITE))
        f.set_font("Helvetica", "B" if i == 0 else "", 9)
        f.set_text_color(*(_C_BRAND_DARK if i == 0 else _C_TEXT))
        f.set_x(f.l_margin)
        for val, w in [
            (str(i + 1),                        c1),
            (_s(str(t.get("label", ""))),         c2),
            (f"{t.get('confidence', 0):.1%}",   c3),
        ]:
            f.cell(w, row_h, f"  {val}", border="LBR", fill=True,
                   new_x=XPos.RIGHT, new_y=YPos.TOP)
        f.set_xy(f.l_margin, f.get_y() + row_h)

    f.set_text_color(*_C_TEXT)


def _kb_table(f, matches: list) -> None:
    """CN Type / Score / Mint / Date four-column table."""
    from fpdf.enums import XPos, YPos
    ew = _ew(f)
    c1, c2, c3 = 28, 24, 55
    c4 = ew - c1 - c2 - c3
    row_h = 7

    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_RULE)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_BRAND_DARK)
    f.set_x(f.l_margin)
    for label, w in [("Type", c1), ("Score", c2), ("Mint", c3), ("Date", c4)]:
        f.cell(w, row_h, f"  {label}", border=1, fill=True,
               new_x=XPos.RIGHT, new_y=YPos.TOP)
    f.set_xy(f.l_margin, f.get_y() + row_h)

    for i, hit in enumerate(matches):
        f.set_fill_color(*(_C_ROW_ALT if i % 2 == 0 else _C_WHITE))
        f.set_font("Helvetica", "", 9)
        f.set_text_color(*_C_TEXT)
        f.set_x(f.l_margin)
        for val, w in [
            (_s(str(hit.get("type_id", ""))),   c1),
            (f"{hit.get('score', 0):.2f}",      c2),
            (_s(hit.get("mint", "")),            c3),
            (_s(hit.get("date", "")),            c4),
        ]:
            f.cell(w, row_h, f"  {val}", border="LBR", fill=True,
                   new_x=XPos.RIGHT, new_y=YPos.TOP)
        f.set_xy(f.l_margin, f.get_y() + row_h)

    f.set_text_color(*_C_TEXT)


def _body_paragraph(f, text: str) -> None:
    from fpdf.enums import XPos, YPos
    f.set_font("Helvetica", "", 10)
    f.set_text_color(*_C_TEXT)
    f.set_x(f.l_margin)
    f.multi_cell(_ew(f), 6, _s(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


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


def _draw_footer_band(f) -> None:
    from fpdf.enums import XPos, YPos
    f.set_y(f.h - 14)
    f.set_fill_color(*_C_BRAND_DARK)
    f.rect(0, f.h - 14, f.w, 14, style="F")
    f.set_xy(20, f.h - 10)
    f.set_text_color(180, 200, 230)
    f.set_font("Helvetica", "", 8)
    f.cell(110, 5, "DeepCoin-Core  |  ESPRIT School of Engineering  |  YEBNI")
    f.set_xy(f.w - 65, f.h - 10)
    f.cell(45, 5, "corpus-nummorum.eu", align="R")
    f.set_text_color(*_C_TEXT)
