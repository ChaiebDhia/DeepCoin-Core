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


def _s(text: str) -> str:
    """Replace non-latin-1 characters with '?' so fpdf2 (Helvetica) can render them."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


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

        # ── footer band ───────────────────────────────────────────────────────
        _draw_footer_band(f)

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
    """Two-column key/value table with borders and alternating shading."""
    from fpdf.enums import XPos, YPos
    if not rows:
        return
    col_k = 52
    col_v = _ew(f) - col_k
    row_h = 7

    # Column header
    f.set_fill_color(*_C_BRAND_LIGHT)
    f.set_draw_color(*_C_RULE)
    f.set_font("Helvetica", "B", 9)
    f.set_text_color(*_C_BRAND_DARK)
    f.set_x(f.l_margin)
    f.cell(col_k, row_h, "  Field",  border=1, fill=True,
           new_x=XPos.RIGHT, new_y=YPos.TOP)
    f.cell(col_v, row_h, "  Value",  border=1, fill=True,
           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for i, (key, val) in enumerate(rows):
        f.set_fill_color(*(_C_ROW_ALT if i % 2 == 0 else _C_WHITE))
        f.set_x(f.l_margin)

        # Key
        f.set_font("Helvetica", "B", 9)
        f.set_text_color(*_C_MUTED)
        f.cell(col_k, row_h, f"  {_s(key)}", border="LBR", fill=True,
               new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Value — multi_cell for long text, anchored to left margin after
        f.set_font("Helvetica", "", 9)
        f.set_text_color(*_C_TEXT)
        f.multi_cell(col_v, row_h, f"  {_s(val)}", border="LBR", fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

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
