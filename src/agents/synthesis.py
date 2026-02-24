"""
src/agents/synthesis.py
========================
Layer 3 — Synthesis Agent

Collects ALL agent outputs from the LangGraph state and assembles
the final DeepCoin report in two formats:
  1. Markdown string  (used in API response, frontend display)
  2. PDF file         (saved to disk, downloadable)

Engineering notes:
  - One method: synthesize(state) → str (markdown)
  - One method: to_pdf(markdown, output_path) → saves PDF
  - PDF uses fpdf2 (pure Python, no system dependencies)
  - No LLM calls here — all LLM work happened in earlier agents
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class Synthesis:
    """
    Synthesis agent — assembles the final report from LangGraph state.

    Expected state keys (all optional — agent handles missing data gracefully):
        cnn_prediction   : dict   (class_id, label, confidence, top5, tta_used)
        route_taken      : str    ("historian" | "validator" | "investigator")
        historian_result : dict   (from Historian agent)
        validator_result : dict   (from Validator agent)
        investigator_result : dict (from Investigator agent)
        image_path       : str
    """

    def synthesize(self, state: dict) -> str:
        """
        Main entry point — returns a full Markdown report string.
        """
        cnn   = state.get("cnn_prediction", {})
        route = state.get("route_taken", "unknown")

        sections: list[str] = [
            self._header(cnn, state.get("image_path", "")),
            self._classification_section(cnn),
        ]

        if route == "historian" or "historian_result" in state:
            sections.append(self._historian_section(state.get("historian_result", {})))

        if route == "validator" or "validator_result" in state:
            sections.append(self._validator_section(state.get("validator_result", {})))
            # Validator route also runs historian for context
            if "historian_result" in state and route == "validator":
                sections.append(self._historian_section(state.get("historian_result", {})))

        if route == "investigator" or "investigator_result" in state:
            sections.append(self._investigator_section(state.get("investigator_result", {})))

        sections.append(self._footer())
        return "\n".join(sections)

    def to_pdf(self, markdown_content: str, output_path: str) -> None:
        """
        Convert the Markdown report to a PDF file.
        Uses fpdf2 — strips Markdown formatting, writes clean A4 PDF.
        """
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
        import re, os

        def _safe(text: str) -> str:
            """Replace unmappable characters with '?' for latin-1 PDF rendering."""
            return text.encode("latin-1", "replace").decode("latin-1")

        # Margins BEFORE add_page so effective_page_width is correct
        pdf = FPDF()
        pdf.set_margins(20, 20, 20)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        for line in markdown_content.split("\n"):
            try:
                # Skip Markdown table separator rows (|---|---|)
                if re.match(r"^\s*\|[-| :]+\|\s*$", line):
                    continue

                # Convert pipe-table data rows to readable text
                if line.startswith("|") and line.endswith("|"):
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    cells = [re.sub(r"[*_`]+", "", c).strip() for c in cells if c.strip()]
                    if cells:
                        row_text = "  |  ".join(cells)
                        if row_text.strip():
                            pdf.set_font("Helvetica", "", 10)
                            pdf.set_x(pdf.l_margin)
                            pdf.multi_cell(0, 6, _safe(row_text),
                                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    continue

                # Strip remaining Markdown markers for regular lines
                clean = re.sub(r"[*_`#]+", "", line).strip()
                if not clean:
                    pdf.ln(3)
                    continue

                if line.startswith("# "):
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.cell(0, 10, _safe(clean), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(2)
                elif line.startswith("## "):
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, _safe(clean), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(1)
                elif line.startswith("### "):
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.cell(0, 7, _safe(clean), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                elif line.startswith("---"):
                    pdf.ln(2)
                    pdf.set_draw_color(180, 180, 180)
                    pdf.line(pdf.get_x(), pdf.get_y(), pdf.w - 20, pdf.get_y())
                    pdf.ln(2)
                else:
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 6, _safe(clean),
                                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            except Exception as _line_err:
                # Skip lines that can't render; log for debug
                print(f"[PDF] skipped line: {_line_err}")
                continue

        # Write binary to avoid Windows codec errors
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as fout:
            pdf.output(fout)

    def _process_line_placeholder(self): pass  # remove in next cleanup

    # ── private section builders ─────────────────────────────────────────────────

    def _header(self, cnn: dict, image_path: str) -> str:
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img  = image_path.split("/")[-1].split("\\")[-1] if image_path else "N/A"
        label = cnn.get("label", "Unknown")
        conf  = cnn.get("confidence", 0.0)
        return (
            f"# DeepCoin Classification Report\n\n"
            f"**Date**: {ts}  \n"
            f"**Image**: {img}  \n"
            f"**Result**: {label} ({conf:.1%} confidence)  \n"
            f"**Source**: Corpus Nummorum (corpus-nummorum.eu)  \n\n"
            f"---"
        )

    def _classification_section(self, cnn: dict) -> str:
        label = cnn.get("label", "Unknown")
        conf  = cnn.get("confidence", 0.0)
        tta   = " (TTA)" if cnn.get("tta_used") else ""
        top5  = cnn.get("top5", [])
        conf_bar = _bar(conf)

        top5_lines = ""
        if top5:
            top5_lines = "\n**Top-5 predictions:**\n"
            for t in top5:
                bar = _bar(t["confidence"])
                top5_lines += f"- `{t['label']}` {bar} {t['confidence']:.1%}\n"

        return (
            f"\n## CNN Classification\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| **Predicted type** | `{label}` |\n"
            f"| **Confidence** | {conf:.1%}{tta} {conf_bar} |\n"
            f"| **Model** | EfficientNet-B3 (438 classes) |\n"
            f"{top5_lines}"
        )

    def _historian_section(self, h: dict) -> str:
        if not h:
            return "\n## Historical Record\n\n*No KB record available.*"

        fields = [
            ("Type ID",      h.get("type_id",      "")),
            ("Mint",         h.get("mint",          "")),
            ("Region",       h.get("region",        "")),
            ("Date",         h.get("date",          "")),
            ("Period",       h.get("period",        "")),
            ("Material",     h.get("material",      "")),
            ("Denomination", h.get("denomination",  "")),
            ("Obverse",      h.get("obverse",       "")),
            ("Reverse",      h.get("reverse",       "")),
            ("Persons",      h.get("persons",       "")),
        ]
        table_rows = "\n".join(
            f"| **{k}** | {v} |"
            for k, v in fields if v
        )
        narrative = h.get("narrative", "")
        source    = h.get("source_url", "")
        llm_note  = " *(AI narrative)*" if h.get("llm_used") else ""

        return (
            f"\n## Historical Record\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"{table_rows}\n\n"
            + (f"### Expert Commentary{llm_note}\n\n{narrative}\n\n" if narrative else "")
            + (f"**Source**: [{source}]({source})\n" if source else "")
        )

    def _validator_section(self, v: dict) -> str:
        if not v:
            return ""
        status   = v.get("status", "unknown")
        detected = v.get("detected_material", "unknown")
        expected = v.get("expected_material", "unknown")
        warning  = v.get("warning", "")
        match    = v.get("match", True)

        badge = "**Material: Consistent**" if match else "**Material: Mismatch**"
        icon  = "\u2705" if match else "\u26a0\ufe0f"

        return (
            f"\n## Forensic Validation\n\n"
            f"{icon} {badge}\n\n"
            f"| | |\n"
            f"|---|---|\n"
            f"| **Status** | {status.upper()} |\n"
            f"| **Detected material** | {detected} |\n"
            f"| **Expected material** | {expected} |\n"
            + (f"\n> {warning}\n" if warning else "")
        )

    def _investigator_section(self, inv: dict) -> str:
        if not inv:
            return ""
        desc     = inv.get("visual_description", "")
        features = inv.get("detected_features", {})
        kb_hits  = inv.get("kb_matches", [])
        suggest  = inv.get("suggested_type_id")
        llm_note = " *(Gemini Vision)*" if inv.get("llm_used") else ""

        feat_lines = ""
        if features:
            feat_lines = "\n**Detected attributes:**\n"
            for k, v in features.items():
                if v and v != "unknown" and v != []:
                    feat_lines += f"- **{k.replace('_', ' ').title()}**: {v}\n"

        kb_lines = ""
        if kb_hits:
            kb_lines = "\n**Closest KB matches (semantic):**\n"
            for hit in kb_hits[:3]:
                kb_lines += (
                    f"- `CN_{hit['type_id']}` — {hit.get('mint','')} "
                    f"{hit.get('date','')} [{hit['score']:.2f}]\n"
                )

        suggest_line = (
            f"\n> **Best guess**: CN type `{suggest}` (from visual description + KB search)\n"
            if suggest else ""
        )

        return (
            f"\n## Visual Investigation (Low Confidence){llm_note}\n\n"
            f"{desc}\n"
            f"{feat_lines}"
            f"{kb_lines}"
            f"{suggest_line}"
        )

    def _footer(self) -> str:
        return (
            "\n---\n\n"
            "*Generated by **DeepCoin-Core** — EfficientNet-B3 + LangGraph Multi-Agent System*  \n"
            "*ESPRIT School of Engineering / YEBNI*  \n"
            "*Data source: [Corpus Nummorum](https://www.corpus-nummorum.eu)*"
        )


# ── helpers ──────────────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 10) -> str:
    """ASCII confidence bar, e.g. [#######---] 70%"""
    filled = int(score * width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"
