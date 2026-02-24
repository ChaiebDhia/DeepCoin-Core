"""
src/agents/gatekeeper.py
=========================
Layer 3 — Gatekeeper (LangGraph Orchestrator)

Routes a coin image through the correct agents based on CNN confidence:

  confidence > 0.85  →  Historian  (high certainty — fetch history, write narrative)
  0.40 – 0.85       →  Validator + Historian  (medium — check material consistency too)
  < 0.40             →  Investigator  (low — send to Gemini Vision, no KB lookup)

The final node in all paths is Synthesis, which assembles the report.

Public API:
    gk = Gatekeeper()
    result = gk.analyze("path/to/coin.jpg", tta=True)
    # result["report"]   → Markdown string
    # result["pdf_path"] → path to saved PDF (or None)
    # result["state"]    → full LangGraph state dict

Engineering notes:
  - CoinInference is loaded ONCE in __init__ (4-rule compliance)
  - KnowledgeBase is a singleton (loaded once per process)
  - LangGraph state is a plain TypedDict — no OOP magic
  - Each node is a pure function: state_in → state_out
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.core.inference import CoinInference
from src.agents.historian    import Historian
from src.agents.validator    import Validator
from src.agents.investigator import Investigator
from src.agents.synthesis    import Synthesis


# ── paths ───────────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent.parent
_MODEL_PATH   = str(_ROOT / "models" / "best_model.pth")
_MAPPING_PATH = str(_ROOT / "models" / "class_mapping.pth")
_REPORTS_DIR  = _ROOT / "reports"


# ── LangGraph state schema ────────────────────────────────────────────────────────

class CoinState(TypedDict, total=False):
    """LangGraph state — travels through every node."""
    # inputs
    image_path    : str
    use_tta       : bool
    # after CNN node
    cnn_prediction: dict          # from CoinInference.predict()
    route_taken   : Literal["historian", "validator", "investigator"]
    # agent outputs
    historian_result   : dict
    validator_result   : dict
    investigator_result: dict
    # final
    report        : str           # Markdown
    pdf_path      : Optional[str] # path to saved PDF


# ══════════════════════════════════════════════════════════════════════════════

class Gatekeeper:
    """
    Gatekeeper — orchestrates the full DeepCoin pipeline.

    Load it once, call .analyze() many times.

        gk     = Gatekeeper()
        result = gk.analyze("data/processed/1015/coin.jpg", tta=True)
        print(result["report"])
    """

    def __init__(
        self,
        model_path:   str = _MODEL_PATH,
        mapping_path: str = _MAPPING_PATH,
        device:       str = "auto",
        save_pdf:     bool = True,
    ) -> None:
        # Resolve "auto" device before passing to CoinInference
        import torch as _torch
        if device == "auto":
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        # Load inference engine ONCE (Rule 4)
        self._inference  = CoinInference(model_path, mapping_path, device)
        self._historian  = Historian()
        self._validator  = Validator()
        self._investigator = Investigator()
        self._synthesis  = Synthesis()
        self._save_pdf   = save_pdf
        self._graph      = self._build_graph()

    # ── public ────────────────────────────────────────────────────────────────

    def analyze(self, image_path: str, tta: bool = False) -> dict:
        """
        Run the full pipeline on one coin image.

        Parameters
        ----------
        image_path : str   path to the coin image
        tta        : bool  whether to use Test-Time Augmentation

        Returns
        -------
        dict with keys:
            report    : str           — full Markdown report
            pdf_path  : str | None    — path to saved PDF
            state     : CoinState     — full internal state
        """
        initial_state: CoinState = {
            "image_path": image_path,
            "use_tta":    tta,
        }
        final_state = self._graph.invoke(initial_state)
        return {
            "report":   final_state.get("report", ""),
            "pdf_path": final_state.get("pdf_path"),
            "state":    final_state,
        }

    # ── graph construction ───────────────────────────────────────────────────────────

    def _build_graph(self):
        """Construct and compile the LangGraph state machine."""
        g = StateGraph(CoinState)

        # Bind agent instances into node functions
        inference   = self._inference
        historian   = self._historian
        validator   = self._validator
        investigator = self._investigator
        synthesis   = self._synthesis
        save_pdf    = self._save_pdf

        # ─ node: CNN inference ───────────────────────────────────────────────────────────
        def cnn_node(state: CoinState) -> CoinState:
            result = inference.predict(state["image_path"], tta=state.get("use_tta", False))
            # Normalise to the field names our agents expect
            cnn = {
                "class_id":   result["class_id"],
                "label":      result["label"],
                "confidence": result["confidence"],
                "top5":       result["top5"],
                "tta_used":   result["tta_used"],
            }
            # Routing decision happens here
            conf = cnn["confidence"]
            if conf > 0.85:
                route = "historian"
            elif conf >= 0.40:
                route = "validator"
            else:
                route = "investigator"
            return {**state, "cnn_prediction": cnn, "route_taken": route}

        # ─ node: Historian ─────────────────────────────────────────────────────────────────
        def historian_node(state: CoinState) -> CoinState:
            result = historian.research(state["cnn_prediction"])
            return {**state, "historian_result": result}

        # ─ node: Validator ────────────────────────────────────────────────────────────────
        def validator_node(state: CoinState) -> CoinState:
            val_result = validator.validate(state["image_path"], state["cnn_prediction"])
            # Validator route also fetches history for context
            hist_result = historian.research(state["cnn_prediction"])
            return {**state, "validator_result": val_result, "historian_result": hist_result}

        # ─ node: Investigator ────────────────────────────────────────────────────────────
        def investigator_node(state: CoinState) -> CoinState:
            result = investigator.investigate(state["image_path"], state["cnn_prediction"])
            return {**state, "investigator_result": result}

        # ─ node: Synthesis ────────────────────────────────────────────────────────────────
        def synthesis_node(state: CoinState) -> CoinState:
            report = synthesis.synthesize(state)
            pdf_path = None
            if save_pdf:
                _REPORTS_DIR.mkdir(exist_ok=True)
                img_stem = Path(state["image_path"]).stem
                ts       = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = str(_REPORTS_DIR / f"report_{img_stem}_{ts}.pdf")
                try:
                    synthesis.to_pdf(report, pdf_path)
                except Exception as _pdf_err:
                    print(f"[Gatekeeper] PDF error: {_pdf_err}")
                    pdf_path = None
            return {**state, "report": report, "pdf_path": pdf_path}

        # ─ wire the graph ─────────────────────────────────────────────────────────────────
        g.add_node("cnn",         cnn_node)
        g.add_node("historian",   historian_node)
        g.add_node("validator",   validator_node)
        g.add_node("investigator",investigator_node)
        g.add_node("synthesis",   synthesis_node)

        g.set_entry_point("cnn")

        def _route(state: CoinState) -> str:
            return state.get("route_taken", "historian")

        g.add_conditional_edges(
            "cnn",
            _route,
            {
                "historian":   "historian",
                "validator":   "validator",
                "investigator":"investigator",
            },
        )

        g.add_edge("historian",    "synthesis")
        g.add_edge("validator",    "synthesis")
        g.add_edge("investigator", "synthesis")
        g.add_edge("synthesis",    END)

        return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON  (lazy)
# ══════════════════════════════════════════════════════════════════════════════

_gk_instance: Gatekeeper | None = None

def get_gatekeeper(**kwargs) -> Gatekeeper:
    """
    Return the shared Gatekeeper instance (created on first call).
    Pass kwargs only on first call to override defaults.
    """
    global _gk_instance
    if _gk_instance is None:
        _gk_instance = Gatekeeper(**kwargs)
    return _gk_instance
