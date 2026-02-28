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

import logging
import threading
import time
from pathlib import Path
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.core.inference import CoinInference
from src.agents.historian    import Historian
from src.agents.validator    import Validator
from src.agents.investigator import Investigator
from src.agents.synthesis    import Synthesis

logger = logging.getLogger(__name__)


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
    # timing (seconds per node, added progressively)
    node_timings  : dict          # {"cnn": 0.31, "historian": 12.4, ...}
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
        # Configure logging once so INFO messages print to stdout
        # WHY basicConfig here and not at module level:
        #   If the caller (FastAPI, test script) has already configured logging,
        #   basicConfig() is a no-op.  If nobody configured it, we get readable
        #   output automatically without imposing a format on the caller.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        # Resolve "auto" device before passing to CoinInference
        import torch as _torch
        if device == "auto":
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        logger.info("Gatekeeper init: device=%s", device)
        self._inference  = CoinInference(model_path, mapping_path, device)
        self._historian  = Historian()
        self._validator  = Validator()
        self._investigator = Investigator()
        self._synthesis  = Synthesis()
        self._save_pdf   = save_pdf
        self._graph      = self._build_graph()
        logger.info("Gatekeeper ready.")

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
        logger.info("Analyzing: %s  (tta=%s)", Path(image_path).name, tta)
        initial_state: CoinState = {
            "image_path": image_path,
            "use_tta":    tta,
            "node_timings": {},
        }
        final_state = self._graph.invoke(initial_state)
        timings = final_state.get("node_timings", {})
        total   = sum(timings.values())
        logger.info(
            "Pipeline complete — route=%s  total=%.2fs  timings=%s",
            final_state.get("route_taken", "?"),
            total,
            {k: f"{v:.2f}s" for k, v in timings.items()},
        )
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

        # ─ retry helper (local to _build_graph) ─────────────────────────────────────────
        def _retry_call(fn, retries: int = 2, backoff: float = 1.5):
            """
            Call fn() with up to `retries` additional attempts on 429/503 errors.

            WHY: LLM providers rate-limit during high traffic.  A blind retry with
            exponential backoff resolves >95% of transient 429 errors without
            requiring the user to restart the pipeline entirely.

            WHAT: checks `exc.status_code` (openai SDK) and the error string for
            "429"/"503"/"rate limit"/"too many" in case a different client is used.
            """
            for attempt in range(retries + 1):
                try:
                    return fn()
                except Exception as exc:
                    code = getattr(exc, "status_code", None)
                    err_str = str(exc).lower()
                    is_rate = code in (429, 503) or any(
                        k in err_str for k in ("429", "503", "rate limit", "too many")
                    )
                    if is_rate and attempt < retries:
                        wait = backoff * (2 ** attempt)   # 1.5 s, 3.0 s
                        logger.warning(
                            "LLM rate-limited (attempt %d/%d) — retrying in %.1f s",
                            attempt + 1, retries, wait,
                        )
                        time.sleep(wait)
                    else:
                        raise

        # ─ node: CNN inference ───────────────────────────────────────────────────────────
        def cnn_node(state: CoinState) -> CoinState:
            """
            Stage 1 — run EfficientNet-B3 inference and decide the routing path.

            HOW it fits: This is the entry node.  Its output `route_taken` drives the
            conditional edge that selects which specialist agent runs next.

            No try/except here because a CNN failure means we have NO prediction at all
            and the rest of the pipeline would be meaningless — surfacing the error
            to the caller is the correct behaviour.
            """
            t0 = time.perf_counter()
            result = inference.predict(state["image_path"], tta=state.get("use_tta", False))
            cnn = {
                "class_id":   result["class_id"],
                "label":      result["label"],
                "confidence": result["confidence"],
                "top5":       result["top5"],
                "tta_used":   result["tta_used"],
            }
            conf = cnn["confidence"]
            # Thresholds: >85% = high certainty (historian only)
            #            40-85% = medium (validate material + historian)
            #            <40%   = unknown  (visual investigator)
            if conf > 0.85:
                route = "historian"
            elif conf >= 0.40:
                route = "validator"
            else:
                route = "investigator"
            elapsed = time.perf_counter() - t0
            logger.info(
                "cnn_node: label=%s  conf=%.1f%%  route=%s  time=%.2fs",
                cnn["label"], conf * 100, route, elapsed,
            )
            timings = {**state.get("node_timings", {}), "cnn": elapsed}
            return {**state, "cnn_prediction": cnn, "route_taken": route, "node_timings": timings}

        # ─ node: Historian ─────────────────────────────────────────────────────────────────
        def historian_node(state: CoinState) -> CoinState:
            """
            Stage 2a — RAG lookup + LLM narrative for high-confidence predictions.

            Graceful degradation: if the historian or LLM raises (network, key, timeout),
            we store an error message in the result and let the pipeline continue to
            synthesis — which will note the unavailability in the final report.
            """
            t0 = time.perf_counter()
            try:
                result = _retry_call(lambda: historian.research(state["cnn_prediction"]))
            except Exception as exc:
                logger.error("historian_node failed: %s", exc, exc_info=True)
                result = {
                    "narrative": f"Historical research unavailable: {exc}",
                    "llm_used":  False,
                    "_error":    str(exc),
                }
            elapsed = time.perf_counter() - t0
            logger.info(
                "historian_node: llm_used=%s  time=%.2fs",
                result.get("llm_used"), elapsed,
            )
            timings = {**state.get("node_timings", {}), "historian": elapsed}
            return {**state, "historian_result": result, "node_timings": timings}

        # ─ node: Validator ────────────────────────────────────────────────────────────────
        def validator_node(state: CoinState) -> CoinState:
            """
            Stage 2b — OpenCV material check + historian narrative (medium confidence).

            The validator uses pure OpenCV (no network) so it cannot rate-limit.
            The historian call inside this node gets the same retry protection.
            If either sub-call fails, the node stores a degraded result and continues.
            """
            t0 = time.perf_counter()
            try:
                val_result = validator.validate(state["image_path"], state["cnn_prediction"])
            except Exception as exc:
                logger.error("validator_node (CV) failed: %s", exc, exc_info=True)
                val_result = {
                    "status":               "unknown",
                    "warning":              f"CV analysis error: {exc}",
                    "detection_confidence": 0.0,
                    "uncertainty":          "high",
                    "_error":               str(exc),
                }
            try:
                hist_result = _retry_call(lambda: historian.research(state["cnn_prediction"]))
            except Exception as exc:
                logger.error("validator_node (historian) failed: %s", exc, exc_info=True)
                hist_result = {
                    "narrative": f"Historical research unavailable: {exc}",
                    "llm_used":  False,
                    "_error":    str(exc),
                }
            elapsed = time.perf_counter() - t0
            logger.info(
                "validator_node: status=%s  conf=%.2f  time=%.2fs",
                val_result.get("status"),
                val_result.get("detection_confidence", 0.0),
                elapsed,
            )
            timings = {**state.get("node_timings", {}), "validator": elapsed}
            return {
                **state,
                "validator_result":  val_result,
                "historian_result":  hist_result,
                "node_timings":      timings,
            }

        # ─ node: Investigator ────────────────────────────────────────────────────────────
        def investigator_node(state: CoinState) -> CoinState:
            """
            Stage 2c — VLM / OpenCV fallback + KB cross-reference (low confidence).

            The investigator already uses _opencv_fallback internally when no vision
            LLM is available, so it never crashes on missing API keys.
            The retry wrapper handles transient API rate-limits.
            """
            t0 = time.perf_counter()
            try:
                result = _retry_call(
                    lambda: investigator.investigate(
                        state["image_path"], state["cnn_prediction"]
                    )
                )
            except Exception as exc:
                logger.error("investigator_node failed: %s", exc, exc_info=True)
                result = {
                    "visual_description": f"Investigation unavailable: {exc}",
                    "detected_features":  {},
                    "kb_matches":         [],
                    "suggested_type_id":  None,
                    "llm_used":           False,
                    "_error":             str(exc),
                }
            elapsed = time.perf_counter() - t0
            logger.info(
                "investigator_node: llm_used=%s  kb_matches=%d  time=%.2fs",
                result.get("llm_used"),
                len(result.get("kb_matches", [])),
                elapsed,
            )
            timings = {**state.get("node_timings", {}), "investigator": elapsed}
            return {**state, "investigator_result": result, "node_timings": timings}

        # ─ node: Synthesis ────────────────────────────────────────────────────────────────
        def synthesis_node(state: CoinState) -> CoinState:
            """
            Stage 3 — assemble the final Markdown report and render the PDF.

            PDF rendering is wrapped separately because a synthesis failure is bad
            but a PDF rendering failure (e.g. missing font) should NOT discard the
            already-assembled text report — we degrade to report-only output.
            """
            t0     = time.perf_counter()
            report = synthesis.synthesize(state)
            pdf_path = None
            if save_pdf:
                _REPORTS_DIR.mkdir(exist_ok=True)
                img_stem = Path(state["image_path"]).stem
                ts       = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = str(_REPORTS_DIR / f"report_{img_stem}_{ts}.pdf")
                try:
                    synthesis.to_pdf(state, pdf_path)
                    logger.info("synthesis_node: PDF saved -> %s", pdf_path)
                except Exception as pdf_err:
                    logger.error("synthesis_node PDF error: %s", pdf_err, exc_info=True)
                    pdf_path = None
            elapsed = time.perf_counter() - t0
            logger.info("synthesis_node: time=%.2fs", elapsed)
            timings = {**state.get("node_timings", {}), "synthesis": elapsed}
            return {**state, "report": report, "pdf_path": pdf_path, "node_timings": timings}

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
#  MODULE-LEVEL SINGLETON  (lazy, thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

_gk_instance: Gatekeeper | None = None
_gk_lock     = threading.Lock()

def get_gatekeeper(**kwargs) -> Gatekeeper:
    """
    Return the shared Gatekeeper instance (created on first call).

    Thread-safe via double-checked locking pattern:
        - First check (outside lock) avoids acquiring the lock on every call
          once the instance is initialised — the common case.
        - Second check (inside lock) prevents two threads that both passed the
          first check from both calling Gatekeeper() simultaneously.

    WHY this matters:
        If two threads both see _gk_instance is None and both call Gatekeeper(),
        both load 79 MB EfficientNet weights into VRAM simultaneously.
        On a 4.3 GB card this risks OOM. The second instance is then silently
        discarded — wasted work and potential crash.

    NOTE: FastAPI uses app.state.gk (set in lifespan, inherently single-init).
          This function is used by scripts and tests that create a Gatekeeper
          outside the FastAPI context.
    """
    global _gk_instance
    if _gk_instance is None:
        with _gk_lock:
            if _gk_instance is None:   # second check inside the lock
                _gk_instance = Gatekeeper(**kwargs)
    return _gk_instance

