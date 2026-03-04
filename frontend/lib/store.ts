/**
 * lib/store.ts
 * ============
 * Zustand global state store.
 *
 * WHY Zustand instead of React context:
 *   React context re-renders EVERY consumer on every update — even if the
 *   consumer only uses one field. Zustand uses selector subscriptions:
 *   a component that only reads `uploadProgress` does NOT re-render when
 *   `result` changes. For an analysis panel that updates every second during
 *   a 20–60 second LLM call, this matters.
 *
 * WHY separate from TanStack Query:
 *   TanStack Query owns SERVER state (cached API responses + refetch logic).
 *   Zustand owns CLIENT state (upload progress, TTA toggle, UI phase).
 *   They're complementary. Rule of thumb:
 *     "Does this data come from the server?" → TanStack Query
 *     "Is this a UI-only state?" → Zustand
 *
 * State machine for the upload flow:
 *   idle → uploading → processing → done
 *                                 → error
 */

import { create } from "zustand";
import type { ClassifyResponse } from "@/types/api";

// ── Upload phase type ─────────────────────────────────────────────────────────

export type UploadPhase = "idle" | "uploading" | "processing" | "done" | "error";

// ── Store shape ───────────────────────────────────────────────────────────────

interface DeepCoinState {
  /** Whether TTA is enabled for the next classify call. Default: true. */
  tta:             boolean;

  /** Current phase of the upload → processing → result flow. */
  phase:           UploadPhase;

  /** 0–100 upload progress (set during the multipart PUT). */
  uploadProgress:  number;

  /** The file the user has selected (or is currently uploading). */
  selectedFile:    File | null;

  /** The most recent successful ClassifyResponse. */
  result:          ClassifyResponse | null;

  /** Error message from the last failed classify call. */
  errorMessage:    string | null;

  // ── Actions ────────────────────────────────────────────────────────────────

  setTta:            (tta: boolean)                 => void;
  setSelectedFile:   (file: File | null)            => void;
  setUploadProgress: (pct: number)                  => void;
  setPhase:          (phase: UploadPhase)           => void;
  setResult:         (result: ClassifyResponse)     => void;
  setError:          (message: string)              => void;

  /** Reset to idle state, clearing file + result + errors. */
  reset:             ()                             => void;

  /**
   * Stores the cancel callback registered by CoinUploader when a classify
   * request is in flight.  AgentPipeline reads this so its X button can
   * trigger the REAL abort (AbortController.abort) + reset, not just reset.
   */
  _cancelFn:         (() => void) | null;
  setCancelFn:       (fn: (() => void) | null)      => void;
}

// ── Store implementation ──────────────────────────────────────────────────────

export const useDeepCoinStore = create<DeepCoinState>()((set) => ({
  tta:             true,
  phase:           "idle",
  uploadProgress:  0,
  selectedFile:    null,
  result:          null,
  errorMessage:    null,

  setTta:            (tta)      => set({ tta }),
  setSelectedFile:   (file)     => set({ selectedFile: file }),
  setUploadProgress: (pct)      => set({ uploadProgress: pct }),
  // WHY clear result on "uploading": when handleAnalyse() begins, the previous
  // analysis result must vanish immediately so its PDF download button cannot
  // persist into the new upload phase. Without this, a completed AnalysisPanel
  // stays visible until setResult() fires for the NEW analysis (up to 60 s).
  setPhase:          (phase)    => phase === "uploading"
    ? set({ phase, result: null, errorMessage: null })
    : set({ phase }),
  setResult:         (result)   => set({ result, phase: "done", errorMessage: null }),
  setError:          (message)  => set({ errorMessage: message, phase: "error" }),
  _cancelFn:         null,
  setCancelFn:       (fn)       => set({ _cancelFn: fn }),
  reset:             ()         => set({
    phase:           "idle",
    uploadProgress:  0,
    selectedFile:    null,
    result:          null,
    errorMessage:    null,
    _cancelFn:       null,
  }),
}));
