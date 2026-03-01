"use client";

/**
 * components/coin/CoinUploader.tsx
 * =================================
 * Drag-and-drop coin image uploader with TTA toggle and upload progress.
 *
 * WHY drag-and-drop matters for museum use:
 *   Numismatists take dozens of photos per session. Drag-and-drop is
 *   faster than open-file-dialog for rapid sequential analysis.
 *
 * WHY validate client-side before sending:
 *   Sending a 50 MB video to FastAPI wastes bandwidth. Client-side
 *   validation (type + size) prevents that with instant feedback.
 *   Server still validates too (defence in depth).
 *
 * State flow:
 *   User drops file → setSelectedFile → user clicks "Analyse"
 *   → classifyCoin() starts → phase: uploading → phase: processing → phase: done
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { UploadCloud, ImageIcon, XCircle, StopCircle } from "lucide-react";
import toast                                  from "react-hot-toast";

import { classifyCoin }                       from "@/lib/api";
import { useDeepCoinStore }                   from "@/lib/store";
import { cn }                                 from "@/lib/utils";
import { Button }                             from "@/components/ui/button";
import { Progress }                           from "@/components/ui/progress";
import { Spinner }                            from "@/components/ui/spinner";

// 10 MB client-side limit (matches FastAPI MAX_UPLOAD_BYTES)
const MAX_SIZE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES  = ["image/jpeg", "image/png"];

export function CoinUploader() {
  const {
    tta, setTta,
    phase,
    uploadProgress,
    selectedFile,
    errorMessage,
    setSelectedFile, setUploadProgress, setPhase, setResult, setError, reset, setCancelFn,
  } = useDeepCoinStore();

  const [isDragging, setIsDragging] = useState(false);
  const inputRef   = useRef<HTMLInputElement>(null);
  /**
   * AbortController ref for the in-flight classifyCoin Axios request.
   *
   * WHY a ref (not state): We don't need a re-render when the controller
   * changes. We only need to call .abort() from the reset/unmount path.
   *
   * WHY abort on reset: Without this, clicking the X button mid-upload
   * resets the UI to idle but leaves a 2-minute Axios POST running in the
   * background. When it eventually resolves it calls setResult() on the
   * already-cleared store, unexpectedly showing results for the cancelled
   * file.
   */
  const abortRef = useRef<AbortController | null>(null);

  /**
   * Stable object-URL for the coin image preview.
   *
   * WHY useMemo + useEffect:
   *   URL.createObjectURL() called inline in JSX creates a NEW blob URL on
   *   EVERY render and never revokes the old ones. Over dozens of uploads
   *   per numismatist session this leaks significant memory.
   *   useMemo creates exactly one URL per selectedFile value;
   *   the cleanup effect revokes it when selectedFile changes or the
   *   component unmounts.
   */
  const previewUrl = useMemo(
    () => (selectedFile ? URL.createObjectURL(selectedFile) : null),
    [selectedFile],
  );
  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  // Abort any in-flight request when the component unmounts
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // ── File validation ────────────────────────────────────────────────────────

  function validateFile(file: File): string | null {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return `Unsupported file type "${file.type}". Please use JPEG or PNG.`;
    }
    if (file.size > MAX_SIZE_BYTES) {
      return `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 10 MB.`;
    }
    return null;
  }

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const file  = files[0];
    const error = validateFile(file);
    if (error) {
      toast.error(error);
      return;
    }
    reset();
    setSelectedFile(file);
  }

  // ── Drag events ────────────────────────────────────────────────────────────

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Upload + analyse ───────────────────────────────────────────────────────

  async function handleAnalyse() {
    if (!selectedFile) return;

    // Cancel any previous in-flight request before starting a new one
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    // Register the cancel callback so AgentPipeline's X button can trigger it
    setCancelFn(handleCancel);

    try {
      setPhase("uploading");
      setUploadProgress(0);

      const result = await classifyCoin(selectedFile, tta, (pct) => {
        setUploadProgress(pct);
        if (pct === 100) setPhase("processing");
      }, signal);

      setResult(result);
      toast.success(`Analysis complete — ${result.route_taken} route`);
    } catch (err: unknown) {
      // Axios names the cancellation error "CanceledError" — ignore it
      // (user intentionally cancelled; the UI already reset to idle)
      if (err instanceof Error && err.name === "CanceledError") return;
      const msg = err instanceof Error ? err.message : "Classification failed";
      setError(msg);
      toast.error(msg);
    }
  }

  // Abort the in-flight request and reset UI to idle
  function handleCancel() {
    abortRef.current?.abort();
    abortRef.current = null;
    setCancelFn(null);
    reset();
    toast("Analysis cancelled.", { icon: "✋" });
  }

  // ── Derived state ──────────────────────────────────────────────────────────

  const isLoading     = phase === "uploading" || phase === "processing";
  const hasFile       = selectedFile !== null;
  const canAnalyse    = hasFile && !isLoading;

  const phaseLabel =
    phase === "uploading"   ? `Uploading… ${uploadProgress}%` :
    phase === "processing"  ? "Processing — CNN + Agents…"     :
    null;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 w-full max-w-xl">
      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop a coin image here or click to browse"
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !hasFile && inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !hasFile && inputRef.current?.click()}
        className={cn(
          "relative flex flex-col items-center justify-center gap-3",
          "rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer",
          "min-h-[200px] px-6 py-8 text-center",
          isDragging
            ? "border-blue-400 bg-blue-900/20 scale-[1.01]"
            : "border-[var(--border)] bg-[var(--surface-1)] hover:border-blue-600 hover:bg-[var(--surface-2)]",
          hasFile && "border-blue-600 bg-[var(--surface-2)]",
        )}
      >
        {hasFile ? (
          <>
            {/* Thumbnail preview */}
            <img
              src={previewUrl!}
              alt="Selected coin"
              className="max-h-40 max-w-full rounded-lg object-contain shadow-md"
            />
            <p className="text-sm text-[var(--text-primary)] font-medium mt-1">
              {selectedFile.name}
            </p>
            <p className="text-xs text-[var(--text-muted)]">
              {(selectedFile.size / 1024).toFixed(0)} KB ·{" "}
              {selectedFile.type === "image/jpeg" ? "JPEG" : "PNG"}
            </p>
            {/* Remove button */}
            <button
              onClick={(e) => { e.stopPropagation(); abortRef.current?.abort(); abortRef.current = null; reset(); }}
              className="absolute top-2 right-2 text-[var(--text-muted)] hover:text-red-400 transition-colors"
              aria-label="Remove file"
            >
              <XCircle size={18} />
            </button>
          </>
        ) : (
          <>
            <div className="rounded-full bg-blue-900/40 p-4">
              {isDragging
                ? <UploadCloud size={36} className="text-blue-400" />
                : <ImageIcon   size={36} className="text-[var(--text-secondary)]" />
              }
            </div>
            <div>
              <p className="text-sm font-medium text-[var(--text-primary)]">
                Drop a coin photograph here
              </p>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                or <span className="text-blue-400 underline underline-offset-2">browse files</span>
                {" "}— JPEG / PNG, max 10 MB
              </p>
            </div>
          </>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {/* TTA toggle */}
      <label className="flex items-center gap-3 cursor-pointer select-none">
        <div
          role="switch"
          aria-checked={tta}
          tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setTta(!tta)}
          onClick={() => setTta(!tta)}
          className={cn(
            "relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent",
            "transition-colors duration-200 cursor-pointer",
            tta ? "bg-blue-600" : "bg-[var(--surface-3)]",
          )}
        >
          <span
            className={cn(
              "pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow-lg",
              "transition-transform duration-200",
              tta ? "translate-x-4" : "translate-x-0",
            )}
          />
        </div>
        <span className="text-sm text-[var(--text-secondary)]">
          Test-Time Augmentation
        </span>
        <span className="text-xs text-[var(--text-muted)] ml-auto">
          {tta ? "+0.78% accuracy (5 passes)" : "Single pass — faster"}
        </span>
      </label>

      {/* Upload progress bar */}
      {isLoading && (
        <div className="flex flex-col gap-1.5">
          <Progress value={phase === "processing" ? 100 : uploadProgress} />
          <p className="text-xs text-[var(--text-muted)] flex items-center gap-1.5">
            <Spinner size={12} />
            {phaseLabel}
          </p>
        </div>
      )}

      {/* Action button — Cancel during loading, Analyse when idle */}
      {isLoading ? (
        <Button
          variant="secondary"
          size="lg"
          onClick={handleCancel}
          className="w-full border border-red-700/50 text-red-400 hover:bg-red-900/30 hover:text-red-300"
        >
          <StopCircle size={16} />
          Cancel Analysis
        </Button>
      ) : (
        <Button
          variant="primary"
          size="lg"
          disabled={!canAnalyse}
          onClick={handleAnalyse}
          className="w-full"
        >
          Analyse Coin
        </Button>
      )}

      {/* Error message */}
      {phase === "error" && (
        <p className="text-sm text-red-400 text-center">
          {errorMessage}
        </p>
      )}
    </div>
  );
}
