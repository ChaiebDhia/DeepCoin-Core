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

// P16 — Canvas downsize target.
// WHY 1024px: The server applies CLAHE then resizes to 299×299. Sending an
// 8MP phone photo (4032×1960, ~4 MB) wastes bandwidth and server memory.
// 1024×1024 is ≥10× more detail than the model needs, encodes at ~150 KB,
// and cuts upload time proportionally. The CLAHE preprocessing still works
// correctly on 1024px — clip limit and tile size are relative to image size.
const DOWNSIZE_MAX_PX = 1024;

/**
 * Downscale a coin image client-side using an offscreen Canvas.
 *
 * WHAT: If the image exceeds DOWNSIZE_MAX_PX in either dimension, draws it
 *   onto a canvas at the reduced size (aspect-preserved) and re-exports as
 *   JPEG quality 0.85. Returns the original File unchanged if it is already
 *   small enough, or if canvas is unavailable (SSR guard).
 *
 * WHY quality=0.85: Visually lossless for photographic coin images.
 *   Quality 1.0 produces bloated files; 0.7 introduces visible artefacts
 *   on fine details (inscription serifs, mint marks).
 *
 * @param file   Original image file from the file input or drag-drop.
 * @param maxPx  Maximum pixel dimension for width or height. Default 1024.
 */
async function downsizeImage(file: File, maxPx: number = DOWNSIZE_MAX_PX): Promise<File> {
  // SSR guard — canvas is browser-only
  if (typeof window === "undefined" || typeof document === "undefined") return file;

  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);

      const { width, height } = img;
      // Already within limits — skip resizing entirely
      if (width <= maxPx && height <= maxPx) {
        resolve(file);
        return;
      }

      // Aspect-preserving scale factor
      const scale  = maxPx / Math.max(width, height);
      const dstW   = Math.round(width  * scale);
      const dstH   = Math.round(height * scale);

      const canvas = document.createElement("canvas");
      canvas.width  = dstW;
      canvas.height = dstH;
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(file); return; }

      ctx.drawImage(img, 0, 0, dstW, dstH);

      canvas.toBlob(
        (blob) => {
          if (!blob) { resolve(file); return; }
          // Preserve original filename but change extension to .jpg
          const resizedName = file.name.replace(/\.[^.]+$/, "") + "_resized.jpg";
          resolve(new File([blob], resizedName, { type: "image/jpeg" }));
        },
        "image/jpeg",
        0.85,
      );
    };

    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

// ── Image quality analyser ──────────────────────────────────────────────────

/**
 * Analyse a coin image for two quality issues and return human-readable warnings.
 *
 * WHAT it checks:
 *   1. Minimum resolution — < 100×100 px is too small for reliable inference
 *   2. Sharpness — Laplacian variance on a 96×96 greyscale canvas.
 *      The Laplacian kernel [0,1,0; 1,-4,1; 0,1,0] is an edge-detector:
 *      a sharp image has strong edges (high variance); a blurry image has
 *      near-zero edges (low variance). Threshold < 60 = noticeably blurry.
 *
 * WHY soft warnings, not blockers:
 *   The 3-route pipeline handles low-quality images gracefully — blurry photos
 *   route to the Investigator which still produces a KB-grounded report.
 *   Blocking uploads would harm the system's core "accept anything" promise.
 *
 * @returns Array of warning strings (empty = no issues detected).
 */
async function analyseImageQuality(file: File): Promise<string[]> {
  if (typeof window === "undefined") return [];
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const warnings: string[] = [];
      const { naturalWidth: W, naturalHeight: H } = img;

      // ── Check 1: minimum resolution ──
      if (W < 100 || H < 100) {
        warnings.push(`Image is very small (${W}×${H} px) — results may be unreliable.`);
      }

      // ── Check 2: sharpness via Laplacian variance on 96×96 canvas ──
      const SIZE = 96;
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = SIZE;
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(warnings); return; }
      ctx.drawImage(img, 0, 0, SIZE, SIZE);
      const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

      // Greyscale conversion (luminance weights: Rec. 601)
      const gray = new Float32Array(SIZE * SIZE);
      for (let i = 0; i < SIZE * SIZE; i++) {
        const p = i * 4;
        gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      }

      // Laplacian [0,1,0; 1,-4,1; 0,1,0] — compute variance of responses
      let sum = 0, sumSq = 0, n = 0;
      for (let y = 1; y < SIZE - 1; y++) {
        for (let x = 1; x < SIZE - 1; x++) {
          const lap =
            gray[(y - 1) * SIZE + x] + gray[(y + 1) * SIZE + x] +
            gray[y * SIZE + (x - 1)] + gray[y * SIZE + (x + 1)] -
            4 * gray[y * SIZE + x];
          sum   += lap;
          sumSq += lap * lap;
          n++;
        }
      }
      const mean     = sum / n;
      const variance = sumSq / n - mean * mean;

      // < 60 is noticeably blurry on coin photographs
      if (variance < 60) {
        warnings.push("Image appears blurry — a sharper photo will improve accuracy.");
      }

      resolve(warnings);
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve([]); };
    img.src = url;
  });
}

export function CoinUploader() {
  const {
    tta, setTta,
    phase,
    uploadProgress,
    selectedFile,
    errorMessage,
    setSelectedFile, setUploadProgress, setPhase, setResult, setError, reset, setCancelFn,
  } = useDeepCoinStore();

  const [isDragging,      setIsDragging]      = useState(false);
  const [qualityWarnings, setQualityWarnings] = useState<string[]>([]);
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
    setQualityWarnings([]);      // clear any previous warnings
    setSelectedFile(file);
    // Run quality check in background — does not block the UI
    analyseImageQuality(file).then(setQualityWarnings);
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

      // P16 — downscale large photos client-side before upload.
      // Reduces upload payload from ~4 MB (phone photo) to ~150 KB (1024px JPEG).
      // The server still applies CLAHE + 299×299 resize, so quality is unaffected.
      const fileToSend = await downsizeImage(selectedFile);

      const result = await classifyCoin(fileToSend, tta, (pct) => {
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
              onClick={(e) => { e.stopPropagation(); abortRef.current?.abort(); abortRef.current = null; setQualityWarnings([]); reset(); }}
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
            {/* Photography guidance — visible in idle state, hidden while dragging */}
            {!isDragging && (
              <div className="w-full rounded-lg px-3 py-2.5 text-left"
                style={{ background: "rgba(59,130,246,0.07)", border: "1px solid rgba(59,130,246,0.15)" }}>
                <p className="text-xs font-semibold text-blue-300 mb-2">📸 For best results</p>
                <ul className="space-y-1">
                  {([
                    ["🪙", "Photograph the obverse — the portrait or main inscription side"],
                    ["💡", "Even lighting, no glare — natural light works best"],
                    ["📐", "Coin flat and in focus, fully visible in the frame"],
                  ] as [string, string][]).map(([emoji, text]) => (
                    <li key={text} className="text-xs text-[var(--text-secondary)] flex gap-2">
                      <span>{emoji}</span><span>{text}</span>
                    </li>
                  ))}
                </ul>
                <p className="text-xs text-[var(--text-muted)] mt-2 italic">
                  Any quality works — the system automatically routes to the right specialist.
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Quality warnings — shown after file is selected if issues detected */}
      {hasFile && qualityWarnings.length > 0 && (
        <div className="rounded-lg px-3 py-2.5 text-xs"
          style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.25)" }}>
          <p className="font-semibold text-amber-300 mb-1.5">⚠ Image quality notice</p>
          <ul className="space-y-0.5">
            {qualityWarnings.map((w, i) => (
              <li key={i} className="text-amber-200/80">{w}</li>
            ))}
          </ul>
          <p className="text-[var(--text-muted)] mt-1.5">
            The system will still analyse this image — accuracy may be lower than usual.
          </p>
        </div>
      )}

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
