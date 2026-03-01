/**
 * types/api.ts
 * ============
 * TypeScript mirror of every Pydantic v2 model in src/api/schemas.py.
 *
 * WHY keep this in sync manually rather than auto-generating:
 *   Auto-generation (openapi-typescript, etc.) adds a build step and
 *   a CI dependency. For a single student project, manual mirroring is
 *   simpler and teaches the contract discipline. If the team grows,
 *   switch to: npx openapi-typescript http://localhost:8000/openapi.json
 *
 * Rule: Every field name and type here MUST match the Pydantic schema.
 *   Mismatches cause silent runtime errors (undefined instead of null).
 */

// ── CNN sub-models ────────────────────────────────────────────────────────────

/** One entry in the CNN top-5 prediction list. */
export interface Top5Item {
  /** 1 = best match */
  rank:        number;
  /** CNN sort-order index (0–437) */
  class_id:    number;
  /** CN type ID string, e.g. "1015" */
  label:       string;
  /** Softmax probability 0.0–1.0 */
  confidence:  number;
}

/** Raw output from the EfficientNet-B3 classifier. */
export interface CnnResult {
  class_id:           number;
  label:              string;
  confidence:         number;
  top5:               Top5Item[];
  inference_time_ms:  number;
  /** Whether Test-Time Augmentation (5 passes) was applied. */
  tta_used:           boolean;
}

// ── Main classify response ────────────────────────────────────────────────────

/**
 * Full response from POST /api/classify.
 * Mirrors ClassifyResponse in schemas.py exactly.
 */
export interface ClassifyResponse {
  id:               string;
  timestamp:        string;
  image_filename:   string;
  route_taken:      "historian" | "validator" | "investigator";
  cnn:              CnnResult;

  // Historian / Investigator narrative fields
  narrative:           string | null;
  mint:                string | null;
  region:              string | null;
  date_range:          string | null;
  material:            string | null;
  denomination:        string | null;

  // Validator fields
  material_status:     string | null;   // "consistent" | "mismatch" | "unknown"
  material_confidence: number | null;

  // Investigator fields
  visual_description:  string | null;
  kb_match_count:      number | null;

  // Output
  pdf_url:             string | null;
  processing_time_s:   number;
}

// ── History models ────────────────────────────────────────────────────────────

/** Compact row for the history list table. */
export interface HistorySummary {
  id:             string;
  timestamp:      string;
  image_filename: string;
  route_taken:    string;
  label:          string;
  confidence:     number;
  pdf_url:        string | null;
}

/** Paginated history list response. */
export interface HistoryListResponse {
  items: HistorySummary[];
  total: number;
  skip:  number;
  limit: number;
}

// ── Health model ──────────────────────────────────────────────────────────────

export interface HealthResponse {
  // FastAPI returns "healthy" (not "ok") — keeping "ok" for fallback compat
  status:  "healthy" | "degraded" | "ok";
  version: string;
  components: Record<string, string>;
  uptime_s?: number;
}
