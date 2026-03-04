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
  /** Whether Test-Time Augmentation was applied. */
  tta_used:           boolean;
  /**
   * Fraction of TTA passes (0.0–1.0) that independently selected the same
   * top-1 class. null when TTA was not used.
   * WHY this matters: a coin at 8% confidence with vote_fraction=1.0 means
   * every augmented view agreed — the low softmax is a photo-quality artefact,
   * not classification uncertainty. The UI surfaces this as "TTA Consensus".
   */
  vote_fraction:      number | null;
  /** Number of TTA forward passes actually performed (1 when TTA off). */
  tta_passes:         number;
  /** Temperature scalar T used in softmax(z/T). 1.0 = no calibration applied. */
  temperature:        number;
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
// ── Public explore types ──────────────────────────────────────────────────────

/** One analysis item from the public /api/explore gallery (no user PII). */
export interface ExploreItem {
  id:          string;
  created_at:  string | null;
  route_taken: string;
  label:       string;
  confidence:  number | null;
}

export interface ExploreListResponse {
  items: ExploreItem[];
  total: number;
  skip:  number;
  limit: number;
}

// ── Admin types ───────────────────────────────────────────────────────────────

/** One "mark as wrong" correction as seen by admins. */
export interface FeedbackItem {
  id:                string;
  created_at:        string | null;
  classification_id: string;
  coin_label:        string | null;
  confidence:        number | null;
  route_taken:       string | null;
  correct_type_id:   string;
  note:              string | null;
  submitted_by:      string;
}

export interface AdminFeedbackResponse {
  items: FeedbackItem[];
  total: number;
  skip:  number;
  limit: number;
  pages: number;
}

/** One analysis row in the admin analyses table. */
export interface AdminAnalysisItem {
  id:          string;
  created_at:  string | null;
  label:       string;
  confidence:  number;
  route_taken: string;
  pdf_url:     string | null;
  user_email:  string;
}

export interface AdminAnalysesResponse {
  items: AdminAnalysisItem[];
  total: number;
  skip:  number;
  limit: number;
  pages: number;
}

/** One user row in the admin users table. */
export interface AdminUserItem {
  id:               string;
  email:            string;
  display_name:     string | null;
  role:             "admin" | "curator" | "analyst";
  status:           "pending" | "active" | "suspended";
  created_at:       string | null;
  last_login_at:    string | null;
  analyses_count:   number;
}

export interface AdminUsersResponse {
  items: AdminUserItem[];
  total: number;
  skip:  number;
  limit: number;
  pages: number;
}

// ── Chat types ────────────────────────────────────────────────────────────────

export interface ChatSource {
  type_id:    string;
  chunk_type: string;
  snippet:    string;
  score:      number;
}

export interface ChatResponse {
  answer:   string;
  sources:  ChatSource[];
  provider: string;
}

// ── Chat session history types ────────────────────────────────────────────────

export interface ChatMessageRecord {
  role:     "user" | "assistant";
  content:  string;
  sources?: ChatSource[];
  provider?: string;
}

export interface ChatSessionSummary {
  id:         string;
  title:      string;
  created_at: string;
  updated_at: string;
  msg_count:  number;
}

export interface ChatSessionDetail {
  id:         string;
  title:      string;
  messages:   ChatMessageRecord[];
  created_at: string;
  updated_at: string;
}

export interface ChatSessionListResponse {
  items: ChatSessionSummary[];
  total: number;
  skip:  number;
  limit: number;
}

// ── KB browse types ───────────────────────────────────────────────────────────

/** One coin type from the Corpus Nummorum knowledge base (9,541 types). */
export interface KbTypeItem {
  type_id:         string;
  denomination:    string;
  region:          string;
  date_range:      string;
  material:        string;
  mint:            string;
  authority:       string;
  in_training_set: boolean;
  text_snippet:    string;
}

export interface KbBrowseResponse {
  items:       KbTypeItem[];
  total:       number;
  search_used: boolean;
}