/**
 * lib/api.ts
 * ==========
 * Centralised Axios instance + typed API call functions.
 *
 * Design decisions:
 *
 * WHY Axios over native fetch:
 *   1. Request interceptors — inject X-API-Key header in ONE place, not in
 *      every component. If the auth scheme changes, only this file changes.
 *   2. Response interceptors — normalise errors into a single ApiError type.
 *      Components never need to check `response.ok` or parse error bodies.
 *   3. Automatic JSON parsing + TypeScript generics.
 *
 * WHY relative base URL ("/api"):
 *   Next.js rewrites proxy /api/* to FastAPI (see next.config.ts).
 *   Components never need to know the backend URL — they just call apiClient.
 *   This also avoids CORS: browser sees same origin.
 *
 * WHY not use React Query's built-in fetch:
 *   React Query handles caching/stale/refetch logic. Axios handles the
 *   transport layer. They're complementary, not competing.
 */

import axios, { type AxiosError } from "axios";

import type {
  ClassifyResponse,
  HistoryListResponse,
  HistorySummary,
  HealthResponse,
  ExploreListResponse,
  AdminFeedbackResponse,
  AdminAnalysesResponse,
  ChatResponse,
  ChatMessageRecord,
  ChatSessionDetail,
  ChatSessionListResponse,
  KbTypeItem,
  KbBrowseResponse,
} from "@/types/api";

// ── Axios instance ────────────────────────────────────────────────────────────

/**
 * apiClient — proxied through Next.js rewrites (/api/* → FastAPI).
 * Used for fast calls: health, history list, history detail.
 */
export const apiClient = axios.create({
  baseURL: "/api",
  timeout: 120_000,   // 2 minutes — generous for API calls
});

/**
 * classifyApiClient — calls FastAPI DIRECTLY (bypasses Next.js proxy).
 *
 * WHY direct for classify only:
 *   The classify pipeline takes 15–60 s when Ollama is generating the LLM
 *   narrative. Next.js Turbopack's reverse proxy has a hard socket timeout
 *   (~30 s) that kills the connection mid-wait, causing ECONNRESET errors.
 *   The browser calling FastAPI directly has no such artificial limit.
 *
 *   This works because FastAPI's CORSMiddleware already lists
 *   http://localhost:3000 in ALLOWED_ORIGINS — preflight passes.
 *
 *   NEXT_PUBLIC_CLASSIFY_URL is set in .env.local:
 *     NEXT_PUBLIC_CLASSIFY_URL=http://127.0.0.1:8000
 *   Falls back to "/api" (proxy) so existing deploys don't break.
 */
const classifyApiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_CLASSIFY_URL ?? "/api",
  timeout: 600_000,   // 10 minutes — covers battery-throttled Ollama (gemma3:4b can take 5+ min on low power)
});

// ── Shared request interceptors ─────────────────────────────────────────────

// ── Auth token cache ─────────────────────────────────────────────────────────

/**
 * Module-level JWT cache.
 *
 * WHY NOT call getSession() inside the interceptor:
 *   getSession() from next-auth/react fires a fetch to /api/auth/session on
 *   EVERY intercepted request. When that endpoint is slow or unreachable,
 *   NextAuth calls console.error("ClientFetchError") INSIDE its own code,
 *   BEFORE throwing — so our try/catch cannot suppress the console noise.
 *
 *   The fix: keep a simple module-level variable that is set ONCE by the
 *   <SessionSync> component whenever the session changes (see
 *   components/auth/SessionSync.tsx). The interceptor reads synchronously
 *   from the cache — zero network overhead, zero console warnings.
 */
let _authToken: string | null = null;

/**
 * Called by <SessionSync> when the NextAuth session updates.
 * Exported so SessionSync can import it without circular deps.
 */
export function setAuthToken(token: string | null): void {
  _authToken = token;
}

/**
 * applyKeyInterceptor — injects the static X-API-Key header if configured.
 * WHY NEXT_PUBLIC_: Next.js only exposes env vars to the browser when they
 *   are prefixed NEXT_PUBLIC_. Server-only vars are inaccessible at runtime.
 */
function applyKeyInterceptor(client: typeof apiClient) {
  client.interceptors.request.use((config) => {
    const key = process.env.NEXT_PUBLIC_API_KEY;
    if (key) config.headers["X-API-Key"] = key;
    return config;
  });
}

/**
 * applyAuthInterceptor — injects the JWT Bearer token from the module-level cache.
 *
 * WHY synchronous (no async):
 *   Reading _authToken is instant — no network call, no Promise chain.
 *   The token is kept fresh by <SessionSync> which watches useSession().
 *
 * WHY only inject when a token exists:
 *   Guest users (not logged in) still use the API (anonymous classify).
 *   If no session, the header is omitted and FastAPI treats the request as
 *   unauthenticated (rate-limited, no history write to a user account).
 *
 * WHY "Bearer" scheme:
 *   FastAPI's auth.deps.get_current_user reads `Authorization: Bearer <jwt>`.
 *   The scheme must match exactly — no "Token" or "JWT" prefix.
 */
function applyAuthInterceptor(client: typeof apiClient) {
  // Synchronous — reads from module-level cache, never fires a network request.
  client.interceptors.request.use((config) => {
    if (_authToken) config.headers["Authorization"] = `Bearer ${_authToken}`;
    return config;
  });
}

applyKeyInterceptor(apiClient);
applyKeyInterceptor(classifyApiClient);
applyAuthInterceptor(apiClient);
applyAuthInterceptor(classifyApiClient);

// ── Normalised error type ─────────────────────────────────────────────────────

/**
 * ApiError wraps Axios errors into a predictable shape for components.
 * WHY: Axios errors contain deeply nested response data. Components should
 * not need to write `err.response?.data?.detail ?? err.message` everywhere.
 */
export class ApiError extends Error {
  readonly status:  number;
  readonly detail:  string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name   = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function toApiError(err: unknown): ApiError {
  const axiosErr = err as AxiosError<{ detail?: string }>;
  const status  = axiosErr.response?.status ?? 0;
  const detail  = axiosErr.response?.data?.detail ?? axiosErr.message ?? "Unknown error";
  return new ApiError(status, detail);
}

// ── API call functions ────────────────────────────────────────────────────────

/**
 * POST /api/classify
 *
 * Upload a coin photograph and run the full DeepCoin pipeline.
 *
 * @param file    JPEG or PNG ≤ 10 MB
 * @param tta     Enable Test-Time Augmentation (default true — +0.78% accuracy)
 * @param onUploadProgress  Optional callback for the upload progress bar (0–100)
 * @returns       Structured ClassifyResponse with CNN result + agent analysis
 */
export async function classifyCoin(
  file:             File,
  tta:              boolean = true,
  onUploadProgress?: (pct: number) => void,
  signal?:          AbortSignal,
): Promise<ClassifyResponse> {
  const form = new FormData();
  form.append("file", file);

  try {
    // Use classifyApiClient (direct to FastAPI) — bypasses Next.js proxy timeout
    const { data } = await classifyApiClient.post<ClassifyResponse>(
      `/api/classify?tta=${tta}`,
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
        signal,
        onUploadProgress: (evt) => {
          if (onUploadProgress && evt.total) {
            onUploadProgress(Math.round((evt.loaded / evt.total) * 100));
          }
        },
      },
    );
    return data;
  } catch (err) {
    // axios wraps AbortController cancellation as a CanceledError
    // — re-throw so the caller can distinguish cancel from real errors
    throw toApiError(err);
  }
}

/**
 * GET /api/history
 *
 * Fetch paginated history (newest first).
 *
 * @param skip   Number of records to skip (for pagination)
 * @param limit  Max records per page (1–100)
 */
export async function getHistory(
  skip  = 0,
  limit = 20,
): Promise<HistoryListResponse> {
  try {
    const { data } = await apiClient.get<HistoryListResponse>(
      `/history?skip=${skip}&limit=${limit}`,
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

/**
 * GET /api/history/{id}
 *
 * Fetch a single past classification by its UUID.
 */
export async function getHistoryItem(id: string): Promise<ClassifyResponse> {
  try {
    const { data } = await apiClient.get<ClassifyResponse>(`/history/${id}`);
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

/**
 * DELETE /api/history/{id}
 *
 * Permanently remove one past classification record.
 * Returns void on success (HTTP 204); throws ApiError on 404 or network error.
 */
export async function deleteHistoryItem(id: string): Promise<void> {
  try {
    await apiClient.delete(`/history/${id}`);
  } catch (err) {
    throw toApiError(err);
  }
}

/**
 * POST /api/history/{id}/feedback
 *
 * Submit a "mark as wrong" correction so the analyst can review
 * misclassifications.  Stores the correction inside the record payload.
 *
 * @param id             - UUID of the classification record to correct.
 * @param correctTypeId  - The CN type ID the user says is correct (e.g. "1015").
 * @param note           - Optional free-text explanation.
 */
export async function submitFeedback(
  id:            string,
  correctTypeId: string,
  note:          string,
): Promise<void> {
  try {
    await apiClient.post(`/history/${id}/feedback`, {
      correct_type_id: correctTypeId,
      note,
    });
  } catch (err) {
    throw toApiError(err);
  }
}

/**
 * GET /api/health
 *
 * Check that FastAPI, the model, RAG engine and ChromaDB are all up.
 * Used by the header status indicator.
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const { data } = await apiClient.get<HealthResponse>("/health");
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

/**
 * Build the absolute URL for a PDF report download.
 *
 * WHY bypass the Next.js proxy:
 *   Next.js Turbopack's reverse-proxy buffers the entire response body before
 *   forwarding it to the browser.  For PDFs (often 400 KB+) this creates a
 *   noticeable delay AND can trip internal timeouts, causing the proxy to
 *   return a JSON error body instead of the binary PDF.  The browser then
 *   saves that error as ".json".
 *
 *   Using the direct FastAPI URL (NEXT_PUBLIC_CLASSIFY_URL base) bypasses
 *   the proxy entirely — the browser fetches the PDF directly from FastAPI,
 *   which streams it with the correct Content-Disposition: attachment header.
 *
 * pdf_url from the API is like "/api/reports/filename.pdf".
 * NEXT_PUBLIC_CLASSIFY_URL is "http://127.0.0.1:8000/api/classify".
 */
const _DIRECT_API_BASE: string = (() => {
  const raw = process.env.NEXT_PUBLIC_CLASSIFY_URL ?? "";
  return raw.replace(/\/api\/classify$/, "").replace(/\/$/, "");
})();

export function pdfDownloadUrl(pdfUrl: string): string {
  // Already a full URL (future-proofing)
  if (pdfUrl.startsWith("http")) return pdfUrl;

  // Defensive: older records may have a full Windows filesystem path embedded
  // inside the /api/reports/ prefix, e.g. "/api/reports/C:/Users/.../report.pdf".
  // This happens when rsplit("/") was used on a backslash Windows path in the
  // backend — the whole path ends up as the "filename" segment.
  // Fix: normalise to forward slashes, take the last segment as the filename.
  let cleanUrl = pdfUrl.replace(/\\/g, "/");
  const reportsPrefix = "/api/reports/";
  if (cleanUrl.startsWith(reportsPrefix)) {
    const afterPrefix = cleanUrl.slice(reportsPrefix.length);
    // If the remainder still contains path separators it's a leaked FS path
    if (afterPrefix.includes("/")) {
      const filename = afterPrefix.split("/").pop()!;
      cleanUrl = `${reportsPrefix}${filename}`;
    }
  }

  // Bypass Next.js proxy — call FastAPI directly for binary file responses
  return `${_DIRECT_API_BASE}${cleanUrl}`;
}
// ── Public explore ────────────────────────────────────────────────────────────

/**
 * GET /api/explore
 *
 * Public gallery — returns recent analyses with NO authentication.
 * Used by the /explore page for anonymous visitors.
 *
 * @param skip   Pagination offset
 * @param limit  Page size (max 50)
 * @param route  Optional route filter ("historian" | "validator" | "investigator")
 */
export async function explorePublic(
  skip   = 0,
  limit  = 12,
  route?: string,
): Promise<ExploreListResponse> {
  try {
    const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
    if (route && route !== "all") params.set("route", route);
    const { data } = await apiClient.get<ExploreListResponse>(`/explore?${params}`);
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

// ── KB browse ────────────────────────────────────────────────────────────────

/**
 * GET /api/kb/types — search or browse all 9,541 CN coin types.
 *
 * WHY classifyApiClient: avoids the Next.js proxy timeout (the search query
 * can hit BM25 + ChromaDB simultaneously, which takes ~300 ms).
 *
 * @param search        Free-text query (denomination, dynasty, region, …)
 * @param skip          Page offset
 * @param limit         Page size (default 20)
 * @param inTrainingSet When true, filter to the 438 CNN-trained types only
 */
export async function browseKb(
  search        = "",
  skip          = 0,
  limit         = 20,
  inTrainingSet = false,
): Promise<KbBrowseResponse> {
  try {
    const params = new URLSearchParams({
      search,
      skip:            String(skip),
      limit:           String(limit),
      in_training_set: String(inTrainingSet),
    });
    const { data } = await classifyApiClient.get<KbBrowseResponse>(
      `/api/kb/types?${params}`,
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

// ── Admin endpoints ─────────────────────────────────────────────────────────────

/**
 * GET /api/admin/feedback (via Next.js proxy at /api/admin/feedback)
 *
 * Fetch paginated user corrections. Requires admin or curator role.
 * Calls the Next.js route handler which server-side proxies to FastAPI.
 */
export async function getAdminFeedback(skip = 0, limit = 20): Promise<AdminFeedbackResponse> {
  try {
    const { data } = await apiClient.get<AdminFeedbackResponse>(
      `/admin/feedback?skip=${skip}&limit=${limit}`,
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

/**
 * GET /api/admin/analyses
 *
 * Paginated full analyses list for all users. Requires admin or curator role.
 */
export async function getAdminAnalyses(
  skip   = 0,
  limit  = 20,
  route?: string,
  search?: string,
): Promise<AdminAnalysesResponse> {
  try {
    const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
    if (route && route !== "all") params.set("route", route);
    if (search && search.trim())  params.set("search", search.trim());
    const { data } = await apiClient.get<AdminAnalysesResponse>(`/admin/analyses?${params}`);
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

// ── AI Chat ───────────────────────────────────────────────────────────────────

/**
 * POST /api/chat
 *
 * Ask the DeepCoin AI a natural-language numismatic question.
 * The answer is grounded in the 9,541-type Corpus Nummorum knowledge base.
 * NO authentication required.
 *
 * @param query     The question (max 500 chars)
 * @param nSources  Number of KB chunks to retrieve (default 5)
 */
export async function chatQuery(
  query: string,
  nSources = 5,
  /** Top-5 CNN predicted CN type IDs — injected as primary context in the backend */
  top5Labels: string[] = [],
): Promise<ChatResponse> {
  // Uses classifyApiClient (direct to FastAPI, 180 s timeout) — same reason as
  // classifyCoin: the LLM call can take 8–20 s; the Next.js proxy would time out.
  // FIX: route is /api/chat (prefix set in chat.py router), was /chat before.
  try {
    const { data } = await classifyApiClient.post<ChatResponse>("/api/chat", {
      query,
      n_sources:   nSources,
      top5_labels: top5Labels,
    });
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

// ── Chat session history ──────────────────────────────────────────────────────
// All chat-session calls are auth-gated — they use apiClient (which attaches
// the NextAuth JWT via applyAuthInterceptor). They go through the Next.js proxy
// because they are small JSON payloads (no binary streaming issues like PDFs).

export interface CreateSessionPayload {
  title:    string;
  messages: ChatMessageRecord[];
}

export interface AppendSessionPayload {
  messages: ChatMessageRecord[];
}

export async function createChatSession(
  payload: CreateSessionPayload
): Promise<ChatSessionDetail> {
  try {
    const { data } = await apiClient.post<ChatSessionDetail>(
      "/chat/sessions",
      payload
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function listChatSessions(
  skip = 0,
  limit = 30
): Promise<ChatSessionListResponse> {
  try {
    const { data } = await apiClient.get<ChatSessionListResponse>(
      "/chat/sessions",
      { params: { skip, limit } }
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function getChatSession(id: string): Promise<ChatSessionDetail> {
  try {
    const { data } = await apiClient.get<ChatSessionDetail>(
      `/chat/sessions/${id}`
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function appendChatSession(
  id: string,
  payload: AppendSessionPayload
): Promise<ChatSessionDetail> {
  try {
    const { data } = await apiClient.patch<ChatSessionDetail>(
      `/chat/sessions/${id}`,
      payload
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function deleteChatSession(id: string): Promise<void> {
  try {
    await apiClient.delete(`/chat/sessions/${id}`);
  } catch (err) {
    throw toApiError(err);
  }
}