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
import { getSession }              from "next-auth/react";

import type {
  ClassifyResponse,
  HistoryListResponse,
  HistorySummary,
  HealthResponse,
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
 * applyAuthInterceptor — injects the JWT Bearer token from the NextAuth session.
 *
 * WHY async interceptor:
 *   getSession() is async — it reads the session cookie / JWT from the
 *   browser. It resolves immediately when the session is cached (no network
 *   call), so the interceptor adds negligible latency.
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
  client.interceptors.request.use(async (config) => {
    const session = await getSession();
    const token   = (session?.user as { access_token?: string })?.access_token;
    if (token) config.headers["Authorization"] = `Bearer ${token}`;
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
 * pdf_url from the API is like "/api/reports/filename.pdf".
 */
export function pdfDownloadUrl(pdfUrl: string): string {
  // Already a full URL (future-proofing)
  if (pdfUrl.startsWith("http")) return pdfUrl;
  // Relative — works because /api/* is proxied by Next.js rewrites
  return pdfUrl;
}
