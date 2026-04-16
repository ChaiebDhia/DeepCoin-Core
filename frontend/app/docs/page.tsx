/**
 * app/docs/page.tsx
 * ==================
 * Permanent redirect to the FastAPI interactive API documentation.
 *
 * WHY redirect instead of a static docs page:
 *   FastAPI already auto-generates interactive Swagger UI at /api/docs
 *   (enabled in dev; can be enabled in production by setting DOCS_URL).
 *   A hand-written static JSX page would:
 *     1. Diverge from the actual API the moment any endpoint changes.
 *     2. Add maintenance overhead for zero benefit  the live spec is
 *        always up-to-date by definition.
 *     3. Duplicate information that Swagger already presents in a richer,
 *        try-it-live format.
 *
 *   Users who land on /docs are developers looking to explore the API.
 *   The best developer experience is an interactive Swagger UI, not a
 *   static page they have to manually cross-reference with the server.
 *
 * PRODUCTION:
 *   Set NEXT_PUBLIC_CLASSIFY_URL to the production FastAPI root.
 *   The redirect target becomes NEXT_PUBLIC_CLASSIFY_URL + "/docs".
 *   In production deployments where Swagger is intentionally disabled,
 *   change this redirect to a static OpenAPI-driven doc generator such
 *   as Stoplight Elements or Redoc.
 */

import { redirect } from "next/navigation";

/**
 * Redirect to the FastAPI interactive Swagger UI.
 *
 * This page intentionally contains no content  it is a permanent redirect.
 * The `/api/docs` path is proxied by next.config.ts to the FastAPI server,
 * so the browser ultimately lands on the live Swagger UI at
 * http://127.0.0.1:8000/docs (dev) or the production FastAPI host.
 */
export default function DocsPage() {
  // /api/docs is rewritten to the FastAPI /docs route by next.config.ts.
  // Using a relative path keeps the redirect portable across environments.
  redirect("/api/docs");
}
