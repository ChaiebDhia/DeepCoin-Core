/**
 * app/api/auth/[...nextauth]/route.ts
 * ====================================
 * Next-Auth v5 catch-all route handler.
 *
 * This file exposes the GET and POST HTTP handlers that next-auth needs to:
 *   - GET  /api/auth/session         → return the current session (used by useSession)
 *   - GET  /api/auth/csrf            → return the CSRF token
 *   - POST /api/auth/callback/credentials → handle the sign-in form submission
 *   - POST /api/auth/signout         → handle sign-out
 *
 * WHY "use server" is NOT needed here:
 *   Route handlers in the App Router are already server-side by definition.
 *   They run in Node.js, not the browser, so all imports are server-safe.
 *
 * WHY both GET and POST:
 *   GET  — session reads, CSRF token, provider redirects
 *   POST — form submissions (credentials sign-in, sign-out)
 */

// NextAuth v5 pattern: destructure GET / POST from the `handlers` export.
// `auth.ts` exports `{ handlers, auth, signIn, signOut }` via NextAuth().
// `handlers` is an object { GET: RouteHandler, POST: RouteHandler }.
// We re-export them as named exports so Next.js App Router wires them up.
import { handlers } from "../../../../auth";
export const { GET, POST } = handlers;
