/**
 * auth.ts
 * =======
 * Next-Auth v5 main entry point.
 *
 * WHY export { handlers, auth, signIn, signOut }:
 *   - handlers:  GET + POST HTTP handlers for /api/auth/[...nextauth]
 *   - auth:      The session getter — use in Server Components, API routes,
 *                and middleware to get the current session
 *   - signIn:    Server Action to trigger sign-in (used by LoginForm)
 *   - signOut:   Server Action to trigger sign-out (used by UserMenu)
 *
 * USAGE EXAMPLES:
 *
 *   // Server Component
 *   const session = await auth()
 *   if (!session) redirect("/login")
 *
 *   // Route Handler
 *   export const { GET, POST } = handlers
 *
 *   // Client Component ("use client")
 *   const { data: session } = useSession()
 */

import NextAuth from "next-auth";
import { authConfig } from "./auth.config";

export const { handlers, auth, signIn, signOut } = NextAuth(authConfig);
