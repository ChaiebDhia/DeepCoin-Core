"use client";

/**
 * providers.tsx
 * =============
 * Root client-side providers wrapper.
 *
 * WHY "use client" here but not in layout.tsx:
 *   Next.js App Router renders layout.tsx as a React Server Component by default.
 *   TanStack QueryClientProvider requires a client context (uses React state internally).
 *   The pattern is: keep layout.tsx as a Server Component and wrap only the
 *   providers that NEED client context in a dedicated "use client" file.
 *   This way, the shell (fonts, metadata, HTML structure) stays server-rendered.
 *
 * WHY staleTime: 60_000:
 *   History results don't change second-to-second. A 60 s stale window means
 *   navigating back to the history page doesn't trigger an unnecessary refetch
 *   if the user just visited it.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools }               from "@tanstack/react-query-devtools";
import { useState, type ReactNode }         from "react";
import { Toaster }                          from "react-hot-toast";
import { SessionProvider }                  from "next-auth/react";
import { SessionSync }                      from "@/components/auth/SessionSync";

/**
 * WHY SessionProvider here:
 *   useSession() and signOut() (used in UserMenu) are React Context hooks.
 *   They need a <SessionProvider> ancestor to read the session JWT.
 *   We add it at the root providers level so every page/component can call
 *   useSession() without wrapping pages individually.
 *   The session object is populated by NextAuth's /api/auth/* route.
 */

interface ProvidersProps {
  children: ReactNode;
}

export default function Providers({ children }: ProvidersProps) {
  /**
   * WHY useState to create QueryClient instead of module-level:
   *   Each Next.js request must get its own QueryClient to avoid sharing
   *   cached server-state between users. useState ensures a new instance
   *   per component mount (i.e. per browser tab), not per build.
   */
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60_000,   // 60 s — history doesn't change frequently
            retry:     1,        // one retry on network failure
          },
        },
      }),
  );

  return (
    <SessionProvider>
      {/* Keeps the module-level auth token cache in lib/api.ts in sync.
          This eliminates Console ClientFetchError: getSession() was called
          in every Axios interceptor; now the token is read synchronously
          from a cache that SessionSync updates when the session changes. */}
      <SessionSync />
    <QueryClientProvider client={queryClient}>
      {children}

      {/* Toast notifications (errors, PDF ready, etc.) */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: "#1e2a3a",
            color:      "#e2e8f0",
            border:     "1px solid #2d3f55",
            fontFamily: "var(--font-geist-sans)",
            fontSize:   "14px",
          },
          success: { iconTheme: { primary: "#22c55e", secondary: "#1e2a3a" } },
          error:   { iconTheme: { primary: "#ef4444", secondary: "#1e2a3a" } },
        }}
      />

      {/* DevTools — only in development; suppressed in production builds */}
      {process.env.NODE_ENV === "development" && <ReactQueryDevtools initialIsOpen={false} buttonPosition="bottom-left" />}
    </QueryClientProvider>
    </SessionProvider>
  );
}
