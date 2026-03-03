/**
 * app/login/page.tsx
 * ==================
 * /login — the sign-in page.
 *
 * WHY a Server Component wrapper:
 *   - Metadata (title, description) is generated server-side
 *   - The LoginForm itself is a Client Component (needs event handlers)
 *   - This pattern lets Next.js statically optimise the page shell while
 *     the interactive form is hydrated on the client
 *
 * WHY Suspense boundary around LoginForm:
 *   LoginForm uses useSearchParams() to read ?callbackUrl=.
 *   Next.js requires components using useSearchParams to be wrapped in
 *   <Suspense> — otherwise the entire page would bail out of static rendering.
 */

import type { Metadata }    from "next";
import { Suspense }         from "react";
import { LoginForm }        from "@/components/auth/LoginForm";

export const metadata: Metadata = {
  title:       "Sign In · DeepCoin",
  description: "Sign in to your DeepCoin account to access your classification history.",
};

export default function LoginPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)] py-12">
      <Suspense>
        <LoginForm />
      </Suspense>
    </div>
  );
}
