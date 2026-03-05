/**
 * app/forgot-password/page.tsx
 * =============================
 * /forgot-password — allows a user to request a password reset link.
 *
 * DESIGN:
 *   - Server Component shell provides metadata and the page layout.
 *   - ForgotPasswordForm is a Client Component (needs event handlers).
 *   - Suspense boundary required because the inner form reads searchParams.
 *
 * FLOW:
 *   1. User enters their email address and submits.
 *   2. POST /auth/forgot-password sends a one-time reset link to their inbox.
 *   3. The server ALWAYS returns 200 — no email enumeration possible.
 *   4. A branded success state replaces the form.
 *
 * WHY server always returns 200:
 *   If we returned 404 when an email doesn't exist, an attacker could probe
 *   which emails have accounts.  The identical success message prevents this.
 */

import type { Metadata } from "next";
import { Suspense }      from "react";
import ForgotPasswordForm from "./ForgotPasswordForm";

export const metadata: Metadata = {
  title:       "Forgot Password · DeepCoin",
  description: "Reset your DeepCoin account password.",
};

export default function ForgotPasswordPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)] py-12">
      <Suspense>
        <ForgotPasswordForm />
      </Suspense>
    </div>
  );
}
