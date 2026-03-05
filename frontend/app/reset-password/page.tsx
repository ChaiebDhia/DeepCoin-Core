/**
 * app/reset-password/page.tsx
 * ============================
 * /reset-password?token=<one-time-token>
 * This URL is embedded in the password reset email and opened by the user.
 *
 * SERVER COMPONENT shell — Suspense wraps the client form because the
 * form needs to read the ?token= query param via useSearchParams().
 */

import type { Metadata } from "next";
import { Suspense }      from "react";
import ResetPasswordForm  from "./ResetPasswordForm";

export const metadata: Metadata = {
  title:       "Reset Password · DeepCoin",
  description: "Set a new password for your DeepCoin account.",
};

export default function ResetPasswordPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)] py-12">
      <Suspense>
        <ResetPasswordForm />
      </Suspense>
    </div>
  );
}
