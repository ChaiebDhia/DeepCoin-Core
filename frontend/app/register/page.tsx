/**
 * app/register/page.tsx
 * =====================
 * /register — the account creation page.
 *
 * WHY Server Component wrapper:
 *   Lets Next.js generate the page metadata on the server while the
 *   interactive form (RegisterForm) is a Client Component.
 */

import type { Metadata } from "next";
import { RegisterForm }  from "@/components/auth/RegisterForm";

export const metadata: Metadata = {
  title:       "Create Account · DeepCoin",
  description: "Register for a DeepCoin account to save and manage your coin classification history.",
};

export default function RegisterPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)] py-12">
      <RegisterForm />
    </div>
  );
}
