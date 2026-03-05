/**
 * app/confirm-subscription/page.tsx
 * ====================================
 * Permanent redirect to the home page.
 *
 * WHY this page was removed:
 *   The original design sent a double opt-in confirmation email via SMTP
 *   containing a unique token link that landed here.  That flow was removed
 *   in commit 8a820b4 when the EmailCapture was simplified to a single-step
 *   waitlist (no SMTP in the PFE environment).  Since no email is ever sent
 *   with a token URL, this page is unreachable from any real user flow.
 *
 *   Dead routes add maintenance overhead and confuse future developers who
 *   discover them in the file system.  The page is kept as a redirect rather
 *   than deleted to gracefully handle any bookmarked or search-engine-indexed
 *   URLs that might still exist.
 *
 * IF you later add Resend SMTP:
 *   1. Restore the confirm page logic (see git history, commit 391e62e)
 *   2. Wire the confirm link into the welcome email template in
 *      src/api/routes/subscribers.py _send_welcome_email()
 *   3. Remove this redirect
 */

import { redirect } from "next/navigation";

export default function ConfirmSubscriptionPage() {
  redirect("/");
}
