/**
 * app/analyse/page.tsx — Dedicated coin analyser page
 * =====================================================
 * WHY a separate page instead of embedding on homepage:
 *   The homepage is now a pure marketing/landing page (Server Component).
 *   Keeping the interactive analyser on /analyse gives it a clean, focused
 *   URL that users can bookmark, share, and navigate to directly from the nav.
 *
 * HOW it fits:
 *   - Header "Analyse" nav link → /analyse
 *   - Hero CTA "Analyse your coin →" → /analyse
 *   - AnalyseSection is a "use client" island (Zustand + CoinUploader)
 *   - This page shell is a Server Component — zero extra JS cost
 */

import { getTranslations } from "next-intl/server";
import { AnalyseSection } from "@/components/home/AnalyseSection";

export async function generateMetadata() {
  const t = await getTranslations("AnalysePage");
  return {
    title: t("title"),
    description: t("description"),
  };
}

export default function AnalysePage() {
  return (
    <div className="py-8">
      <AnalyseSection />
    </div>
  );
}
