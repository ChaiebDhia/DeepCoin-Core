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

import type { Metadata } from "next";
import { AnalyseSection } from "@/components/home/AnalyseSection";

export const metadata: Metadata = {
  title:       "Analyse a Coin · DeepCoin",
  description: "Upload a photograph of an ancient coin. DeepCoin classifies it against 9,716 Corpus Nummorum types and generates a professional PDF report.",
};

export default function AnalysePage() {
  return (
    <div className="py-8">
      <AnalyseSection />
    </div>
  );
}
