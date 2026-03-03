/**
 * app/page.tsx — Homepage (Server Component)
 * =============================================
 * WHY Server Component:
 *   All sections above the fold are purely static — no client state, no hooks.
 *   The interactive analyser lives in AnalyseSection (a client island), so we
 *   get zero-JS rendering for Hero, PipelineSteps, StatsBar, ValueCards,
 *   ForWhoCards, Testimonials, and EmailCapture at no cost.
 *
 * HOW it fits:
 *   layout.tsx wraps this in <main> with max-w-6xl px-5 (no py-8 anymore, so
 *   the hero can start at the very top edge of the content area).
 */

import { HeroSection }    from "@/components/home/HeroSection";
import { PipelineSteps }  from "@/components/home/PipelineSteps";
import { StatsBar }       from "@/components/home/StatsBar";
import { ValueCards }     from "@/components/home/ValueCards";
import { ForWhoCards }    from "@/components/home/ForWhoCards";
import { TechStack }      from "@/components/home/TechStack";
import { EmailCapture }   from "@/components/home/EmailCapture";

export default function HomePage() {
  return (
    <>
      <HeroSection />
      <PipelineSteps />
      <StatsBar />
      <ValueCards />
      <ForWhoCards />
      <TechStack />
      <EmailCapture />
    </>
  );
}
