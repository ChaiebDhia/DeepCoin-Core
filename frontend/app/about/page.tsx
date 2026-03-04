/**
 * app/about/page.tsx — About DeepCoin
 * =====================================
 * Server Component — fully static, zero JS shipped to the browser.
 *
 * WHAT:  Tells the story behind the project:
 *          - Mission & academic context (PFE × YEBNI × ESPRIT)
 *          - The technical philosophy (graceful degradation > confident failure)
 *          - End-to-end pipeline overview with visual cards
 *          - CNN model card with key metrics
 *          - Open-source invitation with GitHub link
 *
 * WHY Server Component:
 *   All content is static text and icons — no browser APIs, no hooks.
 *   Server rendering means the page reaches Google's crawler as raw HTML,
 *   boosting SEO with zero JavaScript cost.
 */

import Link      from "next/link";
import { Github, BookOpen, Cpu, Database, Layers, ShieldCheck,
         ExternalLink, GraduationCap, Building2, FlaskConical, Award } from "lucide-react";

/* ── static data ──────────────────────────────────────────────────────────── */

const PIPELINE_STEPS = [
  {
    step:  "01",
    title: "Upload a Photo",
    body:  "Any photograph of an ancient coin — museum archive, field photo, or degraded specimen — is accepted. DeepCoin preprocesses it with CLAHE in LAB colour space to balance contrast without destroying the metal patina.",
    icon:  Layers,
    color: "#3b82f6",
  },
  {
    step:  "02",
    title: "CNN Classification",
    body:  "EfficientNet-B3 (12 M parameters, ImageNet pretrained) processes the 299×299 normalised image and returns a top-5 confidence distribution across 438 trained coin types. Test-Time Augmentation (8 passes) boosts accuracy to 80.03%.",
    icon:  Cpu,
    color: "#8b5cf6",
  },
  {
    step:  "03",
    title: "Confidence Routing",
    body:  "A LangGraph state machine routes the CNN output to the right specialist agent: Historian (high confidence), Validator (mid-range), or Investigator (low confidence / unknown coin). Every route leads to a complete analysis — never a failure.",
    icon:  ShieldCheck,
    color: "#d4a853",
  },
  {
    step:  "04",
    title: "RAG + LLM Narrative",
    body:  "Hybrid BM25 + vector search (47,705 chunks, 9,541 CN types) retrieves the relevant structured records. Gemini or Ollama writes a 3-paragraph grounded narrative, citing each [CONTEXT N] block — zero hallucination on structured facts.",
    icon:  Database,
    color: "#10b981",
  },
];

const METRICS = [
  { label: "CNN Top-1 (TTA ×8)", value: "80.03%",   color: "#8b5cf6" },
  { label: "Training images",     value: "7,677",    color: "#3b82f6" },
  { label: "Coin types (CNN)",    value: "438",       color: "#3b82f6" },
  { label: "KB coin types",       value: "9,541",    color: "#10b981" },
  { label: "RAG chunks",          value: "47,705",   color: "#10b981" },
  { label: "PDF latency",         value: "<500 ms",  color: "#d4a853" },
];

const TEAM = [
  {
    icon:  GraduationCap,
    label: "Student Engineer",
    value: "Dhia Chaieb",
    color: "#d4a853",
    sub:   "ESPRIT School of Engineering",
  },
  {
    icon:  Building2,
    label: "Host Company",
    value: "YEBNI",
    color: "#3b82f6",
    sub:   "ICT, Tunis, Tunisia",
  },
  {
    icon:  FlaskConical,
    label: "Programme",
    value: "PFE Internship",
    color: "#8b5cf6",
    sub:   "February – July 2026",
  },
  {
    icon:  Award,
    label: "Dataset",
    value: "Corpus Nummorum v1",
    color: "#10b981",
    sub:   "115 k images · DFG-funded",
  },
];

/* ── component ────────────────────────────────────────────────────────────── */

export default function AboutPage() {
  return (
    <div className="py-12 max-w-4xl space-y-20">

      {/* Hero */}
      <section className="space-y-4">
        <p className="text-xs font-black uppercase tracking-widest" style={{ color: "var(--brand-gold)" }}>
          About the Project
        </p>
        <h1 className="text-3xl sm:text-4xl font-black leading-tight" style={{ color: "var(--text-primary)" }}>
          Why we built{" "}
          <span style={{ color: "var(--brand-gold)" }}>DeepCoin</span>
        </h1>
        <p className="text-base leading-relaxed max-w-2xl" style={{ color: "var(--text-secondary)" }}>
          Ancient coin classification is brutally hard. Worn surfaces, oxidation, fragmentary legends, and
          9,716 known types across centuries of minting make even expert numismatists uncertain.
          DeepCoin combines deep learning with multi-agent AI so that{" "}
          <strong style={{ color: "var(--text-primary)" }}>no coin is left without an answer</strong> —
          even when the CNN cannot classify it with confidence.
        </p>
      </section>

      {/* Philosophy */}
      <section className="rounded-2xl border p-8 space-y-4"
               style={{ borderColor: "rgba(212,168,83,0.3)", backgroundColor: "rgba(212,168,83,0.04)" }}>
        <h2 className="text-lg font-bold" style={{ color: "var(--brand-gold)" }}>
          The core philosophy
        </h2>
        <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          <strong style={{ color: "var(--text-primary)" }}>Failing gracefully is better than failing confidently.</strong>
          {" "}A traditional classifier returns a class label and confidence score — but what happens with coins
          that were never in the training set? It returns a wrong answer with high confidence, which is worse
          than useless for a museum curator.
        </p>
        <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          DeepCoin routes low-confidence predictions to a{" "}
          <strong style={{ color: "var(--text-primary)" }}>Visual Investigator agent</strong> that uses a
          vision-language model (or a pure-OpenCV fallback) to describe the coin&apos;s attributes,
          then searches a 9,541-type knowledge base for the closest cultural neighbours.
          The system always returns maximum useful information — never a blank screen.
        </p>
      </section>

      {/* Pipeline */}
      <section className="space-y-6">
        <h2 className="text-xl font-black" style={{ color: "var(--text-primary)" }}>End-to-end pipeline</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {PIPELINE_STEPS.map(({ step, title, body, icon: Icon, color }) => (
            <div
              key={step}
              className="rounded-2xl border p-6 space-y-3"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
            >
              <div className="flex items-center gap-3">
                <span className="text-xs font-black tabular-nums" style={{ color }}>
                  {step}
                </span>
                <Icon size={16} style={{ color }} />
                <h3 className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>{title}</h3>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                {body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Metrics */}
      <section className="space-y-4">
        <h2 className="text-xl font-black" style={{ color: "var(--text-primary)" }}>Key metrics</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {METRICS.map(({ label, value, color }) => (
            <div
              key={label}
              className="rounded-xl border p-5 text-center"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
            >
              <p className="text-2xl font-black tabular-nums" style={{ color }}>{value}</p>
              <p className="text-[11px] mt-1" style={{ color: "var(--text-muted)" }}>{label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Team / context */}
      <section className="space-y-4">
        <h2 className="text-xl font-black" style={{ color: "var(--text-primary)" }}>Context</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {TEAM.map(({ icon: Icon, label, value, color, sub }) => (
            <div
              key={label}
              className="rounded-xl border p-5"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
            >
              <Icon size={16} className="mb-3" style={{ color }} />
              <p className="text-[10px] font-medium uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                {label}
              </p>
              <p className="text-sm font-bold mt-1" style={{ color: "var(--text-primary)" }}>{value}</p>
              <p className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>{sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Open source CTA */}
      <section
        className="rounded-2xl border p-8 flex flex-col sm:flex-row items-start sm:items-center
                   justify-between gap-6"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
      >
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Github size={18} style={{ color: "var(--text-primary)" }} />
            <h2 className="font-bold text-base" style={{ color: "var(--text-primary)" }}>Open source</h2>
          </div>
          <p className="text-sm max-w-md" style={{ color: "var(--text-secondary)" }}>
            The full codebase — CNN training, agents, FastAPI backend, Next.js frontend,
            and Docker stack — is available on GitHub. Issues and PRs are welcome.
          </p>
        </div>
        <div className="flex flex-col gap-3 shrink-0">
          <a
            href="https://github.com/ChaiebDhia/DeepCoin-Core"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-opacity hover:opacity-80"
            style={{ backgroundColor: "var(--surface-2)", color: "var(--text-primary)", border: "1px solid var(--border)" }}
          >
            View on GitHub <ExternalLink size={13} />
          </a>
          <a
            href="https://github.com/ChaiebDhia/DeepCoin-Core/blob/main/ENGINEERING_JOURNAL.md"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-opacity hover:opacity-80"
            style={{ backgroundColor: "rgba(212,168,83,0.1)", color: "var(--brand-gold)", border: "1px solid rgba(212,168,83,0.3)" }}
          >
            <BookOpen size={13} /> Engineering Journal
          </a>
        </div>
      </section>

      {/* CTA */}
      <section className="text-center space-y-4 pb-8">
        <h2 className="text-xl font-black" style={{ color: "var(--text-primary)" }}>
          Ready to try it?
        </h2>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Upload a coin photograph and get a full historical analysis, typically in 15–60 seconds.
        </p>
        <Link
          href="/login?callbackUrl=/analyse"
          className="inline-block px-8 py-3 rounded-xl font-bold text-sm transition-opacity hover:opacity-80"
          style={{ backgroundColor: "var(--brand-gold)", color: "#0d1520" }}
        >
          Analyse a coin →
        </Link>
      </section>
    </div>
  );
}
