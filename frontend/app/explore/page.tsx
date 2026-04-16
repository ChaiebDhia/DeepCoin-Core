"use client";

/**
 * app/explore/page.tsx — {t("subtitle")}
 * =============================================
 * WHAT: A public browser for all 9,541 coin types in the Corpus Nummorum
 *       knowledge base.  Anyone can search "silver tetradrachm Athens" and
 *       get real scholarly records with AI chat and external CN links.
 *
 * WHY THIS REPLACES THE OLD COMMUNITY GALLERY:
 *   The old explore page showed a list of user analyses — essentially the
 *   same data as the history page but stripped of PII. It provided little
 *   value to anonymous visitors who wanted to *learn* about ancient coins.
 *
 *   The KB browser solves a real problem: numismatists, students, and museum
 *   staff need a fast way to *discover* what the platform knows before they
 *   submit a coin photo.  Exposing all 9,541 CN types here also surfaces the
 *   depth of the underlying knowledge base, which builds trust.
 *
 * DATA SOURCE:
 *   GET /api/kb/types  (no auth required)
 *   — hybrid BM25 + vector search when a query is typed
 *   — paginated identity-chunk browse when no query
 *
 * WHY "use client":
 *   Uses useQuery (TanStack Query) + useState for search/pagination state.
 *   Server Component would require route params or server actions for that.
 */

import { useState, useRef }               from "react";
import { useTranslations } from "next-intl";
import { useQuery }                        from "@tanstack/react-query";
import Link                                from "next/link";
import { Search, ExternalLink, Sparkles, BookOpen, ChevronLeft, ChevronRight, X } from "lucide-react";
import { browseKb }                        from "@/lib/api";
import { KbTypeItem }                      from "@/types/api";

// ── Constants ─────────────────────────────────────────────────────────────────

const PAGE_SIZE = 20;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Colour classes for the material pill. */
function materialPillClass(material: string): string {
  const m = material.toLowerCase();
    if (m.includes("gold") || m.includes("aurum"))       return "bg-yellow-500/10 text-yellow-700 border-yellow-500/20 dark:text-yellow-400";
    if (m.includes("silver") || m.includes("argentum"))  return "bg-slate-200 text-slate-950 font-bold border-slate-400 dark:bg-slate-500/10 dark:text-slate-300 dark:border-slate-500/20";
    if (m.includes("bronze") || m.includes("aes") 
      || m.includes("copper"))                            return "bg-orange-600/10 text-orange-800 border-orange-600/20 dark:text-orange-400";
    if (m.includes("electrum"))                           return "bg-amber-600/10 text-amber-800 border-amber-600/20 dark:text-amber-400";
    return "bg-[var(--surface-2)] text-[var(--text-secondary)] border-[var(--border)]";
}
  /** Build the "{t("ask_ai")}" chat URL with the CN type pre-loaded as context. */
function chatUrl(item: KbTypeItem): string {
  const label = item.denomination
    ? `${item.denomination}${item.authority ? " of " + item.authority : ""} (CN ${item.type_id})`
    : `CN ${item.type_id}`;
  return `/chat?q=${encodeURIComponent("Tell me about " + label + " — " + item.date_range)}`;
}

// ── KbTypeCard ────────────────────────────────────────────────────────────────

function KbTypeCard({ item }: { item: KbTypeItem }) {
  const t = useTranslations("ExplorePage");
  const mat = item.material || "—";
  return (
    <div
      className="group relative rounded-xl border transition-all duration-200
                 hover:border-[var(--accent-primary)]/40 hover:shadow-lg hover:shadow-[var(--accent-primary)]/5"
      style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}
    >
      {/* CNN badge */}
      {item.in_training_set && (
        <span
          className="absolute top-2.5 right-2.5 rounded-full px-2 py-0.5 text-[10px] font-semibold
                     tracking-wide uppercase"
          style={{ background: "rgba(16,185,129,0.12)", color: "#34d399",
                   border: "1px solid rgba(16,185,129,0.25)" }}
        >
          CNN trained
        </span>
      )}

      <div className="p-4 pb-3">
        {/* Type number */}
        <p className="font-mono text-xs font-bold tracking-widest mb-1"
           style={{ color: "var(--brand-mid)" }}>
          CN {item.type_id}
        </p>

        {/* Denomination + authority */}
        <h3 className="text-sm font-semibold leading-snug mb-0.5"
            style={{ color: "var(--text-primary)" }}>
          {item.denomination || "Unknown denomination"}
          {item.authority && (
            <span className="font-normal" style={{ color: "var(--text-secondary)" }}>
              {" · "}{item.authority}
            </span>
          )}
        </h3>

        {/* Region + date */}
        <p className="text-xs leading-relaxed mb-2.5"
           style={{ color: "var(--text-secondary)" }}>
          {[item.region, item.date_range, item.mint].filter(Boolean).join(" · ") || "—"}
        </p>

        {/* Material pill */}
        {mat !== "—" && (
          <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium border mb-3 ${materialPillClass(mat)}`}>
            {mat}
          </span>
        )}

        {/* Text snippet */}
        {item.text_snippet && (
          <p className="text-[11px] leading-relaxed line-clamp-2"
             style={{ color: "var(--text-secondary)", opacity: 0.75 }}>
            {item.text_snippet}
          </p>
        )}
      </div>

      {/* Action bar */}
      <div className="flex gap-2 px-4 pb-4 pt-1">
        <Link
          href={chatUrl(item)}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium
                     transition-all hover:brightness-110 active:scale-95"
          style={{ background: "rgba(99,102,241,0.15)", color: "#a5b4fc",
                   border: "1px solid rgba(99,102,241,0.25)" }}
        >
          <Sparkles className="w-3 h-3" />
          {t("ask_ai")}
        </Link>
        <a
          href={`https://www.corpus-nummorum.eu/types/${item.type_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium
                     transition-all hover:brightness-110 active:scale-95"
          style={{ background: "var(--surface-2)", color: "var(--text-secondary)",
                   border: "1px solid rgba(255,255,255,0.10)" }}
        >
          <ExternalLink className="w-3 h-3" />
          {t("view_record")}
        </a>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ExplorePage() {
  const t = useTranslations("ExplorePage");
  const [query, setQuery]              = useState("");
  const [debouncedQuery, setDebounced] = useState("");
  const [page, setPage]                = useState(0);
  const [cnnOnly, setCnnOnly]          = useState(false);
  const debounceRef                    = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleSearch(value: string) {
    setQuery(value);
    setPage(0);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setDebounced(value), 350);
  }

  function clearSearch() {
    setQuery("");
    setDebounced("");
    setPage(0);
  }

  const skip = page * PAGE_SIZE;

  const { data, isLoading, isError } = useQuery({
    queryKey: ["kb-types", debouncedQuery, skip, cnnOnly],
    queryFn:  () => browseKb(debouncedQuery, skip, PAGE_SIZE, cnnOnly),
    staleTime: 60_000,
    placeholderData: (prev) => prev,
  });

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;

  return (
    <main className="min-h-screen" style={{ background: "var(--surface-0)" }}>

      {/* ── Hero ────────────────────────────────────────────────────────── */}
      <section className="px-4 pt-16 pb-10 text-center">
        <div className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium mb-5"
             style={{ background: "rgba(99,102,241,0.12)", color: "#a5b4fc",
                      border: "1px solid rgba(99,102,241,0.20)" }}>
          <BookOpen className="w-3.5 h-3.5" />
          {t("title")}
        </div>

        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3"
            style={{ color: "var(--text-primary)" }}>
          {t("subtitle")}
        </h1>

        <p className="text-base max-w-xl mx-auto mb-2"
           style={{ color: "var(--text-secondary)" }}>
          
          {t("desc_1")}
        </p>

        {data && !isLoading && (
          <p className="text-xs mb-8" style={{ color: "var(--text-secondary)", opacity: 0.6 }}>
            {data.total.toLocaleString()} types{" "}
            {debouncedQuery ? `match "${debouncedQuery}"` : "in database"}
            {data.search_used ? " · hybrid keyword + semantic search" : " · browsing all types"}
          </p>
        )}
        {!data && !isLoading && <div className="mb-8" />}

        {/* ── Search bar ────────────────────────────────────────────────── */}
        <div className="relative max-w-lg mx-auto mb-6">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                  style={{ color: "var(--text-secondary)" }} />
          <input
            type="text"
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder={t("search")}
            className="w-full rounded-xl py-3 pl-10 pr-10 text-sm outline-none transition-all
                       focus:ring-2 focus:ring-[var(--accent-primary)]/40"
            style={{
              background: "var(--surface-2)",
              border:     "1px solid rgba(255,255,255,0.12)",
              color:      "var(--text-primary)",
            }}
          />
          {query && (
            <button
              onClick={clearSearch}
              className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full p-0.5
                         transition-colors hover:bg-white/10"
              style={{ color: "var(--text-secondary)" }}
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* ── Toggle ────────────────────────────────────────────────────── */}
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => { setCnnOnly(false); setPage(0); }}
            className={`rounded-full px-4 py-1.5 text-xs font-medium transition-all border
                        ${!cnnOnly
                          ? "border-[var(--accent-primary)]/50 text-[var(--accent-primary)]"
                          : "border-[var(--border)] text-[var(--text-primary)] text-[var(--text-secondary)] hover:border-[var(--brand-mid)]/40"}`}
            style={{ background: !cnnOnly ? "rgba(99,102,241,0.12)" : "rgba(255,255,255,0.03)" }}
          >
            {t("all_types")}
          </button>
          <button
            onClick={() => { setCnnOnly(true); setPage(0); }}
            className={`rounded-full px-4 py-1.5 text-xs font-medium transition-all border
                        ${cnnOnly
                          ? "border-emerald-500/50 text-emerald-400"
                          : "border-[var(--border)] text-[var(--text-primary)] text-[var(--text-secondary)] hover:border-[var(--brand-mid)]/40"}`}
            style={{ background: cnnOnly ? "rgba(16,185,129,0.10)" : "rgba(255,255,255,0.03)" }}
          >
            {t("cnn_trained")}
          </button>
        </div>
      </section>

      {/* ── Results ───────────────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-4 pb-16">

        {/* Loading skeleton */}
        {isLoading && !data && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {Array.from({ length: PAGE_SIZE }).map((_, i) => (
              <div key={i} className="rounded-xl h-52 animate-pulse"
                   style={{ background: "var(--surface-2)", border: "1px solid rgba(255,255,255,0.06)" }} />
            ))}
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="rounded-xl p-8 text-center"
               style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.15)" }}>
            <p className="text-sm" style={{ color: "#f87171" }}>
              Could not load coin types. Make sure the backend is running.
            </p>
          </div>
        )}

        {/* Empty */}
        {data && data.items.length === 0 && !isLoading && (
          <div className="text-center py-20">
            <p className="text-4xl mb-4">🪙</p>
            <p className="text-base font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
              No types match &ldquo;{debouncedQuery}&rdquo;
            </p>
            <p className="text-sm mb-5" style={{ color: "var(--text-secondary)" }}>
              Try a denomination (denarius, tetradrachm), a region (Thrace, Lydia), or a dynasty (Ptolemaic, Seleucid).
            </p>
            <button
              onClick={clearSearch}
              className="rounded-full px-4 py-2 text-sm font-medium transition-all hover:brightness-110"
              style={{ background: "rgba(99,102,241,0.15)", color: "#a5b4fc",
                       border: "1px solid rgba(99,102,241,0.25)" }}
            >
              Clear search
            </button>
          </div>
        )}

        {/* Cards + pagination */}
        {data && data.items.length > 0 && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-8">
              {data.items.map((item) => (
                <KbTypeCard key={item.type_id} item={item} />
              ))}
            </div>

            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-3">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium
                             transition-all disabled:opacity-30 hover:enabled:brightness-110"
                  style={{ background: "var(--surface-2)", color: "var(--text-secondary)",
                           border: "1px solid rgba(255,255,255,0.10)" }}
                >
                  <ChevronLeft className="w-4 h-4" />
                  {t("previous")}
                </button>
                <span className="text-sm tabular-nums" style={{ color: "var(--text-secondary)" }}>
                  {t("page")} {page + 1} / {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium
                             transition-all disabled:opacity-30 hover:enabled:brightness-110"
                  style={{ background: "var(--surface-2)", color: "var(--text-secondary)",
                           border: "1px solid rgba(255,255,255,0.10)" }}
                >
                  {t("next")}
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </>
        )}
      </section>

      {/* ── Footer CTA ───────────────────────────────────────────────────────── */}
      <section className="border-t px-4 py-10 text-center"
               style={{ borderColor: "rgba(255,255,255,0.07)" }}>
        <p className="text-sm mb-4" style={{ color: "var(--text-secondary)" }}>
          {t("have_photo")}
        </p>
        <Link
          href="/analyse"
          className="inline-flex items-center gap-2 rounded-full px-6 py-2.5 text-sm font-medium
                     transition-all hover:brightness-110 active:scale-95"
          style={{ background: "var(--brand-mid)", color: "#fff" }}
        >
          <Sparkles className="w-4 h-4" />
          {t("analyse_coin")}
        </Link>
      </section>
    </main>
  );
}




