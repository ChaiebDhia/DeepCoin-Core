"use client";

/**
 * app/admin/page.tsx  Enterprise Admin Dashboard v2
 * ====================================================
 * WHAT: Tab-based admin dashboard  scales cleanly to 1,000+ records.
 *
 * WHY tabs instead of one long scroll:
 *   The old design rendered ALL panels (analyses, corrections, subscribers)
 *   on a single scrolling page. With 1,000+ rows each, the page became slow
 *   (3 simultaneous API calls on mount), hard to navigate, and visually noisy.
 *   Tabs isolate concerns:
 *     - Overview: system health + stats + quick links (always visible)
 *     - Analyses: full paginated table for all classifications
 *     - Corrections: feedback / "mark as wrong" submissions
 *     - Subscribers: waitlist emails + CSV export
 *   Each tab fetches data ONLY when it becomes active (lazy loading).
 *   This drops initial page load from 3 API calls to 1.
 *
 * WHO CAN ACCESS:
 *   Any logged-in user sees Overview.
 *   Admin / curator role sees all tabs and privileged data tables.
 *
 * HOW the PDF download works:
 *   All pdf_url values go through pdfDownloadUrl() which bypasses the
 *   Next.js proxy and goes directly to FastAPI  required for binary files.
 *   Also defensively strips full Windows filesystem paths leaked by old records
 *   (Bug 33 fix: admin.py rsplit("/") -- Path.name).
 */

import { useState }                from "react";
import { useSession }              from "next-auth/react";
import { useQuery }                from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import Link                        from "next/link";
import { redirect }                from "next/navigation";
import {
  Activity, Database, Cpu, FileText, Github, ExternalLink,
  CheckCircle, AlertTriangle, XCircle, BarChart3,
  BookOpen, Shield, Mail, Download, UserCheck, MessageSquareWarning,
  Search, ChevronLeft, ChevronRight, LayoutDashboard, Users,
  FileBarChart2, ThumbsDown,
} from "lucide-react";

import {
  getHealth, getHistory, getAdminFeedback, getAdminAnalyses, pdfDownloadUrl,
} from "@/lib/api";
import type { HistorySummary, FeedbackItem, AdminAnalysisItem } from "@/types/api";

// -- Types -------------------------------------------------------------------

type SessionUser = {
  email?:        string;
  display_name?: string;
  role?:         string;
};

type Subscriber = {
  email:         string;
  subscribed_at: string;
  status?:       string;
};

type TabId = "overview" | "analyses" | "corrections" | "subscribers";

// -- Constants ---------------------------------------------------------------

const PAGE_SIZE    = 20;
const SUB_PER_PAGE = 25;

const ROLE_COLORS: Record<string, string> = {
  admin:   "#d4a853",
  curator: "#3b82f6",
  analyst: "#10b981",
};

// -- Helpers -----------------------------------------------------------------

function downloadCSV(data: Subscriber[]) {
  const rows = [
    ["Email", "Subscribed At", "Status"],
    ...data.map(r => [r.email, r.subscribed_at, r.status ?? "confirmed"]),
  ];
  const csv = rows.map(r => r.map(c => `"${c}"`).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a   = document.createElement("a");
  a.href = url;
  a.download = `subscribers_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function routeColor(r: string) {
  return r === "historian"    ? { bg: "#3b82f620", text: "#60a5fa" }
       : r === "validator"    ? { bg: "#f59e0b20", text: "#fbbf24" }
       :                        { bg: "#8b5cf620", text: "#a78bfa" };
}

// -- Sub-components ----------------------------------------------------------

function StatusDot({ status }: { status: string }) {
  const color =
    status === "healthy"  ? "#22c55e" :
    status === "degraded" ? "#f59e0b" : "#ef4444";
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold" style={{ color }}>
      <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: color }} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function ComponentRow({ name, status }: { name: string; status: string }) {
  const Icon  = status === "ok"      ? CheckCircle : status === "warning" ? AlertTriangle : XCircle;
  const color = status === "ok"      ? "#22c55e"   : status === "warning" ? "#f59e0b"     : "#ef4444";
  return (
    <div
      className="flex items-center justify-between py-2.5 border-b last:border-0"
      style={{ borderColor: "var(--border)" }}
    >
      <span className="text-xs capitalize" style={{ color: "var(--text-secondary)" }}>
        {name.replace(/_/g, " ")}
      </span>
      <Icon size={13} style={{ color }} />
    </div>
  );
}

function Pagination({
  page, pages, onChange,
}: {
  page: number; pages: number; onChange: (p: number) => void;
}) {
  if (pages <= 1) return null;
  return (
    <div
      className="flex items-center justify-between px-5 py-3 border-t"
      style={{ borderColor: "var(--border)" }}
    >
      <span className="text-xs" style={{ color: "var(--text-muted)" }}>
        Page {page} / {pages}
      </span>
      <div className="flex gap-2">
        <button
          disabled={page === 1}
          onClick={() => onChange(page - 1)}
          className="p-1.5 rounded-lg disabled:opacity-30 transition-opacity"
          style={{ backgroundColor: "var(--surface-2)", border: "1px solid var(--border)" }}
        >
          <ChevronLeft size={12} style={{ color: "var(--text-secondary)" }} />
        </button>
        <button
          disabled={page >= pages}
          onClick={() => onChange(page + 1)}
          className="p-1.5 rounded-lg disabled:opacity-30 transition-opacity"
          style={{ backgroundColor: "var(--surface-2)", border: "1px solid var(--border)" }}
        >
          <ChevronRight size={12} style={{ color: "var(--text-secondary)" }} />
        </button>
      </div>
    </div>
  );
}

function TableSkeleton({ cols }: { cols: number }) {
  return (
    <>
      {Array.from({ length: 5 }).map((_, i) => (
        <tr key={i}>
          {Array.from({ length: cols }).map((_, j) => (
            <td key={j} className="px-4 py-3">
              <div
                className="h-3 rounded animate-pulse"
                style={{ backgroundColor: "var(--surface-2)", width: `${50 + (j * 13) % 40}%` }}
              />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

// -- Tab: Overview -----------------------------------------------------------

function OverviewTab({
  isPrivileged,
  sessionStatus,
}: {
  isPrivileged:  boolean;
  sessionStatus: string;
}) {
  const authed = sessionStatus === "authenticated";

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey:        ["health"],
    queryFn:         getHealth,
    refetchInterval: 30_000,
    // health endpoint is public — no auth needed; always enabled
  });

  const { data: historyData } = useQuery({
    queryKey: ["history", 0, 5],
    queryFn:  () => getHistory(0, 5),
    // WHY enabled: JWT isn't in _authToken until SessionSync's useEffect runs.
    // Without this guard the query fires before the token arrives → 401.
    enabled:  authed,
  });

  return (
    <div className="space-y-6">
      {/* Health + Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

        {/* System Health */}
        <div
          className="rounded-xl border p-5"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Activity size={15} style={{ color: "#22c55e" }} />
              <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
                System Health
              </span>
            </div>
            {health        && <StatusDot status={health.status} />}
            {healthLoading && <span className="text-xs" style={{ color: "var(--text-muted)" }}>Checking</span>}
          </div>
          {health?.components
            ? Object.entries(health.components).map(([n, s]) => (
                <ComponentRow key={n} name={n} status={s as string} />
              ))
            : <p className="text-xs" style={{ color: "var(--text-muted)" }}>Loading components</p>}
        </div>

        {/* Pipeline Stats */}
        <div
          className="rounded-xl border p-5"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
        >
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 size={15} style={{ color: "var(--brand-gold)" }} />
            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
              Pipeline Statistics
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { icon: FileText,  label: "Total analyses",  value: historyData?.total?.toLocaleString() ?? "", sub: "all time",          color: "#3b82f6" },
              { icon: Cpu,       label: "CNN accuracy",    value: "80.03%",  sub: "TTA 8",                                               color: "#8b5cf6" },
              { icon: Database,  label: "RAG chunks",      value: "47,705",  sub: "9,541 CN types",                                       color: "#d4a853" },
              { icon: Activity,  label: "Max latency",     value: "< 20 s",  sub: "Ollama LLM path",                                      color: "#10b981" },
            ].map(({ icon: Icon, label, value, sub, color }) => (
              <div key={label} className="rounded-lg p-3" style={{ backgroundColor: "var(--surface-2)" }}>
                <Icon size={13} className="mb-1.5" style={{ color }} />
                <p className="text-sm font-black tabular-nums" style={{ color }}>{value}</p>
                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{label}</p>
                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{sub}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent analyses (my account) */}
      <div
        className="rounded-xl border"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
      >
        <div
          className="flex items-center gap-2 px-5 py-3.5 border-b"
          style={{ borderColor: "var(--border)" }}
        >
          <FileText size={14} style={{ color: "var(--brand-gold)" }} />
          <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
            Recent Analyses (My Account)
          </span>
          <Link href="/history" className="ml-auto text-xs hover:underline"
                style={{ color: "var(--text-muted)" }}>
            View all 
          </Link>
        </div>
        {historyData?.items?.length ? (
          <div>
            {historyData.items.map((item: HistorySummary) => {
              const rc = routeColor(item.route_taken);
              return (
                <Link
                  key={item.id}
                  href={`/history/${item.id}`}
                  className="flex items-center justify-between px-5 py-3 border-b last:border-0 hover:bg-[var(--surface-2)] transition-colors text-xs"
                  style={{ borderColor: "var(--border)" }}
                >
                  <span className="font-mono" style={{ color: "var(--text-secondary)" }}>
                    {item.label ?? item.id.slice(0, 8)}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="tabular-nums font-semibold" style={{ color: "var(--text-muted)" }}>
                      {Math.round((item.confidence ?? 0) * 100)}%
                    </span>
                    <span
                      className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold"
                      style={{ backgroundColor: rc.bg, color: rc.text }}
                    >
                      {item.route_taken}
                    </span>
                  </div>
                </Link>
              );
            })}
          </div>
        ) : (
          <p className="px-5 py-8 text-xs text-center" style={{ color: "var(--text-muted)" }}>
            No analyses yet.
          </p>
        )}
      </div>

      {/* Quick links */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { icon: Cpu,      label: "FastAPI Docs",        desc: "OpenAPI / Swagger UI",       href: "http://127.0.0.1:8000/docs", color: "#10b981", external: true  },
          { icon: Github,   label: "GitHub Repo",         desc: "ChaiebDhia/DeepCoin-Core",   href: "https://github.com/ChaiebDhia/DeepCoin-Core", color: "#e2e8f0", external: true },
          { icon: BookOpen, label: "Eng. Journal",        desc: "Full development log",       href: "https://github.com/ChaiebDhia/DeepCoin-Core/blob/main/ENGINEERING_JOURNAL.md", color: "#d4a853", external: true },
          { icon: Users,    label: "History",             desc: "All coin analyses",          href: "/history", color: "#8b5cf6", external: false },
        ].map(({ icon: Icon, label, desc, href, color, external }) =>
          external ? (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col gap-2 rounded-xl border p-4 hover:ring-1 hover:ring-white/10 transition-all"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
            >
              <Icon size={16} style={{ color }} />
              <div>
                <div className="text-xs font-bold flex items-center gap-1"
                     style={{ color: "var(--text-primary)" }}>
                  {label} <ExternalLink size={9} style={{ color: "var(--text-muted)" }} />
                </div>
                <div className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>{desc}</div>
              </div>
            </a>
          ) : (
            <Link
              key={label}
              href={href}
              className="flex flex-col gap-2 rounded-xl border p-4 hover:ring-1 hover:ring-white/10 transition-all"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
            >
              <Icon size={16} style={{ color }} />
              <div>
                <div className="text-xs font-bold" style={{ color: "var(--text-primary)" }}>{label}</div>
                <div className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>{desc}</div>
              </div>
            </Link>
          )
        )}
      </div>
    </div>
  );
}

// -- Tab: All Analyses -------------------------------------------------------

function AnalysesTab({ sessionStatus }: { sessionStatus: string }) {
  const authed = sessionStatus === "authenticated";
  const [page,   setPage]   = useState(1);
  const [route,  setRoute]  = useState("");
  const [search, setSearch] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["admin", "analyses", page, route, search],
    queryFn:  () => getAdminAnalyses(
      (page - 1) * PAGE_SIZE,
      PAGE_SIZE,
      route  || undefined,
      search || undefined,
    ),
    staleTime: 30_000,
    enabled:   authed,
  });

  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
    >
      {/* Toolbar */}
      <div
        className="flex flex-wrap items-center gap-3 px-5 py-3.5 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <BarChart3 size={14} style={{ color: "#3b82f6" }} />
        <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
          All Analyses
        </span>
        {data && (
          <span
            className="text-[10px] font-black px-2 py-0.5 rounded-full tabular-nums"
            style={{ backgroundColor: "#3b82f620", color: "#3b82f6" }}
          >
            {data.total.toLocaleString()}
          </span>
        )}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search size={11}
                    className="absolute left-2 top-1/2 -translate-y-1/2 pointer-events-none"
                    style={{ color: "var(--text-muted)" }} />
            <input
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(1); }}
              placeholder="Search CN label"
              className="pl-6 pr-3 py-1.5 text-xs rounded-lg outline-none"
              style={{
                backgroundColor: "var(--surface-2)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                width: 160,
              }}
            />
          </div>
          <select
            value={route}
            onChange={e => { setRoute(e.target.value); setPage(1); }}
            className="text-xs px-2 py-1.5 rounded-lg outline-none"
            style={{
              backgroundColor: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
            }}
          >
            <option value="">All routes</option>
            <option value="historian">Historian</option>
            <option value="validator">Validator</option>
            <option value="investigator">Investigator</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Date", "CN Label", "Conf", "Route", "User", "PDF"].map(h => (
                <th
                  key={h}
                  className="px-4 py-3 text-left font-semibold whitespace-nowrap"
                  style={{ color: "var(--text-muted)" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <TableSkeleton cols={6} />
            ) : data?.items?.length ? (
              (data.items as AdminAnalysisItem[]).map(row => {
                const rc = routeColor(row.route_taken);
                return (
                  <tr
                    key={row.id}
                    className="border-b last:border-0 hover:bg-[var(--surface-2)] transition-colors"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="px-4 py-2.5 tabular-nums whitespace-nowrap"
                        style={{ color: "var(--text-muted)" }}>
                      {row.created_at
                        ? new Date(row.created_at).toLocaleDateString(undefined, {
                            month: "short", day: "numeric", year: "2-digit",
                          })
                        : ""}
                    </td>
                    <td className="px-4 py-2.5 font-mono">
                      <Link
                        href={`/history/${row.id}`}
                        className="hover:underline transition-colors"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {row.label ?? ""}
                      </Link>
                    </td>
                    <td
                      className="px-4 py-2.5 tabular-nums font-bold"
                      style={{
                        color: row.confidence >= 0.7 ? "#22c55e"
                             : row.confidence >= 0.4 ? "#f59e0b"
                             : "#a78bfa",
                      }}
                    >
                      {Math.round(row.confidence * 100)}%
                    </td>
                    <td className="px-4 py-2.5">
                      <span
                        className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold"
                        style={{ backgroundColor: rc.bg, color: rc.text }}
                      >
                        {row.route_taken}
                      </span>
                    </td>
                    <td
                      className="px-4 py-2.5 max-w-[130px] truncate"
                      style={{ color: "var(--text-muted)" }}
                      title={row.user_email}
                    >
                      {row.user_email}
                    </td>
                    <td className="px-4 py-2.5">
                      {row.pdf_url ? (
                        /* WHY pdfDownloadUrl: bypasses Next.js proxy for binary
                           files AND defensively strips leaked Windows FS paths.
                           Bug 33: admin.py used rsplit("/") instead of Path.name. */
                        <a
                          href={pdfDownloadUrl(row.pdf_url)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-[10px] font-medium hover:underline"
                          style={{ color: "#3b82f6" }}
                        >
                          <FileText size={10} /> PDF
                        </a>
                      ) : (
                        <span style={{ color: "var(--surface-3)" }}></span>
                      )}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center"
                    style={{ color: "var(--text-muted)" }}>
                  No analyses match your filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {data && <Pagination page={page} pages={data.pages} onChange={setPage} />}
    </div>
  );
}

// -- Tab: Corrections --------------------------------------------------------

function CorrectionsTab({ sessionStatus }: { sessionStatus: string }) {
  const authed = sessionStatus === "authenticated";
  const [page, setPage] = useState(1);

  const { data, isLoading } = useQuery({
    queryKey: ["admin", "feedback", page],
    queryFn:  () => getAdminFeedback((page - 1) * PAGE_SIZE, PAGE_SIZE),
    staleTime: 30_000,
    enabled:   authed,
  });

  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
    >
      {/* Header */}
      <div
        className="flex flex-wrap items-center gap-2 px-5 py-3.5 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <ThumbsDown size={14} style={{ color: "#ef4444" }} />
        <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
          User Corrections
        </span>
        {data && (
          <span
            className="ml-1 text-[10px] font-black px-2 py-0.5 rounded-full tabular-nums"
            style={{ backgroundColor: "#ef444420", color: "#ef4444" }}
          >
            {data.total.toLocaleString()}
          </span>
        )}
        <span className="ml-auto text-[10px]" style={{ color: "var(--text-muted)" }}>
          "Mark as wrong" reports  active learning signal
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Date", "Coin (CNN)", "Conf", "Route", "Suggested CN", "Note", "By"].map(h => (
                <th
                  key={h}
                  className="px-4 py-3 text-left font-semibold"
                  style={{ color: "var(--text-muted)" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <TableSkeleton cols={7} />
            ) : data?.items?.length ? (
              (data.items as FeedbackItem[]).map(fb => {
                const rc = routeColor(fb.route_taken ?? "");
                return (
                  <tr
                    key={fb.id}
                    className="border-b last:border-0 hover:bg-[var(--surface-2)] transition-colors"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="px-4 py-2.5 tabular-nums whitespace-nowrap"
                        style={{ color: "var(--text-muted)" }}>
                      {fb.created_at
                        ? new Date(fb.created_at).toLocaleDateString(undefined, {
                            month: "short", day: "numeric",
                          })
                        : ""}
                    </td>
                    <td className="px-4 py-2.5 font-mono">
                      <Link
                        href={`/history/${fb.classification_id}`}
                        className="hover:underline"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {fb.coin_label ?? ""}
                      </Link>
                    </td>
                    <td className="px-4 py-2.5 tabular-nums"
                        style={{ color: "var(--text-muted)" }}>
                      {fb.confidence != null ? `${Math.round(fb.confidence * 100)}%` : ""}
                    </td>
                    <td className="px-4 py-2.5">
                      {fb.route_taken ? (
                        <span
                          className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold"
                          style={{ backgroundColor: rc.bg, color: rc.text }}
                        >
                          {fb.route_taken}
                        </span>
                      ) : ""}
                    </td>
                    <td className="px-4 py-2.5">
                      {fb.correct_type_id ? (
                        <a
                          href={`https://www.corpus-nummorum.eu/types/${fb.correct_type_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 hover:underline"
                          style={{ color: "#3b82f6" }}
                        >
                          CN {fb.correct_type_id} <ExternalLink size={9} />
                        </a>
                      ) : ""}
                    </td>
                    <td
                      className="px-4 py-2.5 max-w-[160px] truncate"
                      style={{ color: "var(--text-muted)" }}
                      title={fb.note ?? ""}
                    >
                      {fb.note || ""}
                    </td>
                    <td className="px-4 py-2.5" style={{ color: "var(--text-muted)" }}>
                      {fb.submitted_by}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={7} className="px-4 py-10 text-center">
                  <MessageSquareWarning
                    size={20}
                    className="mx-auto mb-2"
                    style={{ color: "var(--text-muted)" }}
                  />
                  <p style={{ color: "var(--text-muted)" }}>No corrections submitted yet.</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {data && <Pagination page={page} pages={data.pages} onChange={setPage} />}
    </div>
  );
}

// -- Tab: Subscribers --------------------------------------------------------

function SubscribersTab({ sessionStatus }: { sessionStatus: string }) {
  const authed = sessionStatus === "authenticated";
  const [subPage, setSubPage] = useState(1);

  const { data: subscribers = [], isLoading } = useQuery<Subscriber[]>({
    queryKey: ["admin", "subscribers"],
    queryFn:  () => fetch("/api/admin/subscribers").then(r => r.ok ? r.json() : []),
    staleTime: 60_000,
    enabled:   authed,
  });

  const totalPages = Math.max(1, Math.ceil(subscribers.length / SUB_PER_PAGE));
  const pageItems  = subscribers.slice(
    (subPage - 1) * SUB_PER_PAGE,
    subPage * SUB_PER_PAGE,
  );

  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-5 py-3.5 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <Mail size={14} style={{ color: "#3b82f6" }} />
        <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
          Subscribers
        </span>
        <span
          className="ml-1 text-[10px] font-black px-2 py-0.5 rounded-full tabular-nums"
          style={{ backgroundColor: "#3b82f620", color: "#3b82f6" }}
        >
          {subscribers.length.toLocaleString()}
        </span>
        {subscribers.length > 0 && (
          <button
            onClick={() => downloadCSV(subscribers)}
            className="ml-auto inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors hover:bg-[var(--surface-2)]"
            style={{ color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            <Download size={12} /> Export CSV
          </button>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Email", "Status", "Subscribed"].map(h => (
                <th
                  key={h}
                  className="px-5 py-3 text-left font-semibold"
                  style={{ color: "var(--text-muted)" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <TableSkeleton cols={3} />
            ) : pageItems.length ? (
              pageItems.map(s => (
                <tr
                  key={s.email}
                  className="border-b last:border-0 hover:bg-[var(--surface-2)] transition-colors"
                  style={{ borderColor: "var(--border)" }}
                >
                  <td className="px-5 py-3 font-mono" style={{ color: "var(--text-secondary)" }}>
                    {s.email}
                  </td>
                  <td className="px-5 py-3">
                    <span
                      className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
                      style={{
                        backgroundColor: (s.status ?? "confirmed") === "confirmed" ? "#22c55e20" : "#f59e0b20",
                        color:           (s.status ?? "confirmed") === "confirmed" ? "#4ade80"   : "#fbbf24",
                      }}
                    >
                      {s.status ?? "confirmed"}
                    </span>
                  </td>
                  <td className="px-5 py-3 tabular-nums" style={{ color: "var(--text-muted)" }}>
                    {new Date(s.subscribed_at).toLocaleDateString(undefined, {
                      year: "numeric", month: "short", day: "numeric",
                    })}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="px-5 py-10 text-center">
                  <UserCheck
                    size={20}
                    className="mx-auto mb-2"
                    style={{ color: "var(--text-muted)" }}
                  />
                  <p style={{ color: "var(--text-muted)" }}>No subscribers yet.</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {/* Client-side pagination  API returns all subscribers at once */}
      <Pagination page={subPage} pages={totalPages} onChange={setSubPage} />
    </div>
  );
}

// -- Main component ----------------------------------------------------------

const TABS: {
  id:         TabId;
  label:      string;
  icon:       React.ElementType;
  privileged: boolean;
}[] = [
  { id: "overview",    label: "Overview",    icon: LayoutDashboard,      privileged: false },
  { id: "analyses",    label: "Analyses",    icon: FileBarChart2,        privileged: true  },
  { id: "corrections", label: "Corrections", icon: MessageSquareWarning, privileged: true  },
  { id: "subscribers", label: "Subscribers", icon: Mail,                 privileged: true  },
];

export default function AdminPage() {
  const { data: session, status: sessionStatus } = useSession();
  const user         = session?.user as SessionUser | undefined;
  const isPrivileged = user?.role === "admin" || user?.role === "curator";

  const [activeTab, setActiveTab] = useState<TabId>("overview");

  // Loading spinner while NextAuth resolves
  if (sessionStatus === "loading") {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div
          className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin"
          style={{ borderColor: "var(--brand-gold)" }}
        />
      </div>
    );
  }

  // Redirect unauthenticated users
  if (sessionStatus === "unauthenticated") {
    redirect("/login?callbackUrl=/admin");
  }

  // Access-restriction page for analyst role
  if (sessionStatus === "authenticated" && !isPrivileged) {
    const userEmail = user?.email ?? "your@email.com";
    return (
      <div className="py-16 max-w-2xl mx-auto flex flex-col items-center gap-8">
        <div
          className="w-16 h-16 rounded-2xl flex items-center justify-center"
          style={{ background: "rgba(239,68,68,0.10)", border: "1px solid rgba(239,68,68,0.30)" }}
        >
          <Shield size={28} style={{ color: "#f87171" }} />
        </div>
        <div className="text-center">
          <h2 className="text-2xl font-black" style={{ color: "var(--text-primary)" }}>
            Access Restricted
          </h2>
          <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
            The Admin Dashboard requires the <strong>admin</strong> or{" "}
            <strong>curator</strong> role.
          </p>
          <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
            Your current role:{" "}
            <span className="font-bold" style={{ color: "#f59e0b" }}>
              {user?.role ?? "analyst"}
            </span>
          </p>
        </div>
        <div
          className="w-full rounded-xl p-6 space-y-4"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
        >
          <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            How to promote your account to admin:
          </p>
          <ol className="text-sm space-y-2 list-decimal list-inside"
              style={{ color: "var(--text-secondary)" }}>
            <li>Open a terminal on the DeepCoin server</li>
            <li>Connect to the PostgreSQL database</li>
            <li>Run the SQL command below</li>
            <li>Sign out and sign back in  your role will update</li>
          </ol>
          <div
            className="rounded-lg p-4 font-mono text-xs overflow-x-auto"
            style={{
              background: "rgba(0,0,0,0.35)",
              border: "1px solid rgba(255,255,255,0.06)",
              color: "#86efac",
            }}
          >
            {`-- Connect: psql -U postgres -d deepcoin`}<br />
            {`UPDATE users SET role='admin' WHERE email='${userEmail}';`}<br />
            {`-- Verify:`}<br />
            {`SELECT email, role FROM users WHERE email='${userEmail}';`}
          </div>
        </div>
      </div>
    );
  }

  // Non-privileged users only see Overview tab
  const visibleTabs = TABS.filter(t => !t.privileged || isPrivileged);

  return (
    <div className="py-8 max-w-5xl space-y-6">

      {/* Page header */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div>
          <h1 className="text-2xl font-black" style={{ color: "var(--text-primary)" }}>
            Admin Dashboard
          </h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-muted)" }}>
            System health  analyses  corrections  subscribers
          </p>
        </div>
        {user && (
          <div
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border self-start"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
          >
            <Shield size={13} style={{ color: ROLE_COLORS[user.role ?? ""] ?? "var(--text-muted)" }} />
            <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              {user.display_name ?? user.email}
            </span>
            {user.role && (
              <span
                className="text-[10px] font-black uppercase px-2 py-0.5 rounded-full"
                style={{
                  backgroundColor: `${ROLE_COLORS[user.role] ?? "#6b7280"}20`,
                  color:           ROLE_COLORS[user.role] ?? "#6b7280",
                }}
              >
                {user.role}
              </span>
            )}
          </div>
        )}
      </motion.div>

      {/* Tab bar */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.06 }}
        className="flex items-center gap-1 p-1 rounded-xl border overflow-x-auto"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
      >
        {visibleTabs.map(tab => {
          const Icon    = tab.icon;
          const active  = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all whitespace-nowrap"
              style={{
                backgroundColor: active ? "var(--surface-2)"    : "transparent",
                color:           active ? "var(--text-primary)"  : "var(--text-muted)",
                boxShadow:       active ? "0 1px 3px rgba(0,0,0,0.2)" : "none",
              }}
            >
              <Icon size={13} />
              {tab.label}
            </button>
          );
        })}
      </motion.div>

      {/* Tab content  AnimatePresence for smooth transitions */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.18 }}
        >
          {activeTab === "overview"    && <OverviewTab isPrivileged={isPrivileged} sessionStatus={sessionStatus} />}
          {activeTab === "analyses"    && <AnalysesTab    sessionStatus={sessionStatus} />}
          {activeTab === "corrections" && <CorrectionsTab sessionStatus={sessionStatus} />}
          {activeTab === "subscribers" && <SubscribersTab sessionStatus={sessionStatus} />}
        </motion.div>
      </AnimatePresence>

    </div>
  );
}