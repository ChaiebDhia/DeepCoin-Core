"use client";

/**
 * app/admin/page.tsx — Admin dashboard
 * ======================================
 * WHAT: Shows system health, pipeline statistics, recent history, and
 *       quick links to backend documentation and monitoring.
 *
 * WHO CAN ACCESS:
 *   Any logged-in user can view the admin dashboard. The admin/curator role
 *   badge is shown from their session. FastAPI enforces role policies on
 *   sensitive endpoints (e.g. /api/metrics requires an API key).
 *   Full RBAC middleware (Next.js route-level protection) is Layer 7.
 *
 * WHY "use client":
 *   Reads session (useSession), fetches health + history live (TanStack Query).
 *
 * HOW to reach this page:
 *   - Header UserMenu dropdown → "Admin" link (shows only for admin/curator)
 *   - Direct URL: /admin
 */

import { useSession }                  from "next-auth/react";
import { useQuery }                    from "@tanstack/react-query";
import { motion }                      from "framer-motion";
import Link                            from "next/link";
import { redirect }                    from "next/navigation";
import {
  Activity, Database, Cpu, FileText, Github, ExternalLink,
  Users, Clock, CheckCircle, AlertTriangle, XCircle, BarChart3,
  BookOpen, Shield, Mail, Download, UserCheck,
} from "lucide-react";
import { getHealth, getHistory }       from "@/lib/api";
import type { HistorySummary }         from "@/types/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type SessionUser = {
  email?:        string;
  display_name?: string;
  role?:         string;
  access_token?: string;
};

type Subscriber = {
  email:         string;
  subscribed_at: string;
};

// ── CSV export helper ─────────────────────────────────────────────────────────

function downloadCSV(data: Subscriber[]) {
  const rows = [["Email", "Subscribed At"], ...data.map(r => [r.email, r.subscribed_at])];
  const csv  = rows.map(r => r.map(c => `"${c}"`).join(",")).join("\n");
  const url  = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a    = document.createElement("a");
  a.href = url;
  a.download = `subscribers_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

const ROLE_COLORS: Record<string, string> = {
  admin:    "#d4a853",
  curator:  "#3b82f6",
  analyst:  "#10b981",
};

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusDot({ status }: { status: string }) {
  const color =
    status === "healthy"  ? "#22c55e" :
    status === "degraded" ? "#f59e0b" : "#ef4444";
  const label =
    status === "healthy"  ? "Healthy" :
    status === "degraded" ? "Degraded" : "Down";
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium" style={{ color }}>
      <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

function ComponentRow({ name, status }: { name: string; status: string }) {
  const Icon =
    status === "ok"      ? CheckCircle :
    status === "warning" ? AlertTriangle : XCircle;
  const color =
    status === "ok"      ? "#22c55e" :
    status === "warning" ? "#f59e0b" : "#ef4444";
  return (
    <div className="flex items-center justify-between py-2 border-b last:border-b-0" style={{ borderColor: "var(--border)" }}>
      <span className="text-xs" style={{ color: "var(--text-secondary)" }}>{name}</span>
      <Icon size={14} style={{ color }} />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function AdminPage() {
  const { data: session, status: sessionStatus } = useSession();
  const user = session?.user as SessionUser | undefined;

  // Redirect unauthenticated users to login
  if (sessionStatus === "unauthenticated") {
    redirect("/login?callbackUrl=/admin");
  }

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ["health"],
    queryFn:  getHealth,
    refetchInterval: 30_000,
  });

  const { data: historyData } = useQuery({
    queryKey: ["history", 1, 5],
    queryFn:  () => getHistory(0, 5),
  });

  const isPrivileged = (user?.role === "admin" || user?.role === "curator");

  const { data: subscribers = [] } = useQuery<Subscriber[]>({
    queryKey: ["admin", "subscribers"],
    queryFn:  () => fetch("/api/admin/subscribers").then(r => r.ok ? r.json() : []),
    enabled:  isPrivileged,
    staleTime: 60_000,
  });

  if (sessionStatus === "loading") {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="w-6 h-6 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: "var(--brand-gold)" }} />
      </div>
    );
  }

  return (
    <div className="py-8 max-w-5xl space-y-8">

      {/* Page header */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div>
          <h1 className="text-2xl font-black" style={{ color: "var(--text-primary)" }}>
            Admin Dashboard
          </h1>
          <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>
            System health, pipeline stats, and quick access links.
          </p>
        </div>
        {user && (
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl border" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
            <Shield size={14} style={{ color: ROLE_COLORS[user.role ?? ""] ?? "var(--text-muted)" }} />
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

      {/* Top row: health + stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* System Health */}
        <motion.div
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-2xl border p-6"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
        >
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <Activity size={16} style={{ color: "#22c55e" }} />
              <h2 className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>System Health</h2>
            </div>
            {health && <StatusDot status={health.status} />}
            {healthLoading && <span className="text-xs" style={{ color: "var(--text-muted)" }}>Checking…</span>}
          </div>
          {health?.components ? (
            <div>
              {Object.entries(health.components).map(([name, st]) => (
                <ComponentRow key={name} name={name} status={st as string} />
              ))}
            </div>
          ) : (
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>Loading components…</p>
          )}
        </motion.div>

        {/* Pipeline Stats */}
        <motion.div
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15 }}
          className="rounded-2xl border p-6"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
        >
          <div className="flex items-center gap-2 mb-5">
            <BarChart3 size={16} style={{ color: "var(--brand-gold)" }} />
            <h2 className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>Pipeline Statistics</h2>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {[
              { icon: FileText, label: "Total analyses",    value: historyData?.total?.toString() ?? "—",    color: "#3b82f6" },
              { icon: Cpu,      label: "CNN accuracy",      value: "80.03%",   sub: "TTA ×8",                color: "#8b5cf6" },
              { icon: Database, label: "RAG chunks",        value: "47,705",   sub: "9,541 types",           color: "#d4a853" },
              { icon: Clock,    label: "Max latency",       value: "~20 s",    sub: "Ollama LLM",            color: "#10b981" },
            ].map(({ icon: Icon, label, value, sub, color }) => (
              <div key={label} className="rounded-xl p-3" style={{ backgroundColor: "var(--surface-2)" }}>
                <Icon size={14} className="mb-2" style={{ color }} />
                <p className="text-base font-black tabular-nums" style={{ color }}>{value}</p>
                <p className="text-[10px] font-medium" style={{ color: "var(--text-muted)" }}>{label}</p>
                {sub && <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{sub}</p>}
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Recent analyses */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="rounded-2xl border"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
      >
        <div className="flex items-center gap-2 px-6 py-4 border-b" style={{ borderColor: "var(--border)" }}>
          <FileText size={15} style={{ color: "var(--brand-gold)" }} />
          <h2 className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>Recent Analyses</h2>
          <Link
            href="/history"
            className="ml-auto text-xs hover:underline"
            style={{ color: "var(--text-muted)" }}
          >
            View all →
          </Link>
        </div>
        {historyData?.items?.length ? (
          <div>
            {historyData.items.map((item: HistorySummary) => (
              <Link
                key={item.id}
                href={`/history/${item.id}`}
                className="flex items-center justify-between px-6 py-3 border-b last:border-b-0 hover:bg-[var(--surface-2)] transition-colors text-xs"
                style={{ borderColor: "var(--border)" }}
              >
                <span className="font-mono" style={{ color: "var(--text-secondary)" }}>
                  {item.label ?? item.id}
                </span>
                <span
                  className="px-2 py-0.5 rounded-full font-medium"
                  style={{
                    backgroundColor:
                      item.route_taken === "historian"   ? "#3b82f620" :
                      item.route_taken === "validator"   ? "#f59e0b20" :
                      item.route_taken === "investigator"? "#8b5cf620" : "#6b728020",
                    color:
                      item.route_taken === "historian"   ? "#3b82f6" :
                      item.route_taken === "validator"   ? "#f59e0b" :
                      item.route_taken === "investigator"? "#8b5cf6" : "#6b7280",
                  }}
                >
                  {item.route_taken ?? "unknown"}
                </span>
              </Link>
            ))}
          </div>
        ) : (
          <p className="px-6 py-4 text-xs" style={{ color: "var(--text-muted)" }}>No analyses yet.</p>
        )}
      </motion.div>

      {/* Subscriber management — admin / curator only */}
      {isPrivileged && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.23 }}
          className="rounded-2xl border"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
        >
          <div
            className="flex items-center gap-2 px-6 py-4 border-b"
            style={{ borderColor: "var(--border)" }}
          >
            <Mail size={15} style={{ color: "#3b82f6" }} />
            <h2 className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
              Subscribers
            </h2>
            <span
              className="ml-1 text-[10px] font-black px-2 py-0.5 rounded-full tabular-nums"
              style={{ backgroundColor: "#3b82f620", color: "#3b82f6" }}
            >
              {subscribers.length}
            </span>
            {subscribers.length > 0 && (
              <button
                onClick={() => downloadCSV(subscribers)}
                className="ml-auto flex items-center gap-1.5 text-xs px-3 py-1 rounded-lg hover:bg-[var(--surface-2)] transition-colors"
                style={{ color: "var(--text-secondary)", border: "1px solid var(--border)" }}
              >
                <Download size={12} />
                Export CSV
              </button>
            )}
          </div>
          {subscribers.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
                    <th className="px-6 py-3 text-left font-medium" style={{ color: "var(--text-muted)" }}>
                      Email
                    </th>
                    <th className="px-6 py-3 text-right font-medium" style={{ color: "var(--text-muted)" }}>
                      Subscribed
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {subscribers.map((s, i) => (
                    <tr
                      key={s.email}
                      className="border-b last:border-b-0 hover:bg-[var(--surface-2)] transition-colors"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <td className="px-6 py-3 font-mono" style={{ color: "var(--text-secondary)" }}>
                        {s.email}
                      </td>
                      <td className="px-6 py-3 text-right tabular-nums" style={{ color: "var(--text-muted)" }}>
                        {new Date(s.subscribed_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="px-6 py-8 text-center">
              <UserCheck size={20} className="mx-auto mb-2" style={{ color: "var(--text-muted)" }} />
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                No subscribers yet. The waitlist will appear here once users sign up.
              </p>
            </div>
          )}
        </motion.div>
      )}

      {/* Quick links */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {[
          {
            icon:  Cpu,
            label: "FastAPI Docs",
            desc:  "OpenAPI / Swagger UI — requires local access",
            href:  "http://127.0.0.1:8000/docs",
            color: "#10b981",
            external: true,
          },
          {
            icon:  Github,
            label: "GitHub Repository",
            desc:  "ChaiebDhia / DeepCoin-Core",
            href:  "https://github.com/ChaiebDhia/DeepCoin-Core",
            color: "#e2e8f0",
            external: true,
          },
          {
            icon:  BookOpen,
            label: "Engineering Journal",
            desc:  "ENGINEERING_JOURNAL.md — full development log",
            href:  "https://github.com/ChaiebDhia/DeepCoin-Core/blob/main/ENGINEERING_JOURNAL.md",
            color: "#d4a853",
            external: true,
          },
          {
            icon:  Users,
            label: "History",
            desc:  "All coin analyses with pagination and filters",
            href:  "/history",
            color: "#8b5cf6",
            external: false,
          },
        ].map(({ icon: Icon, label, desc, href, color, external }) => (
          external ? (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col gap-2 rounded-xl border p-5 hover:ring-1 transition-all"
              style={{
                borderColor:     "var(--border)",
                backgroundColor: "var(--surface-1)",
              }}
            >
              <Icon size={18} style={{ color }} />
              <div>
                <div className="text-xs font-bold flex items-center gap-1" style={{ color: "var(--text-primary)" }}>
                  {label} <ExternalLink size={10} style={{ color: "var(--text-muted)" }} />
                </div>
                <div className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>{desc}</div>
              </div>
            </a>
          ) : (
            <Link
              key={label}
              href={href}
              className="flex flex-col gap-2 rounded-xl border p-5 hover:ring-1 transition-all"
              style={{
                borderColor:     "var(--border)",
                backgroundColor: "var(--surface-1)",
              }}
            >
              <Icon size={18} style={{ color }} />
              <div>
                <div className="text-xs font-bold" style={{ color: "var(--text-primary)" }}>{label}</div>
                <div className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>{desc}</div>
              </div>
            </Link>
          )
        ))}
      </motion.div>

      {/* How to log in note */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="rounded-xl border p-5 text-xs"
        style={{ borderColor: "rgba(212,168,83,0.3)", backgroundColor: "rgba(212,168,83,0.05)", color: "var(--text-secondary)" }}
      >
        <p className="font-semibold mb-2" style={{ color: "var(--brand-gold)" }}>How to create an admin account</p>
        <ol className="list-decimal list-inside space-y-1">
          <li>Register at <Link href="/register" className="underline">/register</Link> with any email and password.</li>
          <li>
            Promote to admin via FastAPI shell:
            <code className="ml-2 px-1.5 py-0.5 rounded" style={{ backgroundColor: "var(--surface-2)", color: "var(--text-primary)" }}>
              UPDATE users SET role=&apos;admin&apos; WHERE email=&apos;your@email.com&apos;;
            </code>
          </li>
          <li>Or use the seed script if one exists in <code style={{ color: "var(--text-primary)" }}>scripts/seed_admin.py</code>.</li>
        </ol>
      </motion.div>
    </div>
  );
}
