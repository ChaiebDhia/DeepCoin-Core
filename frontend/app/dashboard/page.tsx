"use client";

/**
 * app/dashboard/page.tsx — Personal User Dashboard
 * ==================================================
 * WHAT:
 *   A personal overview page showing the authenticated user's own stats,
 *   activity history, and quick-action shortcuts.
 *
 * WHY a separate dashboard (vs just /history):
 *   /history is a raw paginated log. The dashboard gives a higher-level
 *   "at a glance" view: total analyses, routing breakdown, confidence trend,
 *   most-classified coin, and recent activity. It is the landing screen after
 *   login for analysts who want to understand their usage at a glance.
 *
 * WHO:
 *   Any authenticated user. You see only your own data.
 *   If unauthenticated, redirected to /login?callbackUrl=/dashboard.
 *
 * DATA:
 *   GET /auth/me           → user profile (display_name, role, created_at)
 *   GET /auth/me/stats     → personal stats (total, by_route, avg_conf, top_label, recent)
 */

import { useEffect }              from "react";
import { useSession }             from "next-auth/react";
import { useQuery }               from "@tanstack/react-query";
import { useRouter }              from "next/navigation";
import Link                       from "next/link";
import {
  User, ShieldCheck, Coins, History, MessageSquare,
  BarChart3, TrendingUp, FileText, Clock,
  Compass, ArrowRight, BookOpen,
} from "lucide-react";
import { motion }                  from "framer-motion";
import { getUserStats }            from "@/lib/api";
import type { UserStatsResponse }  from "@/types/api";

// ── Helpers ───────────────────────────────────────────────────────────────────

const ROUTE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  historian:    { bg: "rgba(59,130,246,0.12)",  text: "#3b82f6", border: "#3b82f630" },
  validator:    { bg: "rgba(245,158,11,0.12)",  text: "#f59e0b", border: "#f59e0b30" },
  investigator: { bg: "rgba(139,92,246,0.12)",  text: "#8b5cf6", border: "#8b5cf630" },
  unknown:      { bg: "rgba(107,114,128,0.12)", text: "#9ca3af", border: "#9ca3af30" },
};

const ROLE_META: Record<string, { label: string; color: string; icon: typeof User }> = {
  admin:   { label: "Admin",    color: "#fca5a5",          icon: ShieldCheck },
  curator: { label: "Curator",  color: "var(--brand-gold)", icon: ShieldCheck },
  analyst: { label: "Analyst",  color: "var(--text-muted)", icon: User        },
};

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function routeColor(route: string) {
  return ROUTE_COLORS[route] ?? ROUTE_COLORS.unknown;
}

// ── Sub-components ────────────────────────────────────────────────────────────

/**
 * StatCard — a single KPI tile.
 * WHY motion.div: entrance animation adds polish without extra library; the
 * cards fan in from below using a staggered delay (passed as custom prop).
 */
function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
  delay = 0,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  sub?: string;
  color: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay }}
      className="rounded-xl border p-5 flex items-center gap-4"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
    >
      <div
        className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
        style={{ backgroundColor: `${color}22` }}
      >
        <Icon size={18} style={{ color }} />
      </div>
      <div>
        <p className="text-2xl font-black tabular-nums leading-none" style={{ color }}>
          {value}
        </p>
        <p className="text-[11px] font-semibold mt-0.5" style={{ color: "var(--text-primary)" }}>
          {label}
        </p>
        {sub && (
          <p className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>
            {sub}
          </p>
        )}
      </div>
    </motion.div>
  );
}

// ── Page component ────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  // Redirect unauthenticated visitors to login
  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login?callbackUrl=/dashboard");
    }
  }, [status, router]);

  const { data: stats, isLoading: statsLoading } = useQuery<UserStatsResponse>({
    queryKey: ["user", "stats"],
    queryFn:  getUserStats,
    enabled:  status === "authenticated",
    // WHY 60 s staleTime: personal stats update slowly between page visits;
    // avoid an unnecessary fetch on every focus.
    staleTime: 60_000,
    retry:     1,
  });

  if (status === "loading" || status === "unauthenticated") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 rounded-full animate-pulse" style={{ background: "var(--surface-2)" }} />
      </div>
    );
  }

  const user        = session!.user;
  const displayName = (user as { display_name?: string }).display_name || user.name || user.email || "User";
  const role        = (user as { role?: string }).role ?? "analyst";
  const roleMeta    = ROLE_META[role] ?? ROLE_META.analyst;
  const RoleIcon    = roleMeta.icon;

  const totalAnalyses = stats?.total_analyses ?? 0;
  const avgConf       = stats ? Math.round(stats.avg_conf * 100) : null;
  const topLabel      = stats?.top_label;
  const byRoute       = stats?.by_route;

  return (
    <div
      className="min-h-screen py-10"
      style={{ backgroundColor: "var(--background)" }}
    >
      <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-8">

        {/* ── Page header ─────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex items-start justify-between"
        >
          <div>
            <h1 className="text-2xl font-black" style={{ color: "var(--text-primary)" }}>
              Welcome back, {displayName.split(" ")[0]}
            </h1>
            <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
              Here&rsquo;s your personal numismatic analytics dashboard
            </p>
          </div>
          {/* Role badge */}
          <div
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold"
            style={{ backgroundColor: `${roleMeta.color}20`, color: roleMeta.color }}
          >
            <RoleIcon size={14} />
            {roleMeta.label}
          </div>
        </motion.div>

        {/* ── KPI tiles ───────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            icon={FileText}
            label="Total Analyses"
            value={statsLoading ? "—" : totalAnalyses.toLocaleString()}
            sub="all time"
            color="#3b82f6"
            delay={0}
          />
          <StatCard
            icon={TrendingUp}
            label="Avg Confidence"
            value={statsLoading || avgConf === null ? "—" : `${avgConf}%`}
            sub="across all submissions"
            color="#10b981"
            delay={0.05}
          />
          <StatCard
            icon={Compass}
            label="Top Coin"
            value={statsLoading ? "—" : topLabel ? topLabel.label : "—"}
            sub={topLabel ? `${topLabel.count}× classified` : "no data yet"}
            color="#d4a853"
            delay={0.1}
          />
          <StatCard
            icon={Clock}
            label="Member Since"
            value={formatDate((user as { created_at?: string }).created_at)}
            sub={`${user.email}`}
            color="#8b5cf6"
            delay={0.15}
          />
        </div>

        {/* ── Route breakdown + Recent activity ─────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          {/* Route distribution */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.2 }}
            className="rounded-xl border p-5"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
          >
            <div className="flex items-center gap-2 mb-5">
              <BarChart3 size={15} style={{ color: "var(--brand-gold)" }} />
              <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
                Route Breakdown
              </span>
              <span
                className="ml-auto text-xs tabular-nums"
                style={{ color: "var(--text-muted)" }}
              >
                {totalAnalyses.toLocaleString()} total
              </span>
            </div>

            {byRoute ? (
              <div className="space-y-3">
                {(["historian", "validator", "investigator"] as const).map((key) => {
                  const count  = byRoute[key] ?? 0;
                  const pct    = totalAnalyses > 0 ? Math.round((count / totalAnalyses) * 100) : 0;
                  const rc     = routeColor(key);
                  return (
                    <div key={key}>
                      <div
                        className="flex justify-between text-[11px] mb-1.5"
                        style={{ color: "var(--text-muted)" }}
                      >
                        <span style={{ color: rc.text }} className="font-semibold capitalize">
                          {key}
                        </span>
                        <span className="tabular-nums">
                          {count.toLocaleString()} ({pct}%)
                        </span>
                      </div>
                      <div
                        className="h-1.5 rounded-full overflow-hidden"
                        style={{ backgroundColor: "var(--surface-2)" }}
                      >
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${pct}%` }}
                          transition={{ duration: 0.7, delay: 0.25, ease: "easeOut" }}
                          className="h-full rounded-full"
                          style={{ backgroundColor: rc.text }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="space-y-3">
                {[1,2,3].map(i => (
                  <div key={i} className="h-4 rounded animate-pulse"
                       style={{ background: "var(--surface-2)", width: `${60+i*10}%` }} />
                ))}
              </div>
            )}
          </motion.div>

          {/* Recent analyses */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.25 }}
            className="rounded-xl border overflow-hidden"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
          >
            <div
              className="flex items-center gap-2 px-5 py-3.5 border-b"
              style={{ borderColor: "var(--border)" }}
            >
              <History size={14} style={{ color: "var(--brand-gold)" }} />
              <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>
                Recent Analyses
              </span>
              <Link
                href="/history"
                className="ml-auto text-xs hover:underline flex items-center gap-1"
                style={{ color: "var(--text-muted)" }}
              >
                All history <ArrowRight size={11} />
              </Link>
            </div>

            {stats?.recent && stats.recent.length > 0 ? (
              <div>
                {stats.recent.map((item) => {
                  const rc = routeColor(item.route_taken);
                  return (
                    <Link
                      key={item.id}
                      href={`/history/${item.id}`}
                      className="flex items-center justify-between px-5 py-3 border-b last:border-0 hover:bg-[var(--surface-2)] transition-colors text-xs"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span
                          className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold shrink-0"
                          style={{ backgroundColor: rc.bg, color: rc.text }}
                        >
                          {item.route_taken}
                        </span>
                        <span
                          className="font-mono truncate"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {item.label}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0 ml-2">
                        <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>
                          {item.confidence !== null
                            ? `${Math.round(item.confidence * 100)}%`
                            : "—"}
                        </span>
                        <span style={{ color: "var(--text-muted)" }}>
                          {formatTime(item.timestamp)}
                        </span>
                      </div>
                    </Link>
                  );
                })}
              </div>
            ) : statsLoading ? (
              <div className="space-y-0">
                {[1,2,3].map(i => (
                  <div key={i} className="px-5 py-3 flex justify-between border-b last:border-0"
                       style={{ borderColor: "var(--border)" }}>
                    <div className="h-3 rounded animate-pulse w-24" style={{ background: "var(--surface-2)" }} />
                    <div className="h-3 rounded animate-pulse w-12" style={{ background: "var(--surface-2)" }} />
                  </div>
                ))}
              </div>
            ) : (
              <p className="px-5 py-8 text-xs text-center" style={{ color: "var(--text-muted)" }}>
                No analyses yet. <Link href="/analyse" className="underline" style={{ color: "var(--brand-gold)" }}>Analyse a coin</Link>
              </p>
            )}
          </motion.div>
        </div>

        {/* ── Quick actions ────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: 0.3 }}
        >
          <h2 className="text-sm font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            Quick Actions
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              {
                icon: Coins,
                label: "Analyse a Coin",
                desc:  "Upload a new photograph",
                href:  "/analyse",
                color: "var(--brand-gold)",
              },
              {
                icon: History,
                label: "My History",
                desc:  "Browse past analyses",
                href:  "/history",
                color: "#3b82f6",
              },
              {
                icon: MessageSquare,
                label: "AI Chat",
                desc:  "Ask numismatic questions",
                href:  "/chat",
                color: "#8b5cf6",
              },
              {
                icon: BookOpen,
                label: "Explore",
                desc:  "Community coin gallery",
                href:  "/explore",
                color: "#10b981",
              },
            ].map(({ icon: Icon, label, desc, href, color }) => (
              <Link
                key={label}
                href={href}
                className="flex flex-col gap-2 rounded-xl border p-4 hover:ring-1 hover:ring-white/10 transition-all group"
                style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `${color}20` }}
                >
                  <Icon size={16} style={{ color }} />
                </div>
                <div>
                  <p className="text-xs font-bold group-hover:underline" style={{ color: "var(--text-primary)" }}>
                    {label}
                  </p>
                  <p className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>
                    {desc}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        </motion.div>

        {/* ── Admin shortcut (only for admin/curator) ──────────────────── */}
        {(role === "admin" || role === "curator") && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.35 }}
          >
            <Link
              href="/admin"
              className="flex items-center gap-3 rounded-xl border px-5 py-4 hover:ring-1 hover:ring-white/10 transition-all"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
            >
              <ShieldCheck size={18} style={{ color: role === "admin" ? "#fca5a5" : "var(--brand-gold)" }} />
              <div>
                <p className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                  Admin Dashboard
                </p>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                  Manage users, view all analyses, and monitor the live feed
                </p>
              </div>
              <ArrowRight size={15} className="ml-auto" style={{ color: "var(--text-muted)" }} />
            </Link>
          </motion.div>
        )}

      </div>
    </div>
  );
}
