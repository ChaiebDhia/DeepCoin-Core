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


 *     - Corrections: feedback / &quot;Mark as wrong&quot; submissions


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





import { useState, useEffect }     from "react";


import { useSession }              from "next-auth/react";


import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";


import { motion, AnimatePresence } from "framer-motion";


import { useTranslations } from "next-intl";
import Link                        from "next/link";


import { redirect }                from "next/navigation";


import {


  Activity, Database, Cpu, FileText, Github, ExternalLink,


  CheckCircle, AlertTriangle, XCircle, BarChart3,


  BookOpen, Shield, Mail, Download, UserCheck, MessageSquareWarning,


  Search, ChevronLeft, ChevronRight, LayoutDashboard, Users, Trash2,


  FileBarChart2, ThumbsDown, UserCog, TrendingUp, Calendar, Wifi, Inbox, Coins, RefreshCw,


} from "lucide-react";

import { CoinInventoryTab } from "@/components/admin/CoinInventoryTab";





import {


  getHealth, getHistory, getAdminFeedback, getAdminAnalyses, pdfDownloadUrl,


  getAdminUsers, updateUserRole, updateUserStatus, deleteAdminUser,


  getAdminStats, getAdminContacts, markContactRead, deleteContactMessage,


  deleteCorrection, deleteSubscriber, triggerTraining, reloadModel,


} from "@/lib/api";


import type { HistorySummary, FeedbackItem, AdminAnalysisItem, AdminUserItem, AdminStatsResponse, AdminStatsActivity, ContactMessage, AdminContactsResponse } from "@/types/api";





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





type TabId = "overview" | "coins" | "analyses" | "corrections" | "subscribers" | "users" | "contacts";





// -- Constants ---------------------------------------------------------------





const PAGE_SIZE    = 15;


const SUB_PER_PAGE = 15;





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


  // Always render the pagination bar  even on a single page.


  // Hidden would leave users confused about whether data is loading or truly empty.


  const t = useTranslations("AdminDashboard");
  const total = Math.max(1, pages);


  return (


    <div


      className="flex items-center justify-between px-5 py-3 border-t"


      style={{ borderColor: "var(--border)" }}


    >


      <span className="text-xs" style={{ color: "var(--text-muted)" }}>


        Page {page} / {total}


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


          disabled={page >= total}


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





function OverviewTab(
  
  {


  isPrivileged,


  sessionStatus,


}: {


  isPrivileged:  boolean;


  sessionStatus: string;


}) {


  
  const t = useTranslations("AdminDashboard");
  const authed = sessionStatus === "authenticated";





  const { data: health, isLoading: healthLoading } = useQuery({


    queryKey:        ["health"],


    queryFn:         getHealth,


    refetchInterval: 30_000,


    // health endpoint is public  no auth needed; always enabled


  });





  const { data: historyData } = useQuery({


    queryKey: ["history", 0, 5],


    queryFn:  () => getHistory(0, 5),


    // WHY enabled: JWT isn't in _authToken until SessionSync's useEffect runs.


    // Without this guard the query fires before the token arrives ? 401.


    enabled:  authed,


  });





  const { data: stats, isError: statsError, refetch: statsRefetch } = useQuery<AdminStatsResponse>({


    queryKey:        ["admin", "stats"],


    queryFn:         getAdminStats,


    enabled:         isPrivileged && authed,


    // WHY 30 s refetchInterval: live activity feed + today KPIs refresh


    // regularly so admins see new analyses as they arrive.


    refetchInterval: 30_000,


    staleTime:       30_000,


    retry:           1,


  });





  return (


    <div className="space-y-6">


      {/* KPI row  live user + activity counters, polls every 30s */}


      {isPrivileged && (


        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">


          {[


            {


              icon: Users,


              label: t("total_users"),


              value: stats?.users_total?.toLocaleString() ?? "",


              sub:   t("registered_accounts"),


              color: "#8b5cf6",


            },


            {


              icon: Calendar,


              label: t("new_today"),


              value: stats?.users_today?.toLocaleString() ?? "",


              sub:   t("registered_today"),


              color: "#10b981",


            },


            {


              icon: TrendingUp,


              label: t("analyses_today"),


              value: stats?.analyses_today?.toLocaleString() ?? "",


              sub:   t("coins_analysed"),


              color: "#3b82f6",


            },


          ].map(({ icon: Icon, label, value, sub, color }) => (


            <div


              key={label}


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


                <p className="text-[11px] font-semibold mt-1" style={{ color: "var(--text-primary)" }}>


                  {label}


                </p>


                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{sub}</p>


              </div>


            </div>


          ))}


        </div>


      )}





      {/* Health + Stats + Route Distribution */}


      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">





        {/* System Health */}


        <div


          className="rounded-xl border p-5"


          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}


        >


          <div className="flex items-center justify-between mb-4">


            <div className="flex items-center gap-2">


              <Activity size={15} style={{ color: "#22c55e" }} />


              <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>


                {t("system_health")}


              </span>


            </div>


            {health        && <StatusDot status={health.status} />}


            {healthLoading && <span className="text-xs" style={{ color: "var(--text-muted)" }}>{t("checking")}</span>}


          </div>


          {health?.components


            ? Object.entries(health.components).map(([n, s]) => (


                <ComponentRow key={n} name={n} status={s as string} />


              ))


            : <p className="text-xs" style={{ color: "var(--text-muted)" }}>{t("loading_components")}</p>}


        </div>





        {/* Pipeline Stats */}


        <div


          className="rounded-xl border p-5"


          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}


        >


          <div className="flex items-center gap-2 mb-4">


            <BarChart3 size={15} style={{ color: "var(--brand-gold)" }} />


            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>


              {t("pipeline_statistics")}


            </span>


          </div>


          <div className="grid grid-cols-2 gap-3">


            {[


              { icon: FileText,  label: t("total_analyses_all"),  value: historyData?.total?.toLocaleString() ?? "", sub: t("all_time"),          color: "#3b82f6" },


              { icon: Cpu,       label: t("cnn_accuracy"),    value: "80.03%",  sub: t("tta_8"),                                               color: "#8b5cf6" },


              { icon: Database,  label: t("rag_chunks"),      value: "47,705",  sub: "9,541 CN types",                                       color: "#d4a853" },


              { icon: Activity,  label: t("max_latency"),     value: "< 20 s",  sub: t("ollama_path"),                                      color: "#10b981" },


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





        {/* Route Distribution  live data from GET /api/admin/stats */}


        {isPrivileged && (


          <div


            className="rounded-xl border p-5"


            style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}


          >


            <div className="flex items-center gap-2 mb-4">


              <BarChart3 size={15} style={{ color: "#8b5cf6" }} />


              <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>


                Route Distribution


              </span>


              {stats && (


                <span className="ml-auto text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>


                  {stats.total.toLocaleString()} total


                </span>


              )}


            </div>





            {stats ? (


              <div className="space-y-3">


                {(


                  [


                    { key: "historian",    label: "Historian",    color: "#3b82f6" },


                    { key: "validator",    label: "Validator",    color: "#f59e0b" },


                    { key: "investigator", label: "Investigator", color: "#8b5cf6" },


                  ] as { key: keyof AdminStatsResponse["by_route"]; label: string; color: string }[]


                ).map(({ key, label, color }) => {


                  const count = stats.by_route[key] ?? 0;


                  const pct   = stats.total > 0 ? Math.round((count / stats.total) * 100) : 0;


                  return (


                    <div key={key}>


                      <div className="flex justify-between text-[10px] mb-1"


                           style={{ color: "var(--text-muted)" }}>


                        <span style={{ color }}>{label}</span>


                        <span className="tabular-nums">{count.toLocaleString()} ({pct}%)</span>


                      </div>


                      <div className="h-1.5 rounded-full overflow-hidden"


                           style={{ backgroundColor: "var(--surface-2)" }}>


                        <div


                          className="h-full rounded-full transition-all duration-700"


                          style={{ width: `${pct}%`, backgroundColor: color }}


                        />


                      </div>


                    </div>


                  );


                })}





                {/* Summary row: avg confidence + top label */}


                <div className="pt-2 border-t grid grid-cols-2 gap-2"


                     style={{ borderColor: "var(--border)" }}>


                  <div className="rounded-lg p-2.5" style={{ backgroundColor: "var(--surface-2)" }}>


                    <p className="text-xs font-black tabular-nums" style={{ color: "#10b981" }}>


                      {(stats.avg_conf * 100).toFixed(1)}%


                    </p>


                    <p className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>


                      avg confidence


                    </p>


                  </div>


                  {stats.top_labels[0] && (


                    <div className="rounded-lg p-2.5" style={{ backgroundColor: "var(--surface-2)" }}>


                      <p className="text-xs font-black font-mono truncate"


                         style={{ color: "#d4a853" }}>


                        {stats.top_labels[0].label}


                      </p>


                      <p className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>


                        top coin ({stats.top_labels[0].count})


                      </p>


                    </div>


                  )}


                </div>


              </div>


            ) : statsError ? (


              <div className="space-y-2">


                <p className="text-xs" style={{ color: "#f87171" }}>Unable to load stats.</p>


                <button


                  onClick={() => statsRefetch()}


                  className="text-xs underline transition-opacity hover:opacity-70"


                  style={{ color: "var(--text-muted)" }}


                >


                  Retry


                </button>


              </div>


            ) : (


              <p className="text-xs" style={{ color: "var(--text-muted)" }}>Loading stats</p>


            )}


          </div>


        )}


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





      {/* Live Activity Feed  last 5 analyses across all users, polls every 30 s */}


      {isPrivileged && (


        <div


          className="rounded-xl border"


          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}


        >


          <div


            className="flex items-center gap-2 px-5 py-3.5 border-b"


            style={{ borderColor: "var(--border)" }}


          >


            <Wifi size={14} style={{ color: "#22c55e" }} />


            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>


              Live Activity Feed


            </span>


            <span


              className="ml-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold animate-pulse"


              style={{ backgroundColor: "rgba(34,197,94,0.18)", color: "#22c55e" }}


            >


              LIVE


            </span>


            <span className="ml-auto text-[10px]" style={{ color: "var(--text-muted)" }}>


              refreshes every 30s


            </span>


          </div>


          {stats?.recent_activity?.length ? (


            <div>


              {stats.recent_activity.map((item) => {


                const rc = routeColor(item.route_taken);


                return (


                  <div


                    key={item.id}


                    className="flex items-center justify-between px-5 py-3 border-b last:border-0 text-xs"


                    style={{ borderColor: "var(--border)" }}


                  >


                    <div className="flex items-center gap-2 min-w-0">


                      <span


                        className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold shrink-0"


                        style={{ backgroundColor: rc.bg, color: rc.text }}


                      >


                        {item.route_taken}


                      </span>


                      <span className="font-mono truncate" style={{ color: "var(--text-secondary)" }}>


                        {item.label}


                      </span>


                    </div>


                    <div className="flex items-center gap-3 shrink-0 ml-2">


                      <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>


                        {item.confidence !== null


                          ? `${Math.round(item.confidence * 100)}%`


                          : ""}


                      </span>


                      <span style={{ color: "var(--text-muted)" }}>{item.user_email}</span>


                      <span style={{ color: "var(--text-muted)" }}>


                        {item.timestamp


                          ? new Date(item.timestamp).toLocaleTimeString([],


                              { hour: "2-digit", minute: "2-digit" })


                          : ""}


                      </span>


                    </div>


                  </div>


                );


              })}


            </div>


          ) : (


            <p className="px-5 py-8 text-xs text-center" style={{ color: "var(--text-muted)" }}>


              {stats ? "No analyses yet." : "Loading"}


            </p>


          )}


        </div>


      )}





      {/* Quick links */}


      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">


        {[


          { icon: Cpu,      label: "FastAPI Docs",        desc: "OpenAPI / Swagger UI",       href: "http://127.0.0.1:8000/api/docs", color: "#10b981", external: true  },


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
  const t = useTranslations("AdminDashboard");

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
  const t = useTranslations("AdminDashboard");

  const authed      = sessionStatus === "authenticated";


  const queryClient = useQueryClient();


  const [page, setPage] = useState(1);





  const { data, isLoading } = useQuery({


    queryKey: ["admin", "feedback", page],


    queryFn:  () => getAdminFeedback((page - 1) * PAGE_SIZE, PAGE_SIZE),


    staleTime: 30_000,


    enabled:   authed,


  });





  const deleteMut = useMutation({


    mutationFn: (id: string) => deleteCorrection(id),


    onSuccess:  () => queryClient.invalidateQueries({ queryKey: ["admin", "feedback"] }),


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

          <div className="ml-auto flex gap-3">
            <button
              onClick={() => {
                triggerTraining()
                  .then(r => alert(r.message))
                  .catch(e => alert("Error: " + e.message));
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded bg-amber-500/10 text-amber-600 hover:bg-amber-500/20 transition-colors"
            >
              <Cpu size={14} />
              Trigger Retraining
            </button>
            <button
              onClick={() => {
                reloadModel()
                  .then(r => alert(r.message))
                  .catch(e => alert("Error: " + e.message));
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded bg-blue-500/10 text-blue-600 hover:bg-blue-500/20 transition-colors"
            >
              <RefreshCw size={14} />
              Hot-Swap Model
            </button>
          </div>



        <span className="ml-auto text-[10px]" style={{ color: "var(--text-muted)" }}>


          &quot;Mark as wrong&quot; reports  active learning signal


        </span>


      </div>





      {/* Table */}


      <div className="overflow-x-auto">


        <table className="w-full text-xs">


          <thead>


            <tr style={{ borderBottom: "1px solid var(--border)" }}>


              {["Date", "Coin (CNN)", "Conf", "Route", "Suggested CN", "Note", "By", ""].map(h => (


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


                    <td className="px-4 py-2.5">


                      <button


                        onClick={() => {


                          if (window.confirm("Delete this correction?")) {


                            deleteMut.mutate(fb.id);


                          }


                        }}


                        disabled={deleteMut.isPending}


                        title="Delete correction"


                        className="p-1 rounded-md transition-colors hover:bg-red-500/10 disabled:opacity-40"


                        style={{ color: "#ef4444" }}


                      >


                        <Trash2 size={11} />


                      </button>


                    </td>


                  </tr>


                );


              })


            ) : (


              <tr>


                <td colSpan={8} className="px-4 py-10 text-center">


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
  const t = useTranslations("AdminDashboard");

  const authed      = sessionStatus === "authenticated";


  const queryClient = useQueryClient();


  const [subPage, setSubPage] = useState(1);





  const { data: subscribers = [], isLoading } = useQuery<Subscriber[]>({


    queryKey: ["admin", "subscribers"],


    queryFn:  () => fetch("/api/admin/subscribers").then(r => r.ok ? r.json() : []),


    staleTime: 60_000,


    enabled:   authed,


  });





  const deleteMut = useMutation({


    mutationFn: (email: string) => deleteSubscriber(email),


    onSuccess:  () => queryClient.invalidateQueries({ queryKey: ["admin", "subscribers"] }),


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


              {["Email", "Status", "Subscribed", ""].map(h => (


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


                  <td className="px-5 py-3">


                    <button


                      onClick={() => {


                        if (window.confirm(`Remove ${s.email} from the waitlist?`)) {


                          deleteMut.mutate(s.email);


                        }


                      }}


                      disabled={deleteMut.isPending}


                      title="Remove subscriber"


                      className="p-1 rounded-md transition-colors hover:bg-red-500/10 disabled:opacity-40"


                      style={{ color: "#ef4444" }}


                    >


                      <Trash2 size={11} />


                    </button>


                  </td>


                </tr>


              ))


            ) : (


              <tr>


                <td colSpan={4} className="px-5 py-10 text-center">


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





// -- UsersTab ----------------------------------------------------------------





const USER_STATUS_COLORS: Record<string, string> = {


  active:    "#22c55e",


  suspended: "#ef4444",


  pending:   "#f59e0b",


};





function UsersTab({ sessionStatus }: { sessionStatus: string }) {
  const t = useTranslations("AdminDashboard");

  const authed      = sessionStatus === "authenticated";


  const queryClient = useQueryClient();


  const [page,    setPage]    = useState(1);


  const [search,  setSearch]  = useState("");


  const [dSearch, setDSearch] = useState("");


  const PAGE_LIMIT            = 15;





  // Debounce search  500 ms


  useEffect(() => {


    const t = setTimeout(() => { setDSearch(search); setPage(1); }, 500);


    return () => clearTimeout(t);


  }, [search]);





  const skip = (page - 1) * PAGE_LIMIT;





  const { data, isLoading } = useQuery({


    queryKey: ["admin", "users", skip, PAGE_LIMIT, dSearch],


    queryFn:  () => getAdminUsers(skip, PAGE_LIMIT, dSearch || undefined),


    staleTime: 30_000,


    enabled:   authed,


  });





  const items      = data?.items ?? [];


  const totalPages = data?.pages ?? 1;


  const total      = data?.total ?? 0;





  // Mutation helpers


  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["admin", "users"] });





  async function handleRoleChange(userId: string, role: string) {


    try {


      await updateUserRole(userId, role);


      invalidate();


    } catch (e: unknown) {


      alert((e as Error).message ?? "Failed to update role");


    }


  }





  async function handleStatusToggle(userId: string, currentStatus: string) {


    const newStatus = currentStatus === "suspended" ? "active" : "suspended";


    try {


      await updateUserStatus(userId, newStatus);


      invalidate();


    } catch (e: unknown) {


      alert((e as Error).message ?? "Failed to update status");


    }


  }





  async function handleDelete(userId: string, email: string) {


    if (!window.confirm(`Permanently delete user "${email}"?\nAll their analyses will be de-associated (not deleted).`)) return;


    try {


      await deleteAdminUser(userId);


      invalidate();


    } catch (e: unknown) {


      alert((e as Error).message ?? "Failed to delete user");


    }


  }





  return (


    <div


      className="rounded-xl border overflow-hidden"


      style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}


    >


      {/* Header */}


      <div


        className="flex flex-wrap items-center gap-3 px-5 py-3.5 border-b"


        style={{ borderColor: "var(--border)" }}


      >


        <UserCog size={14} style={{ color: "var(--brand-gold)" }} />


        <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>{t('users')}</span>


        <span


          className="text-[10px] font-black px-2 py-0.5 rounded-full tabular-nums"


          style={{ backgroundColor: "rgba(212,168,83,0.15)", color: "var(--brand-gold)" }}


        >


          {total.toLocaleString()}


        </span>


        <div


          className="ml-auto flex items-center gap-2 px-3 py-1.5 rounded-lg flex-1 min-w-[180px] max-w-xs"


          style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}


        >


          <Search size={12} style={{ color: "var(--text-muted)", flexShrink: 0 }} />


          <input


            value={search}


            onChange={e => setSearch(e.target.value)}


            placeholder="Search by email"


            className="bg-transparent text-xs outline-none w-full"


            style={{ color: "var(--text-primary)" }}


          />


        </div>


      </div>





      {/* Table */}


      <div className="overflow-x-auto">


        <table className="w-full text-xs">


          <thead>


            <tr style={{ borderBottom: "1px solid var(--border)" }}>


              {["Email", "Name", "Role", "Status", "Joined", "Analyses", "Actions"].map(h => (


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


              <TableSkeleton cols={7} />


            ) : items.length ? (


              items.map((u: AdminUserItem) => (


                <tr


                  key={u.id}


                  className="border-b last:border-0 hover:bg-[var(--surface-2)] transition-colors"


                  style={{ borderColor: "var(--border)" }}


                >


                  {/* Email */}


                  <td className="px-4 py-3 font-mono max-w-[200px] truncate" style={{ color: "var(--text-secondary)" }}>


                    {u.email}


                  </td>





                  {/* Display name */}


                  <td className="px-4 py-3" style={{ color: "var(--text-muted)" }}>


                    {u.display_name ?? ""}


                  </td>





                  {/* Role badge + change select */}


                  <td className="px-4 py-3">


                    <select


                      value={u.role}


                      onChange={e => handleRoleChange(u.id, e.target.value)}


                      className="text-[10px] font-bold px-2 py-0.5 rounded-full border-0 outline-none cursor-pointer"


                      style={{


                        backgroundColor: `${ROLE_COLORS[u.role] ?? "#6b7280"}20`,


                        color:           ROLE_COLORS[u.role] ?? "#6b7280",


                      }}


                    >


                      <option value="admin">admin</option>


                      <option value="curator">curator</option>


                      <option value="analyst">analyst</option>


                    </select>


                  </td>





                  {/* Status badge + toggle */}


                  <td className="px-4 py-3">


                    <button


                      onClick={() => handleStatusToggle(u.id, u.status)}


                      title={u.status === "suspended" ? "Click to reactivate" : "Click to suspend"}


                      className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full transition-opacity hover:opacity-70"


                      style={{


                        backgroundColor: `${USER_STATUS_COLORS[u.status] ?? "#6b7280"}20`,


                        color:           USER_STATUS_COLORS[u.status] ?? "#6b7280",


                      }}


                    >


                      {u.status}


                    </button>


                  </td>





                  {/* Joined */}


                  <td className="px-4 py-3 whitespace-nowrap tabular-nums" style={{ color: "var(--text-muted)" }}>


                    {u.created_at


                      ? new Date(u.created_at).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })


                      : ""}


                  </td>





                  {/* Analyses count */}


                  <td className="px-4 py-3 tabular-nums text-center" style={{ color: "var(--text-secondary)" }}>


                    {u.analyses_count}


                  </td>





                  {/* Delete */}


                  <td className="px-4 py-3">


                    <button


                      onClick={() => handleDelete(u.id, u.email)}


                      title="Delete user"


                      className="p-1.5 rounded-lg transition-colors hover:bg-red-500/10"


                      style={{ color: "#f87171" }}


                    >


                      <Trash2 size={13} />


                    </button>


                  </td>


                </tr>


              ))


            ) : (


              <tr>


                <td colSpan={7} className="px-5 py-10 text-center">


                  <Users size={20} className="mx-auto mb-2" style={{ color: "var(--text-muted)" }} />


                  <p style={{ color: "var(--text-muted)" }}>No users match your search.</p>


                </td>


              </tr>


            )}


          </tbody>


        </table>


      </div>


      <Pagination page={page} pages={totalPages} onChange={setPage} />


    </div>


  );


}





// -- ContactsTab sub-component -----------------------------------------------





function ContactsTab({ sessionStatus }: { sessionStatus: string }) {
  const t = useTranslations("AdminDashboard");

  const { data, isLoading, refetch } = useQuery<AdminContactsResponse>({


    queryKey:  ["admin", "contacts"],


    queryFn:   getAdminContacts,


    enabled:   sessionStatus === "authenticated",


    staleTime: 30_000,


  });





  const [expanded, setExpanded] = useState<string | null>(null);





  async function handleMarkRead(id: string) {


    try { await markContactRead(id); refetch(); } catch { /* ignore */ }


  }


  async function handleDelete(id: string) {


    if (!window.confirm("Delete this message permanently?")) return;


    try { await deleteContactMessage(id); refetch(); } catch { /* ignore */ }


  }





  return (


    <div className="space-y-4 mt-4">


      {/* Header row */}


      <div className="flex items-center gap-3">


        <Inbox size={16} style={{ color: "var(--brand-gold)" }} />


        <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>


          Contact Inbox


        </span>


        {data && data.unread > 0 && (


          <span


            className="px-2 py-0.5 rounded-full text-[10px] font-black"


            style={{ backgroundColor: "#ef444420", color: "#f87171" }}


          >


            {data.unread} unread


          </span>


        )}


        <span className="ml-auto text-xs" style={{ color: "var(--text-muted)" }}>


          {data ? `${data.total} message${data.total !== 1 ? "s" : ""}` : ""}


        </span>


      </div>





      {isLoading ? (


        <div className="space-y-3">


          {[1, 2, 3].map(i => (


            <div key={i} className="h-16 rounded-xl animate-pulse" style={{ background: "var(--surface-2)" }} />


          ))}


        </div>


      ) : !data || data.items.length === 0 ? (


        <div


          className="rounded-xl border p-10 text-center"


          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}


        >


          <Inbox size={28} className="mx-auto mb-2" style={{ color: "var(--text-muted)" }} />


          <p className="text-sm" style={{ color: "var(--text-muted)" }}>No contact messages yet.</p>


          <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>


            Messages sent via the /contact form will appear here.


          </p>


        </div>


      ) : (


        <div className="space-y-2">


          {data.items.map((msg: ContactMessage) => {


            const isOpen = expanded === msg.id;


            return (


              <div


                key={msg.id}


                className="rounded-xl border overflow-hidden transition-all"


                style={{


                  borderColor: msg.read ? "var(--border)" : "#d4a85350",


                  backgroundColor: msg.read ? "var(--surface-1)" : "#d4a85308",


                }}


              >


                {/* Header row */}


                <button


                  className="w-full flex items-center gap-3 px-4 py-3 text-left"


                  onClick={() => {


                    setExpanded(isOpen ? null : msg.id);


                    if (!msg.read) handleMarkRead(msg.id);


                  }}


                >


                  {/* Unread dot */}


                  <div


                    className="w-2 h-2 rounded-full shrink-0"


                    style={{ backgroundColor: msg.read ? "transparent" : "#d4a853" }}


                  />


                  <div className="flex-1 min-w-0">


                    <div className="flex items-center gap-2">


                      <span className="font-semibold text-xs" style={{ color: "var(--text-primary)" }}>


                        {msg.name}


                      </span>


                      <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>


                        {msg.email}


                      </span>


                    </div>


                    <p className="text-[11px] truncate" style={{ color: "var(--text-secondary)" }}>


                      {msg.subject}


                    </p>


                  </div>


                  <div className="flex items-center gap-2 shrink-0">


                    <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>


                      {new Date(msg.created_at).toLocaleDateString(undefined, {


                        month: "short", day: "numeric", year: "numeric",


                      })}


                    </span>


                    <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>


                      {isOpen ? "?" : "?"}


                    </span>


                  </div>


                </button>





                {/* Expanded body */}


                {isOpen && (


                  <div


                    className="px-4 pb-4 pt-1 space-y-3 border-t"


                    style={{ borderColor: "var(--border)" }}


                  >


                    <p className="text-sm whitespace-pre-wrap" style={{ color: "var(--text-primary)" }}>


                      {msg.message}


                    </p>


                    <div className="flex items-center gap-2">


                      <a


                        href={`mailto:${msg.email}?subject=Re: [DeepCoin] ${encodeURIComponent(msg.subject)}`}


                        className="text-xs px-3 py-1.5 rounded-lg font-semibold hover:opacity-80 transition-opacity"


                        style={{ backgroundColor: "var(--brand-gold)", color: "var(--surface-0)" }}


                      >


                        Reply via email


                      </a>


                      {!msg.read && (


                        <button


                          className="text-xs px-3 py-1.5 rounded-lg font-semibold"


                          style={{ backgroundColor: "var(--surface-2)", color: "var(--text-secondary)" }}


                          onClick={() => handleMarkRead(msg.id)}


                        >


                          Mark as read


                        </button>


                      )}


                      <button


                        className="text-xs px-3 py-1.5 rounded-lg font-semibold ml-auto"


                        style={{ backgroundColor: "#ef444420", color: "#f87171" }}


                        onClick={() => handleDelete(msg.id)}


                      >


                        Delete


                      </button>


                    </div>


                  </div>


                )}


              </div>


            );


          })}


        </div>


      )}


    </div>


  );


}





// -- Main component ----------------------------------------------------------





const TABS: {


  id:         TabId;


  labelKey:   string;


  icon:       React.ElementType;


  privileged: boolean;


}[] = [


  { id: "overview",    labelKey: "overview",    icon: LayoutDashboard,      privileged: false },


  { id: "coins",       labelKey: "coins",       icon: Coins,                privileged: true  },


  { id: "analyses",    labelKey: "analyses",    icon: FileBarChart2,        privileged: true  },


  { id: "corrections", labelKey: "corrections", icon: MessageSquareWarning, privileged: true  },


  { id: "subscribers", labelKey: "subscribers", icon: Mail,                 privileged: true  },


  { id: "users",       labelKey: "users",       icon: UserCog,              privileged: true  },


  { id: "contacts",    labelKey: "contacts",    icon: Inbox,                privileged: true  },


];





export default function AdminPage() {
  const tAdmin = useTranslations("AdminDashboard");

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


              border: "1px solid var(--surface-1)",


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


            {tAdmin("adminDashboard")}


          </h1>


          <p className="text-sm mt-0.5" style={{ color: "var(--text-muted)" }}>


            {tAdmin("system_health")} · {tAdmin("coins")} · {tAdmin("analyses")} · {tAdmin("corrections")} · {tAdmin("subscribers")}


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


              {tAdmin(tab.labelKey)}


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


          {activeTab === "coins"       && <CoinInventoryTab sessionStatus={sessionStatus} />}


          {activeTab === "analyses"    && <AnalysesTab    sessionStatus={sessionStatus} />}


          {activeTab === "corrections" && <CorrectionsTab sessionStatus={sessionStatus} />}


          {activeTab === "subscribers" && <SubscribersTab sessionStatus={sessionStatus} />}


          {activeTab === "users"       && <UsersTab       sessionStatus={sessionStatus} />}


          {activeTab === "contacts"    && <ContactsTab    sessionStatus={sessionStatus} />}


        </motion.div>


      </AnimatePresence>





    </div>


  );


}








