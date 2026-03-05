"""
Patch 2+3 for frontend/app/admin/page.tsx
Adds KPI tiles row and Live Activity Feed to OverviewTab.
File has UTF-8-BOM and LF line endings.
"""
import sys

f = r'C:\Users\Administrator\deepcoin\frontend\app\admin\page.tsx'
c = open(f, encoding='utf-8-sig').read()

# ── Patch 2: Insert KPI tiles row BEFORE "Health + Stats" grid ───────────────
old2 = '      {/* Health + Stats + Route Distribution */}\n'
new2 = '''\
      {/* KPI row \u2014 live user + activity counters, polls every 30\u00a0s */}
      {isPrivileged && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            {
              icon: Users,
              label: "Total Users",
              value: stats?.users_total?.toLocaleString() ?? "\u2014",
              sub:   "registered accounts",
              color: "#8b5cf6",
            },
            {
              icon: Calendar,
              label: "New Today",
              value: stats?.users_today?.toLocaleString() ?? "\u2014",
              sub:   "registered today (UTC)",
              color: "#10b981",
            },
            {
              icon: TrendingUp,
              label: "Analyses Today",
              value: stats?.analyses_today?.toLocaleString() ?? "\u2014",
              sub:   "coins analysed today (UTC)",
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
'''

if old2 in c:
    c = c.replace(old2, new2, 1)
    print('✓ Patch 2: KPI tiles row inserted')
else:
    print('✗ Patch 2: marker not found')
    sys.exit(1)

# ── Patch 3: Insert Live Activity Feed BEFORE "Quick links" section ──────────
old3 = '      {/* Quick links */}\n'
new3 = '''\
      {/* Live Activity Feed \u2014 last 5 analyses across all users, polls every 30 s */}
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
              refreshes every 30\u00a0s
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
                          : "\u2014"}
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
              {stats ? "No analyses yet." : "Loading\u2026"}
            </p>
          )}
        </div>
      )}

      {/* Quick links */}
'''

if old3 in c:
    c = c.replace(old3, new3, 1)
    print('✓ Patch 3: Live Activity Feed inserted')
else:
    print('✗ Patch 3: Quick links marker not found')
    sys.exit(1)

# ── Write back (preserve original encoding) ──────────────────────────────────
# Write without BOM since utf-8-sig adds BOM on write too
open(f, 'w', encoding='utf-8-sig').write(c)
print('✓ File saved')
