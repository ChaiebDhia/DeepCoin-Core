"""
Patch script for frontend/app/admin/page.tsx
- Updates stats query to use refetchInterval: 30_000
- Adds KPI tiles row (Users Total, New Today, Analyses Today)
- Adds Live Activity Feed
"""
import sys

f = r'C:\Users\Administrator\deepcoin\frontend\app\admin\page.tsx'

with open(f, 'r', encoding='utf-8') as fh:
    c = fh.read()

# --- Patch 1: Update stats query -----------------------------------------------------------
old1 = (
    '    queryKey:  ["admin", "stats"],\r\n'
    '    queryFn:   getAdminStats,\r\n'
    '    enabled:   isPrivileged && authed,\r\n'
    '    // WHY 60 s staleTime: route distribution changes slowly;\r\n'
    '    // avoids a redundant refetch every time the tab is re-focused.\r\n'
    '    staleTime: 60_000,\r\n'
    '    // WHY retry 1: admin/stats returns 401 if called too early (before\r\n'
    '    // SessionSync sets _authToken). One retry is enough to recover;\r\n'
    '    // beyond that it\'s a real auth problem best surfaced to the user.\r\n'
    '    retry:     1,\r\n'
    '  });'
)

new1 = (
    '    queryKey:        ["admin", "stats"],\r\n'
    '    queryFn:         getAdminStats,\r\n'
    '    enabled:         isPrivileged && authed,\r\n'
    '    // WHY 30 s refetchInterval: live activity feed + today KPIs refresh\r\n'
    '    // regularly so admins see new analyses as they arrive.\r\n'
    '    refetchInterval: 30_000,\r\n'
    '    staleTime:       30_000,\r\n'
    '    retry:           1,\r\n'
    '  });'
)

if old1 in c:
    c = c.replace(old1, new1, 1)
    print('✓ Patch 1: stats query updated')
else:
    # Try LF-only version
    old1_lf = old1.replace('\r\n', '\n')
    if old1_lf in c:
        c = c.replace(old1_lf, new1.replace('\r\n', '\n'), 1)
        print('✓ Patch 1: stats query updated (LF)')
    else:
        print('✗ Patch 1: not found, dumping region...')
        idx = c.find('queryKey:  ["admin", "stats"]')
        print(repr(c[max(0,idx-5):idx+400]))
        sys.exit(1)

# --- Patch 2: Add KPI tiles row + live activity feed ----------------------------------------

# Insert KPI row BEFORE the existing "Health + Stats + Route Distribution" comment
kpi_block = (
    '      {/* KPI row \u2014 live user + activity counters from /api/admin/stats */}\r\n'
    '      {isPrivileged && (\r\n'
    '        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">\r\n'
    '          {[\r\n'
    '            {\r\n'
    '              icon: Users,\r\n'
    '              label: "Total Users",\r\n'
    '              value: stats ? stats.users_total.toLocaleString() : "\u2014",\r\n'
    '              sub:   "registered accounts",\r\n'
    '              color: "#8b5cf6",\r\n'
    '            },\r\n'
    '            {\r\n'
    '              icon: Calendar,\r\n'
    '              label: "New Today",\r\n'
    '              value: stats ? stats.users_today.toLocaleString() : "\u2014",\r\n'
    '              sub:   "registered today (UTC)",\r\n'
    '              color: "#10b981",\r\n'
    '            },\r\n'
    '            {\r\n'
    '              icon: TrendingUp,\r\n'
    '              label: "Analyses Today",\r\n'
    '              value: stats ? stats.analyses_today.toLocaleString() : "\u2014",\r\n'
    '              sub:   "coins analysed today (UTC)",\r\n'
    '              color: "#3b82f6",\r\n'
    '            },\r\n'
    '          ].map(({ icon: Icon, label, value, sub, color }) => (\r\n'
    '            <div\r\n'
    '              key={label}\r\n'
    '              className="rounded-xl border p-5 flex items-center gap-4"\r\n'
    '              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}\r\n'
    '            >\r\n'
    '              <div\r\n'
    '                className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"\r\n'
    '                style={{ backgroundColor: `${color}22` }}\r\n'
    '              >\r\n'
    '                <Icon size={18} style={{ color }} />\r\n'
    '              </div>\r\n'
    '              <div>\r\n'
    '                <p className="text-2xl font-black tabular-nums leading-none" style={{ color }}>\r\n'
    '                  {value}\r\n'
    '                </p>\r\n'
    '                <p className="text-[11px] font-semibold mt-1" style={{ color: "var(--text-primary)" }}>\r\n'
    '                  {label}\r\n'
    '                </p>\r\n'
    '                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{sub}</p>\r\n'
    '              </div>\r\n'
    '            </div>\r\n'
    '          ))}\r\n'
    '        </div>\r\n'
    '      )}\r\n'
    '\r\n'
    '      {/* Health + Stats + Route Distribution */}\r\n'
)

old2_marker = '      {/* Health + Stats + Route Distribution */}\r\n'
if old2_marker in c:
    c = c.replace(old2_marker, kpi_block, 1)
    print('✓ Patch 2: KPI tiles row inserted')
else:
    old2_marker_lf = old2_marker.replace('\r\n', '\n')
    if old2_marker_lf in c:
        c = c.replace(old2_marker_lf, kpi_block.replace('\r\n', '\n'), 1)
        print('✓ Patch 2: KPI tiles row inserted (LF)')
    else:
        print('✗ Patch 2: marker not found')
        sys.exit(1)

# --- Patch 3: Add Live Activity Feed before "Quick links" section -------------------------

live_feed = (
    '\r\n'
    '      {/* Live Activity Feed \u2014 last 5 analyses across all users (polls every 30 s) */}\r\n'
    '      {isPrivileged && (\r\n'
    '        <div\r\n'
    '          className="rounded-xl border"\r\n'
    '          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}\r\n'
    '        >\r\n'
    '          <div\r\n'
    '            className="flex items-center gap-2 px-5 py-3.5 border-b"\r\n'
    '            style={{ borderColor: "var(--border)" }}\r\n'
    '          >\r\n'
    '            <Wifi size={14} style={{ color: "#22c55e" }} />\r\n'
    '            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>\r\n'
    '              Live Activity Feed\r\n'
    '            </span>\r\n'
    '            {/* Pulsing LIVE badge */}\r\n'
    '            <span\r\n'
    '              className="ml-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold animate-pulse"\r\n'
    '              style={{ backgroundColor: "rgba(34,197,94,0.18)", color: "#22c55e" }}\r\n'
    '            >\r\n'
    '              LIVE\r\n'
    '            </span>\r\n'
    '            <span className="ml-auto text-[10px]" style={{ color: "var(--text-muted)" }}>\r\n'
    '              refreshes every 30 s\r\n'
    '            </span>\r\n'
    '          </div>\r\n'
    '          {stats?.recent_activity?.length ? (\r\n'
    '            <div>\r\n'
    '              {stats.recent_activity.map((item) => {\r\n'
    '                const rc = routeColor(item.route_taken);\r\n'
    '                return (\r\n'
    '                  <div\r\n'
    '                    key={item.id}\r\n'
    '                    className="flex items-center justify-between px-5 py-3 border-b last:border-0 text-xs"\r\n'
    '                    style={{ borderColor: "var(--border)" }}\r\n'
    '                  >\r\n'
    '                    <div className="flex items-center gap-2 min-w-0">\r\n'
    '                      <span\r\n'
    '                        className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold shrink-0"\r\n'
    '                        style={{ backgroundColor: rc.bg, color: rc.text }}\r\n'
    '                      >\r\n'
    '                        {item.route_taken}\r\n'
    '                      </span>\r\n'
    '                      <span className="font-mono truncate" style={{ color: "var(--text-secondary)" }}>\r\n'
    '                        {item.label}\r\n'
    '                      </span>\r\n'
    '                    </div>\r\n'
    '                    <div className="flex items-center gap-3 shrink-0 ml-2">\r\n'
    '                      <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>\r\n'
    '                        {item.confidence !== null ? `${Math.round(item.confidence * 100)}%` : "\u2014"}\r\n'
    '                      </span>\r\n'
    '                      <span style={{ color: "var(--text-muted)" }}>\r\n'
    '                        {item.user_email}\r\n'
    '                      </span>\r\n'
    '                      <span style={{ color: "var(--text-muted)" }}>\r\n'
    '                        {item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : ""}\r\n'
    '                      </span>\r\n'
    '                    </div>\r\n'
    '                  </div>\r\n'
    '                );\r\n'
    '              })}\r\n'
    '            </div>\r\n'
    '          ) : (\r\n'
    '            <p className="px-5 py-8 text-xs text-center" style={{ color: "var(--text-muted)" }}>\r\n'
    '              {stats ? "No analyses yet." : "Loading\u2026"}\r\n'
    '            </p>\r\n'
    '          )}\r\n'
    '        </div>\r\n'
    '      )}\r\n'
    '\r\n'
    '      {/* Quick links */}\r\n'
)

old3 = '      {/* Quick links */}\r\n'
if old3 in c:
    c = c.replace(old3, live_feed, 1)
    print('✓ Patch 3: Live Activity Feed inserted')
else:
    old3_lf = old3.replace('\r\n', '\n')
    if old3_lf in c:
        c = c.replace(old3_lf, live_feed.replace('\r\n', '\n'), 1)
        print('✓ Patch 3: Live Activity Feed inserted (LF)')
    else:
        print('✗ Patch 3: Quick links marker not found')
        sys.exit(1)

# --- Write back ---------------------------------------------------------------------------------
with open(f, 'w', encoding='utf-8') as fh:
    fh.write(c)
print('✓ File written successfully')
