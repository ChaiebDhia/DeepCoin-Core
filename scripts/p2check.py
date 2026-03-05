import sys
f = r'C:\Users\Administrator\deepcoin\frontend\app\admin\page.tsx'
c = open(f, encoding='utf-8-sig').read()
nl = '\n'
old2 = '      {/* Health + Stats + Route Distribution */}' + nl
kpi = (
  '      {/* KPI row ' + chr(8212) + ' live user + activity counters from /api/admin/stats */}' + nl
  '      {isPrivileged && (' + nl
  '        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">' + nl
  '          {[' + nl
  '            { icon: Users, label: "Total Users", value: stats ? stats.users_total.toLocaleString() : chr(8212), sub: "registered accounts", color: "#8b5cf6" },' + nl
  '            { icon: Calendar, label: "New Today", value: stats ? stats.users_today.toLocaleString() : chr(8212), sub: "registered today (UTC)", color: "#10b981" },' + nl
  '            { icon: TrendingUp, label: "Analyses Today", value: stats ? stats.analyses_today.toLocaleString() : chr(8212), sub: "coins analysed today (UTC)", color: "#3b82f6" },' + nl
  '          ].map(({ icon: Icon, label, value, sub, color }) => (' + nl
  '            <div key={label} className="rounded-xl border p-5 flex items-center gap-4" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>' + nl
  '              <div className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0" style={{ backgroundColor: color + "22" }}>' + nl
  '                <Icon size={18} style={{ color }} />' + nl
  '              </div>' + nl
  '              <div>' + nl
  '                <p className="text-2xl font-black tabular-nums leading-none" style={{ color }}>{value}</p>' + nl
  '                <p className="text-[11px] font-semibold mt-1" style={{ color: "var(--text-primary)" }}>{label}</p>' + nl
  '                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{sub}</p>' + nl
  '              </div>' + nl
  '            </div>' + nl
  '          ))}' + nl
  '        </div>' + nl
  '      )}' + nl
  nl
  '      {/* Health + Stats + Route Distribution */}' + nl
)
print('old2 found:', old2 in c)
