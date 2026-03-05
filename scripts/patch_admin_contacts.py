"""
patch_admin_contacts.py
========================
Adds a Contacts tab (contact form inbox) to the admin page.
Uses utf-8-sig encoding to handle the file's UTF-8 BOM.
"""
import sys

ADMIN_PAGE = "frontend/app/admin/page.tsx"

# -- Load --
with open(ADMIN_PAGE, encoding="utf-8-sig") as f:
    content = f.read()

changes = 0

# ── 1. Add "contacts" to TabId type ──────────────────────────────────────────
OLD1 = 'type TabId = "overview" | "analyses" | "corrections" | "subscribers" | "users";'
NEW1 = 'type TabId = "overview" | "analyses" | "corrections" | "subscribers" | "users" | "contacts";'
if OLD1 in content:
    content = content.replace(OLD1, NEW1, 1)
    changes += 1
    print("✅ TabId updated")
else:
    print("⚠️  TabId not found")

# ── 2. Add Inbox to lucide-react imports ─────────────────────────────────────
OLD2 = '  TrendingUp, Calendar, Wifi,'
NEW2 = '  TrendingUp, Calendar, Wifi, Inbox,'
if OLD2 in content:
    content = content.replace(OLD2, NEW2, 1)
    changes += 1
    print("✅ Lucide Inbox import added")
else:
    print("⚠️  Lucide import target not found")

# ── 3. Add API function imports ───────────────────────────────────────────────
OLD3 = '  getAdminStats,'
NEW3 = '  getAdminStats, getAdminContacts, markContactRead, deleteContactMessage,'
if OLD3 in content:
    content = content.replace(OLD3, NEW3, 1)
    changes += 1
    print("✅ API imports updated")
else:
    print("⚠️  API imports target not found")

# ── 4. Add ContactMessage type import ────────────────────────────────────────
OLD4 = 'import type { HistorySummary, FeedbackItem, AdminAnalysisItem, AdminUserItem, AdminStatsResponse, AdminStatsActivity } from "@/types/api";'
NEW4 = 'import type { HistorySummary, FeedbackItem, AdminAnalysisItem, AdminUserItem, AdminStatsResponse, AdminStatsActivity, ContactMessage, AdminContactsResponse } from "@/types/api";'
if OLD4 in content:
    content = content.replace(OLD4, NEW4, 1)
    changes += 1
    print("✅ Type imports updated")
else:
    print("⚠️  Type imports target not found")

# ── 5. Add ContactsTab sub-component before TABS constant ────────────────────
OLD5 = '// -- Main component ----------------------------------------------------------\n\nconst TABS:'
CONTACTS_TAB = '''\
// -- ContactsTab sub-component -----------------------------------------------

function ContactsTab({ sessionStatus }: { sessionStatus: string }) {
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
                      {isOpen ? "▲" : "▼"}
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
                        style={{ backgroundColor: "var(--brand-gold)", color: "#0a1628" }}
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

'''

NEW5 = CONTACTS_TAB + '// -- Main component ----------------------------------------------------------\n\nconst TABS:'
if OLD5 in content:
    content = content.replace(OLD5, NEW5, 1)
    changes += 1
    print("✅ ContactsTab component inserted")
else:
    print("⚠️  ContactsTab insertion point not found")

# ── 6. Add to TABS array ──────────────────────────────────────────────────────
OLD6 = '  { id: "users",       label: "Users",       icon: UserCog,              privileged: true  },'
NEW6 = ('  { id: "users",       label: "Users",       icon: UserCog,              privileged: true  },\n'
        '  { id: "contacts",    label: "Contacts",    icon: Inbox,                privileged: true  },')
if OLD6 in content:
    content = content.replace(OLD6, NEW6, 1)
    changes += 1
    print("✅ TABS entry added")
else:
    print("⚠️  TABS entry target not found")

# ── 7. Add contacts tab to the render block ───────────────────────────────────
OLD8 = '          {activeTab === "users"       && <UsersTab       sessionStatus={sessionStatus} />}'
NEW8 = ('          {activeTab === "users"       && <UsersTab       sessionStatus={sessionStatus} />}\n'
        '          {activeTab === "contacts"    && <ContactsTab    sessionStatus={sessionStatus} />}')
if OLD8 in content:
    content = content.replace(OLD8, NEW8, 1)
    changes += 1
    print("✅ Contacts render added")
else:
    print("⚠️  Render block target not found")

# -- Save --
with open(ADMIN_PAGE, "w", encoding="utf-8-sig") as f:
    f.write(content)

print(f"\n{'✅ Done' if changes >= 7 else '⚠️ Partial'} — {changes}/7 changes applied")
sys.exit(0 if changes >= 7 else 1)
