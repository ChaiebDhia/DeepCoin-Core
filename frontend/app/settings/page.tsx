"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getSubscriptionStatus, unsubscribeMe } from "@/lib/api";
import { User, Bell, ShieldCheck } from "lucide-react";
import { motion } from "framer-motion";

export default function SettingsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login?callbackUrl=/settings");
    }
  }, [status, router]);

  const { data: subStatus, refetch: refetchSub } = useQuery({
    queryKey: ["user", "subscription_status"],
    queryFn:  getSubscriptionStatus,
    enabled:  status === "authenticated",
    staleTime: 60_000,
  });

  if (status === "loading" || status === "unauthenticated") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 rounded-full animate-pulse" style={{ background: "var(--surface-2)" }} />
      </div>
    );
  }

  const nextUser = session!.user as { name?: string | null, email?: string | null, image?: string | null, display_name?: string, role?: string };
  const label = nextUser.display_name || nextUser.name || "User";
  const role = nextUser.role ?? "analyst";

  return (
    <div className="min-h-screen py-10" style={{ backgroundColor: "var(--background)" }}>
      <div className="max-w-3xl mx-auto px-4 sm:px-6 space-y-8">
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <h1 className="text-2xl font-black" style={{ color: "var(--text-primary)" }}>
            Account Settings
          </h1>
          <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
            Manage your profile and preferences.
          </p>
        </motion.div>

        <div className="space-y-6">
          {/* Profile Section */}
          <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: "var(--text-muted)" }}>Profile</h2>
            <div className="rounded-xl border p-5" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full flex items-center justify-center bg-[var(--surface-2)]">
                  <User size={24} style={{ color: "var(--text-secondary)" }} />
                </div>
                <div>
                  <p className="font-medium" style={{ color: "var(--text-primary)" }}>{label}</p>
                  <p className="text-sm" style={{ color: "var(--text-muted)" }}>{nextUser.email}</p>
                  <p className="text-xs font-semibold mt-1 capitalize" style={{ color: "var(--brand-gold)" }}>Role: {role}</p>
                </div>
              </div>
            </div>
          </motion.section>

          {/* Preferences Section */}
          <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: "var(--text-muted)" }}>Preferences</h2>
            <div className="rounded-xl border divide-y" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
              <div className="p-5 flex items-center justify-between gap-4">
                <div className="flex gap-3">
                  <div className="mt-0.5">
                    <Bell size={18} style={{ color: "var(--text-secondary)" }} />
                  </div>
                  <div>
                    <p className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>Email Notifications</p>
                    <p className="text-xs max-w-md mt-0.5" style={{ color: "var(--text-muted)" }}>
                      Receive updates about new features, model upgrades, and product news.
                    </p>
                  </div>
                </div>
                <div>
                  {subStatus?.subscribed ? (
                    <button
                      onClick={async () => {
                        try {
                          await unsubscribeMe();
                          await refetchSub();
                        } catch(e) { console.error(e); }
                      }}
                      className="px-4 py-1.5 rounded-lg border text-sm font-medium transition-colors hover:bg-[var(--surface-2)]"
                      style={{ borderColor: "var(--border)", color: "#ef4444" }}
                    >
                      Unsubscribe
                    </button>
                  ) : (
                    <span className="text-xs px-3 py-1 font-medium rounded-full" style={{ backgroundColor: "var(--surface-2)", color: "var(--text-muted)" }}>
                      Unsubscribed
                    </span>
                  )}
                </div>
              </div>
            </div>
          </motion.section>
        </div>
      </div>
    </div>
  );
}

