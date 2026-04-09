"use client";
import { usePathname } from "next/navigation";

export function MainLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isChat = pathname === "/chat";
  
  return (
    <main className={`flex-1 w-full mx-auto ${!isChat ? "max-w-6xl px-5" : ""}`}>
      {children}
    </main>
  );
}
