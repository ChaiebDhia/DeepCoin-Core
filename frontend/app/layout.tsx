import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Providers    from "@/providers";
import { Header }  from "@/components/ui/header";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title:       "DeepCoin — Ancient Coin Intelligence",
  description: "AI-powered archaeological coin classification and historical analysis. EfficientNet-B3 CNN + Multi-Agent RAG System.",
  keywords:    ["numismatics", "ancient coins", "AI", "deep learning", "Corpus Nummorum"],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col`}
        style={{ backgroundColor: "var(--surface-0)" }}
      >
        <Providers>
          <Header />
          <main className="flex-1 mx-auto w-full max-w-6xl px-5 py-8">
            {children}
          </main>
          <footer
            className="border-t border-[var(--border)] py-4 text-center text-xs"
            style={{ color: "var(--text-muted)" }}
          >
            DeepCoin · ESPRIT × YEBNI · Dhia Chaieb · 2026
          </footer>
        </Providers>
      </body>
    </html>
  );
}
