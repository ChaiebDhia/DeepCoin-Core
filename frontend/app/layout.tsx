import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Providers      from "@/providers";
import { Header }    from "@/components/ui/header";
import { Footer }    from "@/components/ui/footer";
import { MainLayout } from "@/components/ui/MainLayout";
import TutorialModal from "@/components/ui/TutorialModal";

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
    <html lang="en" suppressHydrationWarning data-scroll-behavior="smooth">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col transition-colors duration-200`}
      >
        <Providers>
          <Header />
          <MainLayout>
            {children}
          </MainLayout>
          <Footer />
          <TutorialModal />
        </Providers>
      </body>
    </html>
  );
}

