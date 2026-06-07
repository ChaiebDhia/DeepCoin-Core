import { cookies } from "next/headers";
import type { Metadata } from "next";
import "./globals.css";
import Providers      from "@/providers";
import { Header }    from "@/components/ui/header";
import { Footer }    from "@/components/ui/footer";
import { MainLayout } from "@/components/ui/MainLayout";
import TutorialModal from "@/components/ui/TutorialModal";

export const metadata: Metadata = {
  title:       "DeepCoin — Ancient Coin Intelligence",
  description: "AI-powered archaeological coin classification and historical analysis. EfficientNet-B3 CNN + Multi-Agent RAG System.",
  keywords:    ["numismatics", "ancient coins", "AI", "deep learning", "Corpus Nummorum"],
};

export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const cookieStore = await cookies();
  const locale = cookieStore.get('NEXT_LOCALE')?.value || 'en';
  let messages = {};
  try {
    messages = (await import("../messages/" + locale + ".json")).default;
  } catch (error) {
    messages = (await import("../messages/en.json")).default;
  }

  return (
    <html lang={locale} suppressHydrationWarning data-scroll-behavior="smooth">
      <body
        suppressHydrationWarning
        className="antialiased min-h-screen flex flex-col transition-colors duration-200"
      >
        <Providers locale={locale} messages={messages}>
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



