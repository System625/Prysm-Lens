import type { Metadata, Viewport } from "next";
import { Nunito } from "next/font/google";
import "./globals.css";

// Prysm's face on the landing page (prysm-web src/fonts/nunito.ts). The display line uses weight 1000.
const nunito = Nunito({ subsets: ["latin"], variable: "--font-nunito" });

export const metadata: Metadata = {
  title: "Prysm Lens",
  description: "Name the video game in any gameplay clip. SAM 3.1 + Meta Perception Encoder + Muse Spark.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#090909" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={nunito.variable}>
      <body className="min-h-dvh antialiased">{children}</body>
    </html>
  );
}
