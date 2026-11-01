import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prysm Lens",
  description: "Name the video game in any gameplay clip. SAM 3.1 + Meta Perception Encoder + Muse Spark.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0a0a0b" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-dvh antialiased">{children}</body>
    </html>
  );
}
