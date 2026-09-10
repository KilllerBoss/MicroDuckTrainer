import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

export const metadata: Metadata = {
  title: "MicroDuck Trainer",
  description:
    "MuJoCo-Roboter-Training: Ente & Unitree G1 mit neuronalen Policies, Touch-Gamepad und ES-Training – komplett offline.",
  applicationName: "MicroDuck Trainer",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#05060a",
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="de" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground">
        {children}
        {/* Toaster MUSS innerhalb von <body> stehen (Hydration-Fehler sonst) */}
        <Toaster />
      </body>
    </html>
  );
}
