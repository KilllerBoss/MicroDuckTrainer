import type { NextConfig } from "next";

// Static-Export-kompatibel: NEXT_OUTPUT=export erzeugt out/ (APK-Verpackung),
// sonst standalone (Dev/Server).
const nextConfig: NextConfig = {
  output: process.env.NEXT_OUTPUT === "export" ? "export" : "standalone",
  images: { unoptimized: true },
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
};

export default nextConfig;
