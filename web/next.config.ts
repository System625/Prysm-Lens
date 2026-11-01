import type { NextConfig } from "next";

const config: NextConfig = {
  images: { remotePatterns: [{ protocol: "https", hostname: "images.igdb.com" }] },
};

export default config;
