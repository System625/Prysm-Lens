import type { NextConfig } from "next";
import { networkInterfaces } from "node:os";

// Testing on a phone means opening the dev server at this Mac's wifi address (http://10.x.x.x:3000). `next dev` blocks
// its scripts for any host but localhost, so the page arrived without JavaScript: only the plain file picker worked and
// the tabs ignored taps (1 Nov). Allow this machine's own LAN addresses, read at startup since the wifi IP changes.
// Dev only: production builds don't serve these resources.
const lanAddresses = Object.values(networkInterfaces())
  .flat()
  .filter((a) => a && a.family === "IPv4" && !a.internal)
  .map((a) => a!.address);

const config: NextConfig = {
  images: { remotePatterns: [{ protocol: "https", hostname: "images.igdb.com" }] },
  allowedDevOrigins: lanAddresses,
};

export default config;
