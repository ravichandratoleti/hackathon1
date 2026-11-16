import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  env: { FRONTEND_ORIGIN: process.env.FRONTEND_ORIGIN || "" },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${process.env.BACKEND_URL || "http://localhost:8000"}/api/:path*` }];
  },
};
export default nextConfig;
