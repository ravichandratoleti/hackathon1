import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "VERITAS AI — Verify what you see", description: "Evidence led fact checking and media verification." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
