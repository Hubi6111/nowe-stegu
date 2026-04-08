import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], display: "swap", variable: "--font-inter" });
const playfair = Playfair_Display({ subsets: ["latin"], display: "swap", variable: "--font-playfair" });

export const metadata: Metadata = {
  title: "Stegu Visualizer",
  description: "Wizualizuj kamień dekoracyjny i cegłę Stegu na swoich ścianach",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl">
      <body className={`${inter.variable} ${playfair.variable} font-sans bg-[#faf9f7] text-stone-800 min-h-screen antialiased`}>
        {children}
      </body>
    </html>
  );
}
