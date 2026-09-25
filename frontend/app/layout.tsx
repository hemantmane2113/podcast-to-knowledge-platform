import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Conversely — Long Conversations. Powerful Ideas.",
  description:
    "Ideas belong to the people who expressed them. We turn the conversation into a readable story.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-paper font-sans text-ink antialiased">{children}</body>
    </html>
  );
}
