import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Warm editorial palette -- all pairings checked against WCAG AA
        // (4.5:1 for normal text) before use: ink 17.1:1, ink-soft 9.0:1,
        // muted 5.6:1, accent 6.8:1, all measured against `paper`.
        paper: "#FAF7F1",
        "paper-alt": "#F2EBDD",
        ink: "#18140F",
        "ink-soft": "#4A443C",
        muted: "#6B6255",
        rule: "#E1D8C6",
        accent: "#9A3324",
      },
      fontFamily: {
        // System/OS serif and sans stacks only -- no remote font fetch,
        // so the build never depends on network access to a font CDN.
        serif: ["Georgia", "Cambria", "Times New Roman", "Times", "serif"],
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
