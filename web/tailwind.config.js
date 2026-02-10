/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        bg: {
          primary: "#0a0a0f",
          card: "#12121a",
          hover: "#1a1a28",
        },
        border: {
          subtle: "#1e1e30",
        },
        text: {
          primary: "#e8e8f0",
          secondary: "#8888aa",
        },
        accent: {
          cyan: "#00d4ff",
          blue: "#0066ff",
        },
        up: "#22c55e",
        down: "#ef4444",
      },
    },
  },
  plugins: [],
};
