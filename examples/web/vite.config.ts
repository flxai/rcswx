import { defineConfig } from "vite";
export default defineConfig({
  base: "/rcswx/",
  worker: { format: "es" },
  build: {
    assetsInlineLimit: 0,
    rolldownOptions: { input: { main: "index.html", harness: "harness.html" } },
  },
  preview: {
    headers: {
      "Content-Security-Policy":
        "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'",
    },
  },
});
