import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  base: "/chronicle-assets/build/",
  build: {
    emptyOutDir: true,
    lib: {
      entry: resolve(__dirname, "frontend/graph.ts"),
      name: "ChronicleGraphBundle",
      formats: ["iife"],
      fileName: () => "graph.js",
    },
    outDir: resolve(__dirname, "static/build"),
    sourcemap: true,
  },
});
