import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  base: "/chronicle-assets/",
  build: {
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, "frontend/graph.ts"),
      name: "ChronicleGraphBundle",
      formats: ["iife"],
      fileName: () => "graph.js",
    },
    outDir: resolve(__dirname, "static"),
    sourcemap: true,
  },
});
