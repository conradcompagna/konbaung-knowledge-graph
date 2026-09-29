const { defineConfig } = require("playwright/test");

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: /test_.*\.spec\.js$/,
  // backups/ holds verbatim copies of the specs alongside the sources they
  // were taken with; without this the runner collects them twice.
  testIgnore: [
    "**/backups/**",
    "**/node_modules/**",
    "**/artifacts/**",
    // These retain diagnostic coverage for the archived raw renderer, which
    // is intentionally no longer reachable from the thematic-only reader UI.
    "**/test_graph_corpus_radial.spec.js",
    "**/test_graph_hard_collision.spec.js",
    "**/test_graph_layout_perf.spec.js",
    "**/test_graph_probe.spec.js",
    "**/test_graph_tag_node_shift.spec.js",
  ],
  reporter: "line",
  use: { channel: "chrome" },
});
