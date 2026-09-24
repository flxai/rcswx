import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  timeout: 45_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["json", { outputFile: "test-results/results.json" }]],
  use: { baseURL: "http://127.0.0.1:4173/rcswx/", trace: "retain-on-failure" },
  projects: [
    { name: "chromium", use: { browserName: "chromium" } },
    { name: "firefox", use: { browserName: "firefox" } },
    { name: "webkit", use: { browserName: "webkit" } },
  ],
  webServer: {
    command: "npm run preview",
    url: "http://127.0.0.1:4173/rcswx/",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
