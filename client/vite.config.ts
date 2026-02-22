import { defineConfig, loadEnv } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  console.log("Starting Vite with proxy to:", env.VITE_QUEUE_API_URL || "http://localhost:8998");
  
  return {
    server: {
      host: "0.0.0.0",
      port: 5173,
      https: false,
      proxy: {
        "/api": {
          target: env.VITE_QUEUE_API_URL || "http://localhost:8998",
          changeOrigin: true,
          secure: false,
        },
      },
    },
    plugins: [
      topLevelAwait({
        promiseExportName: "__tla",
        promiseImportName: i => `__tla_${i}`,
      }),
    ],
  };
});
