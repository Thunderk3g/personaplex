import { defineConfig, loadEnv } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";
import fs from "node:fs";
import path from "node:path";

const loadHttpsConfig = (keyPath: string, certPath: string) => {
  if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
    throw new Error(
      `HTTPS is enabled but TLS files are missing. Expected key at "${keyPath}" and cert at "${certPath}".`,
    );
  }
  return {
    key: fs.readFileSync(keyPath),
    cert: fs.readFileSync(certPath),
  };
};

export default defineConfig(({ mode, command }) => {
  const env = loadEnv(mode, process.cwd());
  const queueApiUrl = env.VITE_QUEUE_API_URL || "http://localhost:8998";
  const useHttps = env.VITE_DEV_HTTPS !== "false";
  const tlsKeyPath = env.VITE_DEV_TLS_KEY || path.resolve(process.cwd(), "key.pem");
  const tlsCertPath = env.VITE_DEV_TLS_CERT || path.resolve(process.cwd(), "cert.pem");
  const devHttps = command === "serve" && useHttps;

  console.log("Starting Vite with proxy to:", queueApiUrl);
  if (command === "serve") {
    console.log("Vite dev HTTPS:", useHttps ? "enabled" : "disabled");
  }
  
  return {
    server: {
      host: "0.0.0.0",
      port: 5173,
      https: devHttps ? loadHttpsConfig(tlsKeyPath, tlsCertPath) : false,
      proxy: {
        "/api": {
          target: queueApiUrl,
          changeOrigin: true,
          secure: false,
          ws: true,
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
