import { defineConfig, createLogger } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const frontendRoot = path.resolve(__dirname)

// Suppress ECONNABORTED noise — Vite 8 logs these via `logger.error` whenever
// the browser disconnects (page refresh, tab close) while the dev proxy is
// forwarding our game WebSocket. It is not an application error.
const logger = createLogger()
const originalError = logger.error.bind(logger)
logger.error = (msg, options) => {
  if (msg.includes('ws proxy') && msg.includes('ECONNABORTED')) return
  originalError(msg, options)
}

export default defineConfig({
  customLogger: logger,
  root: frontendRoot,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        configure: (proxy) => {
          proxy.on('error', () => {})
        },
      },
    },
  },
})
