import { defineConfig, createLogger } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const frontendRoot = path.resolve(__dirname)

// Suppress ws proxy noise — Vite logs these via `logger.error` for transient
// WebSocket connection failures that are never actual application errors:
//   ECONNREFUSED  backend not yet ready when the browser opens at startup
//   ECONNABORTED  browser disconnected (page refresh, tab close)
//   ECONNRESET    connection torn down mid-flight
// All are self-healing: the frontend reconnects automatically once the backend
// is up. Showing them just pollutes the terminal on every start.
const logger = createLogger()
const originalError = logger.error.bind(logger)
logger.error = (msg, options) => {
  if (msg.includes('ws proxy')) return
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
