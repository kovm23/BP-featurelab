import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    // TADY JE TA ZMĚNA:
    allowedHosts: true, // Povolí přístup z jakékoli Cloudflare adresy
    host: true,         // Nutné pro poslech na síťovém rozhraní
    port: 5173,
    proxy: {
      '/upload': { target: 'http://127.0.0.1:5000', changeOrigin: true },
      '/train': { target: 'http://127.0.0.1:5000', changeOrigin: true },
      '/status': { target: 'http://127.0.0.1:5000', changeOrigin: true },
    },
  },
  preview: {
    allowedHosts: true,
    host: true,
    port: 5173,
  },
})