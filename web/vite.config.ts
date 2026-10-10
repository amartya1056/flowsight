import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `vite build --mode static` -> static replay build for GitHub Pages (served under /flowsight/, VITE_STATIC=1 via .env.static)
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === 'static' ? '/flowsight/' : '/',
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', ws: true, changeOrigin: true } },
  },
  build: { chunkSizeWarningLimit: 1600, outDir: mode === 'static' ? 'dist-static' : 'dist' },
}))
