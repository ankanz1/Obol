import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const isDev = mode === 'development'
  return {
    plugins: [react()],
    server: isDev ? {
      port: 3000,
      proxy: {
        '/api': 'http://localhost:8000',
        '/ws': {
          target: 'ws://localhost:8000',
          ws: true
        }
      }
    } : undefined,
    define: {
      'process.env': {}
    }
  }
})