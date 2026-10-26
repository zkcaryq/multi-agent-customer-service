import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: 'src/main.tsx',
      name: 'MultiAgentCustomerServiceWidget',
      formats: ['iife'],
      fileName: () => 'ma-cs-widget.js',
    },
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        assetFileNames: 'ma-cs-widget.[ext]',
      },
    },
  },
  server: {
    port: 5173,
    open: true,
  },
});
