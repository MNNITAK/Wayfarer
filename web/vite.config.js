import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// local job server (pipeline/server.py) for the upload studio
const proxy = { '/api': 'http://127.0.0.1:8765' };

// `npm run phone`: HTTPS on the LAN with a self-signed certificate, because phone browsers only allow the camera
// and motion sensors on secure pages (the Capture Coach). Plain `npm run dev` stays HTTP for the desktop.
export default defineConfig(({ mode }) => ({
  plugins: mode === 'phone' ? [basicSsl()] : [],
  server: { proxy },
  preview: { proxy },
  build: { rolldownOptions: { input: { main: 'index.html', upload: 'upload.html', capture: 'capture.html' } } },
}));
