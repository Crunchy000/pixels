import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

// `--mode single` inlines everything into one self-contained index.html
// (handy for sharing the demo as a single file).
export default defineConfig(({ mode }) => ({
  base: './',
  plugins: mode === 'single' ? [viteSingleFile()] : [],
  build: {
    outDir: mode === 'single' ? 'dist-single' : 'dist',
    target: 'es2022',
  },
}));
