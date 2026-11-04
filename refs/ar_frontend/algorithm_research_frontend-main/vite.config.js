export default {
  base: './',
  esbuild: { jsx: 'automatic' },
  server: { host: '127.0.0.1', watch: { ignored: ['**/test_results/**'] } },
  build: { sourcemap: false },
};
