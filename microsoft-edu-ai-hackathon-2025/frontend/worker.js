/**
 * Cloudflare Worker — proxies API requests to the Flask backend,
 * serves static assets for everything else.
 *
 * Required environment variable (set via Cloudflare dashboard or
 * `wrangler secret put BACKEND_URL`):
 *   BACKEND_URL — e.g. https://your-tunnel.trycloudflare.com
 */

const API_ROUTES = [
  '/discover',
  '/extract',
  '/extract-local',
  '/train',
  '/predict',
  '/analyze',
  '/status',
  '/state',
  '/health',
  '/queue-info',
  '/reset',
];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    const isApi = API_ROUTES.some(
      (r) => url.pathname === r || url.pathname.startsWith(r + '/'),
    );

    if (isApi) {
      if (!env.BACKEND_URL) {
        return new Response('BACKEND_URL secret not configured', { status: 502 });
      }

      const backendBase = env.BACKEND_URL.replace(/\/$/, '');
      const targetUrl = backendBase + url.pathname + url.search;

      // Stream the body (needed for large file uploads).
      // NOTE: Cloudflare Workers free tier caps request bodies at 100 MB.
      // For larger uploads set BACKEND_URL in the frontend build instead
      // (VITE_API_BASE) so the browser uploads directly to the backend.
      const proxyReq = new Request(targetUrl, {
        method: request.method,
        headers: request.headers,
        body: request.method !== 'GET' && request.method !== 'HEAD'
          ? request.body
          : undefined,
        redirect: 'follow',
        duplex: 'half',
      });

      try {
        return await fetch(proxyReq);
      } catch (err) {
        return new Response(`Backend unreachable: ${err.message}`, { status: 502 });
      }
    }

    // Fall through to static assets
    return env.ASSETS.fetch(request);
  },
};
