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

      // Read the body into a buffer so it survives the new Request() constructor.
      // Streaming bodies (request.body) can be silently dropped in some CF runtime versions.
      let body = undefined;
      if (request.method !== 'GET' && request.method !== 'HEAD') {
        body = await request.arrayBuffer();
      }

      // Copy headers, removing 'host' so the backend sees its own host.
      const headers = new Headers(request.headers);
      headers.delete('host');

      try {
        const proxyRes = await fetch(targetUrl, {
          method: request.method,
          headers,
          body,
          redirect: 'follow',
        });
        // Clone the response so we can return it (responses are single-use).
        return new Response(proxyRes.body, {
          status: proxyRes.status,
          statusText: proxyRes.statusText,
          headers: proxyRes.headers,
        });
      } catch (err) {
        return new Response(`Backend unreachable: ${err.message}`, { status: 502 });
      }
    }

    // Fall through to static assets
    return env.ASSETS.fetch(request);
  },
};
