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
      // Handle CORS preflight
      if (request.method === 'OPTIONS') {
        const origin = request.headers.get('Origin') || '*';
        return new Response(null, {
          status: 204,
          headers: {
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, X-Session-ID',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Max-Age': '86400',
          },
        });
      }

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

        const resHeaders = new Headers(proxyRes.headers);
        const origin = request.headers.get('Origin') || '*';
        resHeaders.set('Access-Control-Allow-Origin', origin);
        resHeaders.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
        resHeaders.set('Access-Control-Allow-Headers', 'Content-Type, X-Session-ID');
        resHeaders.set('Access-Control-Allow-Credentials', 'true');

        return new Response(proxyRes.body, {
          status: proxyRes.status,
          statusText: proxyRes.statusText,
          headers: resHeaders,
        });
      } catch (err) {
        return new Response(`Backend unreachable: ${err.message}`, { status: 502 });
      }
    }

    // Fall through to static assets
    return env.ASSETS.fetch(request);
  },
};
