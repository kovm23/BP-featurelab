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
  '/export-session',
  '/import-session',
];

const DEFAULT_PERMISSIONS_POLICY = [
  'camera=()',
  'microphone=()',
  'geolocation=()',
  'payment=()',
  'usb=()',
  'fullscreen=(self)',
].join(', ');

function sanitizePermissionsPolicy(value) {
  if (!value || typeof value !== 'string') {
    return DEFAULT_PERMISSIONS_POLICY;
  }

  const cleaned = value
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part && !part.toLowerCase().startsWith('browsing-topics='));

  return cleaned.length > 0 ? cleaned.join(', ') : DEFAULT_PERMISSIONS_POLICY;
}

function resolveBackendCandidates(env) {
  const raw = [env.BACKEND_URL, env.BACKEND_URL_FALLBACK]
    .filter(Boolean)
    .join(',');

  const urls = raw
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => entry.replace(/\/$/, ''));

  return [...new Set(urls)];
}

function applyCommonResponseHeaders(headers, request) {
  const origin = request.headers.get('Origin') || '*';
  headers.set('Access-Control-Allow-Origin', origin);
  headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  headers.set('Access-Control-Allow-Headers', 'Content-Type, X-Session-ID');
  headers.set('Access-Control-Allow-Credentials', 'true');
  headers.set(
    'Permissions-Policy',
    sanitizePermissionsPolicy(headers.get('Permissions-Policy')),
  );
}

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
            'Permissions-Policy': DEFAULT_PERMISSIONS_POLICY,
          },
        });
      }

      const backendCandidates = resolveBackendCandidates(env);
      if (backendCandidates.length === 0) {
        return new Response('No backend URL configured (BACKEND_URL)', {
          status: 502,
          headers: { 'Content-Type': 'text/plain; charset=utf-8' },
        });
      }

      // Read the body into a buffer so it survives the new Request() constructor.
      // Streaming bodies (request.body) can be silently dropped in some CF runtime versions.
      let body = undefined;
      if (request.method !== 'GET' && request.method !== 'HEAD') {
        body = await request.arrayBuffer();
      }

      // Copy headers, removing 'host' so the backend sees its own host.
      const headers = new Headers(request.headers);
      headers.delete('host');

      const failures = [];
      for (const backendBase of backendCandidates) {
        const targetUrl = backendBase + url.pathname + url.search;
        try {
          const proxyRes = await fetch(targetUrl, {
            method: request.method,
            headers,
            body,
            redirect: 'follow',
          });

          // Retry next backend candidate on origin edge failures.
          if (proxyRes.status >= 520 && proxyRes.status <= 530) {
            failures.push({ backend: backendBase, status: proxyRes.status });
            continue;
          }

          // Buffer the response body so it's not lost in stream handoff.
          const resBody = await proxyRes.arrayBuffer();
          const resHeaders = new Headers(proxyRes.headers);
          applyCommonResponseHeaders(resHeaders, request);

          return new Response(resBody, {
            status: proxyRes.status,
            statusText: proxyRes.statusText,
            headers: resHeaders,
          });
        } catch (err) {
          failures.push({
            backend: backendBase,
            error: err && err.message ? err.message : String(err),
          });
        }
      }

      return new Response(
        JSON.stringify(
          {
            error: 'All configured backends are unreachable',
            failures,
          },
          null,
          2,
        ),
        {
          status: 502,
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
            'Permissions-Policy': DEFAULT_PERMISSIONS_POLICY,
          },
        });
    }

    // Fall through to static assets
    const assetRes = await env.ASSETS.fetch(request);
    const headers = new Headers(assetRes.headers);
    applyCommonResponseHeaders(headers, request);
    return new Response(assetRes.body, {
      status: assetRes.status,
      statusText: assetRes.statusText,
      headers,
    });
  },
};
