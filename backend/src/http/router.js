import { parse } from 'url';

export class Router {
  constructor() {
    this.routes = [];
  }

  register(method, path, handler) {
    const parts = path.split('/').filter(Boolean);
    const params = parts.map((part) => part.startsWith(':'));
    this.routes.push({ method, parts, params, handler });
  }

  async handle(req, res, context) {
    const { pathname } = parse(req.url, true);
    const segments = pathname.split('/').filter(Boolean);
    for (const route of this.routes) {
      if (route.method !== req.method) continue;
      if (route.parts.length !== segments.length) continue;
      const params = {};
      let match = true;
      for (let i = 0; i < route.parts.length; i++) {
        const routePart = route.parts[i];
        const seg = segments[i];
        if (route.params[i]) {
          params[routePart.slice(1)] = decodeURIComponent(seg);
        } else if (routePart !== seg) {
          match = false;
          break;
        }
      }
      if (!match) continue;
      return route.handler(req, res, { ...context, params });
    }
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not Found' }));
  }
}
