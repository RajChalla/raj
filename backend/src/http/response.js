export function sendJson(res, statusCode, payload, headers = {}) {
  res.writeHead(statusCode, { 'Content-Type': 'application/json', ...headers });
  res.end(JSON.stringify(payload));
}

export function unauthorized(res) {
  sendJson(res, 401, { error: 'Unauthorized' });
}

export function forbidden(res) {
  sendJson(res, 403, { error: 'Forbidden' });
}

export function badRequest(res, message) {
  sendJson(res, 400, { error: message });
}
