"""A tiny local stand-in for the Gemini REST API used to test the real SDK."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class State:
    mode = "ok"            # ok | 403_key | 429 | 503 | empty | blocked | 404
    requests = []          # captured request bodies
    reply = "FAKE ANSWER [1]"
    fail_first_n = 0       # transient failures before succeeding
    calls = 0


def _ok(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        State.requests.append({"path": self.path, "body": body,
                               "key": self.headers.get("x-goog-api-key")})
        State.calls += 1
        m = State.mode
        if State.calls <= State.fail_first_n:
            return self._send(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
        if m == "403_key":
            return self._send(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}})
        if m == "429":
            return self._send(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        if m == "503":
            return self._send(503, {"error": {"code": 503, "message": "down", "status": "UNAVAILABLE"}})
        if m == "404":
            return self._send(404, {"error": {"code": 404, "message": "model not found", "status": "NOT_FOUND"}})
        if m == "empty":
            return self._send(200, {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]})
        if m == "blocked":
            return self._send(200, {"promptFeedback": {"blockReason": "SAFETY"}})
        return self._send(200, _ok(State.reply))


def start():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"
