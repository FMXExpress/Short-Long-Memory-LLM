"""A stand-in for an OpenJev helper / Jev API: POST /v1/systemone with the same
request and response shapes. It "decides" with a keyword check on the state,
so jevban can be run end to end without a GPU. Usage: mock_jev.py [port]"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BAD = ("env-file", "vcs-dir", "traversal", "log4shell", "sqli", "php-rce", "scanner-agent", "binary-request")

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state = req["state"]
        hits = state.get("signature_hits", "")
        bad = any(t in hits for t in BAD) or "config-file" in hits
        p = 0.99 if bad else 0.08
        intent = "vulnerability_scanner" if bad else "legitimate_visitor"
        sys.stderr.write("mock-jev: %s hits=%s recent=%d -> %.2f\n" % (
            state["client_ip"], hits, len(state["recent_requests_from_this_client_oldest_first"]), p))
        body = json.dumps({"id": "shim-1", "model": "mock",
            "answers": {"malicious": {"type": "noul", "noul": p},
                        "intent": {"type": "choice", "choice": intent, "probabilities": {}, "confidence": 0.9}},
            "usage": {"input_tokens": 0, "output_tokens": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 3000), H).serve_forever()
