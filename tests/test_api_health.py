import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer


class APIHealthSmokeTest(unittest.TestCase):
    def test_health_endpoint_boots_without_gradio(self):
        # A missing optional UI dependency must not prevent API-only startup.
        sys.modules["gradio"] = None

        from backend.api_server import APIServer

        server = ThreadingHTTPServer(("127.0.0.1", 0), APIServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            connection.request("GET", "/api/v1/health")
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(payload["status"], "ok")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
