import random
import socket
import threading
import time

import websocket


class WebSocketRsyncTunnel:
    def __init__(self, local_port: int, ws_url: str):
        self.requested_port = local_port
        self.local_port = None  # Will be set in __enter__
        self.ws_url = ws_url
        self.running = False
        self.server_socket = None

    def __enter__(self):
        self.running = True

        # Add randomization to reduce concurrent collision probability
        # Try multiple times with different socket instances
        max_attempts = 20
        port_range = 100  # Much wider range to avoid collisions

        for attempt in range(max_attempts):
            # Add random offset to spread out concurrent requests
            random_offset = random.randint(0, 50)
            start_port = self.requested_port + random_offset

            # Create a new socket for each attempt
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Try ports in this range
            for i in range(port_range):
                port = start_port + i
                try:
                    server_socket.bind(("127.0.0.1", port))
                    server_socket.listen(5)
                    # Success! Save the socket and port
                    self.server_socket = server_socket
                    self.local_port = port
                    break
                except OSError as e:
                    if e.errno == 98:  # Address already in use
                        continue
                    else:
                        server_socket.close()
                        raise
            else:
                # No ports available in this range, close socket and try again
                server_socket.close()
                # Add small random delay to reduce simultaneous collision probability
                time.sleep(random.uniform(0.01, 0.05))
                continue

            # Websocket server successfully bound and is listening
            break
        else:
            raise RuntimeError(
                f"Could not find available port after {max_attempts} attempts starting from {self.requested_port}"
            )

        threading.Thread(target=self._accept_loop, daemon=True).start()

        # Wait for ready
        for _ in range(50):
            try:
                with socket.socket() as s:
                    s.settimeout(0.1)
                    if s.connect_ex(("127.0.0.1", self.local_port)) == 0:
                        return self
            except:
                pass
            time.sleep(0.1)
        raise RuntimeError("Tunnel failed to start")

    def __exit__(self, *args):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass  # Already closed

    def _accept_loop(self):
        while self.running:
            try:
                client_sock, _ = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True).start()
            except:
                break

    def _handle_client(self, client_sock):
        ws = None
        try:
            ws = websocket.create_connection(self.ws_url)

            def tcp_to_ws():
                while self.running:
                    try:
                        data = client_sock.recv(65536)
                        if not data:
                            break
                        ws.send_binary(data)
                    except:
                        break

            def ws_to_tcp():
                while self.running:
                    try:
                        data = ws.recv()
                        if isinstance(data, bytes):
                            client_sock.send(data)
                    except:
                        break

            t1 = threading.Thread(target=tcp_to_ws, daemon=True)
            t2 = threading.Thread(target=ws_to_tcp, daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        except Exception as e:
            print(f"WebSocket connection error: {e}")

        finally:
            # close both connections
            for conn in [client_sock, ws]:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
