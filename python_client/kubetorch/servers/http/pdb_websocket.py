"""WebSocket-based remote PDB debugging.

This module provides remote PDB debugging over WebSocket. The architecture is:

1. When breakpoint() is called, we start a WebSocket server and redirect stdin/stdout
2. PDB runs normally with set_trace() in the user's execution context
3. A remote client connects via WebSocket and streams stdin/stdout

The key insight is that PDB must run in the user's execution flow, not in a separate
context after waiting for a client. We achieve this by:
- Starting the WebSocket server in a background thread
- Redirecting stdin/stdout to WebSocket-backed streams BEFORE calling set_trace()
- PDB then blocks on our custom stdin, which waits for WebSocket client input
"""

import asyncio
import os
import pdb
import queue
import sys
import threading

import websockets

from kubetorch.logger import get_logger

logger = get_logger(__name__)

# Global state
_websocket_server = None
_websocket_connection = None
_server_thread = None
_event_loop = None


class WebSocketIO:
    """File-like object that bridges PDB's stdin/stdout to a WebSocket connection.

    This class provides blocking readline() for stdin and immediate write() for stdout.
    It uses thread-safe queues to communicate between:
    - The async WebSocket thread (receives from client, sends to client)
    - The main thread where PDB runs (reads via readline(), writes via write())
    """

    def __init__(self, connection_timeout: int = 300):
        self.input_queue = queue.Queue()  # WebSocket -> PDB
        self.output_queue = queue.Queue()  # PDB -> WebSocket
        self._closed = False
        self._connected = threading.Event()
        self._line_buffer = ""  # Buffer for accumulating characters into lines
        self._connection_timeout = connection_timeout

    def set_connected(self):
        """Signal that a WebSocket client has connected."""
        self._connected.set()

    def wait_for_connection(self, timeout: float = 300) -> bool:
        """Wait for a WebSocket client to connect."""
        return self._connected.wait(timeout=timeout)

    def write(self, data: str) -> int:
        """Write data to be sent to the WebSocket client."""
        if self._closed:
            return 0
        if data:
            self.output_queue.put(data)
        return len(data)

    def readline(self) -> str:
        """Read a line from the WebSocket client. Blocks until a complete line is available."""
        if self._closed:
            return ""

        # Wait for connection first
        if not self._connected.is_set():
            logger.info("PDB waiting for client connection...")
            if not self._connected.wait(timeout=self._connection_timeout):
                logger.warning("No client connected, returning empty line")
                return ""

        # Accumulate input until we get a newline
        while "\n" not in self._line_buffer:
            try:
                data = self.input_queue.get(timeout=1.0)
                if data is None:  # Signals close
                    self._closed = True
                    return ""
                self._line_buffer += data
            except queue.Empty:
                if self._closed:
                    return ""
                continue

        # Extract the first complete line
        newline_pos = self._line_buffer.index("\n")
        line = self._line_buffer[: newline_pos + 1]
        self._line_buffer = self._line_buffer[newline_pos + 1 :]
        return line

    def flush(self):
        """Flush output (no-op since we send immediately)."""
        pass

    def isatty(self) -> bool:
        """Return True to make PDB think we're a terminal."""
        return True

    def fileno(self) -> int:
        """Return -1 since we don't have a real file descriptor."""
        raise OSError("WebSocketIO does not have a file descriptor")

    def close(self):
        """Close the I/O streams."""
        self._closed = True
        self._connected.set()  # Unblock any waiting readline()
        try:
            self.input_queue.put_nowait(None)  # Signal EOF
        except queue.Full:
            pass

    @property
    def closed(self) -> bool:
        return self._closed


async def _handle_websocket_client(websocket, ws_io: WebSocketIO):
    """Handle a WebSocket client connection."""
    global _websocket_connection

    logger.info(f"PDB client connected from {websocket.remote_address}")
    _websocket_connection = websocket
    ws_io.set_connected()

    async def send_output():
        """Send output from PDB to the WebSocket client."""
        try:
            while not ws_io.closed:
                try:
                    data = ws_io.output_queue.get_nowait()
                    await websocket.send(data)
                except queue.Empty:
                    await asyncio.sleep(0.01)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed while sending")
        except Exception as e:
            logger.debug(f"Send loop error: {e}")

    async def receive_input():
        """Receive input from the WebSocket client and queue it for PDB."""
        try:
            async for message in websocket:
                ws_io.input_queue.put(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by client")
        except Exception as e:
            logger.debug(f"Receive loop error: {e}")
        finally:
            ws_io.close()

    # Run send and receive concurrently
    send_task = asyncio.create_task(send_output())
    try:
        await receive_input()
    finally:
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass

    logger.info("PDB WebSocket session ended")


def _run_websocket_server(port: int, ws_io: WebSocketIO, server_ready: threading.Event):
    """Run the WebSocket server in a background thread."""
    global _websocket_server, _event_loop

    async def run():
        global _websocket_server
        try:
            _websocket_server = await websockets.serve(
                lambda ws: _handle_websocket_client(ws, ws_io),
                "",
                port,
            )
            logger.info(f"PDB WebSocket server listening on port {port}")
            server_ready.set()

            # Keep running until closed
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        except OSError as e:
            logger.error(f"Failed to start WebSocket server on port {port}: {e}")
            server_ready.set()
            raise
        finally:
            if _websocket_server:
                _websocket_server.close()
                await _websocket_server.wait_closed()
                _websocket_server = None

    _event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_event_loop)
    try:
        _event_loop.run_until_complete(run())
    except Exception as e:
        logger.error(f"WebSocket server error: {e}")
    finally:
        _event_loop.close()
        _event_loop = None


def start_debugger(port: int, timeout: int = 300):
    """Start the remote PDB debugger.

    This function:
    1. Starts a WebSocket server in a background thread
    2. Redirects stdin/stdout to WebSocket-backed streams
    3. Calls pdb.set_trace() on the CALLER's frame

    The caller should call this from their breakpoint location.
    PDB will then run normally, with I/O going through the WebSocket.

    Args:
        port: Port for the WebSocket server
        timeout: How long to wait for a client to connect (seconds)
    """
    global _server_thread

    # Create the I/O bridge with the connection timeout
    ws_io = WebSocketIO(connection_timeout=timeout)

    # Start WebSocket server in background thread
    server_ready = threading.Event()
    _server_thread = threading.Thread(
        target=_run_websocket_server,
        args=(port, ws_io, server_ready),
        daemon=True,
    )
    _server_thread.start()

    # Wait for server to be ready
    if not server_ready.wait(timeout=5):
        logger.error("Failed to start WebSocket server")
        return

    # Print connection instructions
    pod_name = os.environ.get("POD_NAME", "unknown")
    pod_namespace = os.environ.get("POD_NAMESPACE", "default")
    pod_ip = os.environ.get("POD_IP", "")

    print("=" * 60)
    print("REMOTE DEBUGGER ACTIVE")
    print("=" * 60)
    print(f"Waiting for debugger client to connect on port {port}...")
    print()
    print("To connect, run:")
    if pod_ip:
        print(f"  kt debug {pod_name} --port {port} --namespace {pod_namespace} --mode pdb --pod-ip {pod_ip}")
    else:
        print(f"  kt debug {pod_name} --port {port} --namespace {pod_namespace} --mode pdb")
    print("=" * 60)
    sys.stdout.flush()

    # Get the caller's frame (skip this function and deep_breakpoint)
    frame = sys._getframe(2)

    # Create PDB instance with our WebSocket I/O
    debugger = pdb.Pdb(stdin=ws_io, stdout=ws_io)

    # We need to skip our own frames when tracing. The debugger should only
    # stop in user code, not in start_debugger() or deep_breakpoint().
    # We do this by setting stopframe to the user's frame initially.
    debugger.reset()

    # Set trace function on all frames from user's frame down
    f = frame
    while f:
        f.f_trace = debugger.trace_dispatch
        debugger.botframe = f
        f = f.f_back

    # Set stopframe to user's frame so we stop there, not in our wrapper code
    debugger.stopframe = frame
    debugger.returnframe = None
    debugger.quitting = False
    debugger.stoplineno = 0

    # Install the trace function
    sys.settrace(debugger.trace_dispatch)

    # Note: We do NOT call set_step() because that would make us stop at ANY
    # frame. Instead, we've set stopframe = frame so we only stop in user's code.
    #
    # When this function returns, execution goes back through deep_breakpoint()
    # and then to the user's code. When the user's next line executes,
    # dispatch_line() will be called, stop_here() will return True (because
    # frame == stopframe), and interaction() will be called, blocking for input.


def cleanup():
    """Clean up any running WebSocket server."""
    global _websocket_server, _event_loop, _websocket_connection

    if _event_loop and _websocket_server:
        try:
            _event_loop.call_soon_threadsafe(_event_loop.stop)
        except Exception:
            pass

    _websocket_server = None
    _websocket_connection = None
