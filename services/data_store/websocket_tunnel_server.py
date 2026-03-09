import asyncio
import logging
import os

import websockets

WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", "8080"))
RSYNC_HOST = "127.0.0.1"
RSYNC_PORT = 873

# Buffer size for reading from rsync (larger = fewer syscalls, more memory)
READ_BUFFER_SIZE = int(os.getenv("READ_BUFFER_SIZE", "65536"))

# Max websocket message size (None = unlimited, needed for large rsync transfers)
MAX_MESSAGE_SIZE = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


async def handle_connection(websocket):
    """Handles a new WebSocket connection."""
    client_addr = websocket.remote_address
    logging.info(f"WebSocket connection received from {client_addr}")

    try:
        # Connect to local rsync daemon
        rsync_reader, rsync_writer = await asyncio.open_connection(
            RSYNC_HOST, RSYNC_PORT
        )
        logging.info(f"Connected to rsync daemon at {RSYNC_HOST}:{RSYNC_PORT}")

        # Shuttle data bidirectionally
        async def ws_to_rsync():
            async for message in websocket:
                if isinstance(message, bytes):
                    rsync_writer.write(message)
                    await rsync_writer.drain()

        async def rsync_to_ws():
            while True:
                data = await rsync_reader.read(READ_BUFFER_SIZE)
                if not data:
                    break
                await websocket.send(data)

        # Run both directions concurrently
        await asyncio.gather(ws_to_rsync(), rsync_to_ws())

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        logging.info(f"Closing connection from {client_addr}")
        try:
            rsync_writer.close()
            await rsync_writer.wait_closed()
        except:
            pass


async def main():
    logging.info(f"Starting WebSocket proxy server on port {WEBSOCKET_PORT}...")
    try:
        # Configure websocket server for high connection counts:
        # - max_size=None: No limit on message size (rsync transfers can be large)
        # - compression=None: Disable compression (raw binary data, saves CPU)
        # - ping_interval/ping_timeout: Keep connections alive
        async with websockets.serve(
            handle_connection,
            "0.0.0.0",
            WEBSOCKET_PORT,
            max_size=MAX_MESSAGE_SIZE,
            compression=None,
            ping_interval=30,
            ping_timeout=10,
        ):
            logging.info(
                f"WebSocket proxy server listening on 0.0.0.0:{WEBSOCKET_PORT}"
            )
            await asyncio.Future()  # Run forever
    except Exception as e:
        logging.error(f"Failed to start WebSocket proxy server: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("WebSocket proxy server shutting down")
    except Exception as e:
        logging.error(f"WebSocket proxy server error: {e}")
        raise
