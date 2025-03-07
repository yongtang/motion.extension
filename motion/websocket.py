import asyncio
import websockets


class WebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.server = None
        self.stop_event = asyncio.Event()

    async def ws_handler(self, websocket, path):
        """Handles incoming WebSocket messages."""
        try:
            async for message in websocket:
                print(f"Received: {message}")  # Process message here
                response = f"Echo: {message}"
                await websocket.send(response)
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")

    async def start(self):
        """Starts the WebSocket server asynchronously."""
        self.server = await websockets.serve(self.ws_handler, self.host, self.port)
        print(f"WebSocket server started at ws://{self.host}:{self.port}")
        await self.stop_event.wait()  # Wait until stop() is called
        await self.server.wait_closed()
        print("WebSocket server stopped")

    async def stop(self):
        """Stops the WebSocket server explicitly."""
        print("Stopping WebSocket server...")
        self.stop_event.set()  # Unblock the event loop
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        print("WebSocket server stopped.")


if __name__ == "__main__":

    async def main():
        server = WebSocketServer()
        try:
            await server.start()
        except KeyboardInterrupt:
            print("\n[WebSocketServer] Received KeyboardInterrupt. Shutting down...")
            await server.stop()

    asyncio.run(main())
