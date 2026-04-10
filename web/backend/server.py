"""FastAPI backend serving LBF game state over WebSocket."""

import sys
import os
import asyncio
import json

# Ensure imports work the same way as the Tkinter app
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_web_dir = os.path.dirname(_backend_dir)
_project_root = os.path.dirname(_web_dir)
_addon_dir = os.path.join(_project_root, "tr_lbf_addon")

for p in (_addon_dir, _project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from tr_lbf_addon.game_runner import GameRunner, default_params
from web.backend.serializer import serialize_frame
from web.backend.metrics_serializer import serialize_metrics_snapshot

app = FastAPI()

# Serve frontend static files if built
_static_dir = os.path.join(_web_dir, "frontend", "dist")
if os.path.isdir(_static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(_static_dir, "index.html"))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    runner = GameRunner(default_params())
    runner.reset()
    paused = True
    speed_ms = 200
    loop_task: asyncio.Task | None = None

    async def send_frame():
        await websocket.send_json(serialize_frame(runner, paused))

    async def send_metrics_snapshot():
        await websocket.send_json(serialize_metrics_snapshot(runner.metrics))

    async def game_loop():
        while not paused:
            await asyncio.to_thread(runner.step)
            await send_frame()
            if runner.episode_over:
                runner.evolve()
                runner.reset()
                await send_frame()
            await asyncio.sleep(speed_ms / 1000)

    def stop_loop():
        nonlocal loop_task
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
            loop_task = None

    # Send initial state
    await send_frame()
    await send_metrics_snapshot()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "play":
                if paused:
                    paused = False
                    stop_loop()
                    loop_task = asyncio.create_task(game_loop())

            elif msg_type == "pause":
                paused = True
                stop_loop()
                await send_frame()

            elif msg_type == "step":
                paused = True
                stop_loop()
                await asyncio.to_thread(runner.step)
                if runner.episode_over:
                    await send_frame()
                    runner.evolve()
                    runner.reset()
                await send_frame()

            elif msg_type == "set_speed":
                speed_ms = max(20, min(2000, int(msg.get("speed_ms", 200))))

            elif msg_type == "request_metrics_snapshot":
                await send_metrics_snapshot()

            elif msg_type == "apply_params":
                paused = True
                stop_loop()
                new_params = msg.get("params", {})
                # Ensure int types for numeric params
                int_keys = (
                    "field_size", "number_players", "max_num_food",
                    "max_episode_steps", "sight",
                    "min_player_level", "max_player_level",
                    "min_food_level", "max_food_level",
                )
                for k in int_keys:
                    if k in new_params:
                        new_params[k] = int(new_params[k])
                merged = {**default_params(), **new_params}
                await asyncio.to_thread(runner.rebuild, merged)
                await websocket.send_json({"type": "metrics_cleared"})
                await send_metrics_snapshot()
                await send_frame()

    except WebSocketDisconnect:
        stop_loop()
    except asyncio.CancelledError:
        stop_loop()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.backend.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[_addon_dir, _backend_dir],
    )
