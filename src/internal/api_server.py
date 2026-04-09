import os
import time
import logging
import traceback
import threading
from queue import Queue, Empty
from typing import Optional
from datetime import datetime, UTC

import torch.distributed as dist

from src.schemas import Payload
from src.exceptions import CriticalError, ValidationError
from src.internal.message_processor import MessageProcessor

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn


class ApiServer:
    def __init__(
        self,
        message_processor: MessageProcessor,
        rank: int = 0,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.message_processor = message_processor
        self.rank = rank
        self.host = host
        self.port = port

        self._request_queue: Queue = Queue()
        self._shutdown = False

        if self.rank == 0:
            self._app = self._create_app()

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            self._shutdown = True

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/generate")
        async def generate(payload: Payload):
            result_future: Queue = Queue()
            self._request_queue.put((payload, result_future))
            result = result_future.get()

            if isinstance(result, Exception):
                raise HTTPException(status_code=500, detail=str(result))

            return result

        return app

    def _start_server(self):
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        server.run()

    def run(self) -> None:
        if self.rank == 0:
            server_thread = threading.Thread(target=self._start_server, daemon=True)
            server_thread.start()
            logging.info(f"API server started on {self.host}:{self.port}")

        is_idle = False
        idle_start_time = None

        while not self._shutdown:
            payload_container = [None]
            result_future_container = [None]

            if self.rank == 0:
                try:
                    payload, result_future = self._request_queue.get(timeout=1.0)
                    payload_container = [payload]
                    result_future_container = [result_future]
                except Empty:
                    if not is_idle:
                        idle_start_time = datetime.now(UTC)
                        is_idle = True

            if dist.is_initialized():
                dist.broadcast_object_list(payload_container, 0)

            payload = payload_container[0]
            result_future = result_future_container[0]

            if payload is not None:

                try:
                    latency = self.message_processor(payload, None)
                    if self.rank == 0 and result_future is not None:
                        result_future.put({
                            "job_set_id": payload.job_set_id,
                            "status": "completed",
                            "latency": latency
                        })
                except CriticalError as e:
                    if self.rank == 0:
                        logging.error(f"Critical error: {e}\n{traceback.format_exc()}")
                        if result_future is not None:
                            result_future.put(e)

                    time.sleep(5)
                    if dist.is_initialized():
                        dist.barrier()
                        dist.destroy_process_group()
                    os._exit(1)

                except Exception as e:
                    if self.rank == 0:
                        logging.error(f"Error processing request: {e}\n{traceback.format_exc()}")
                        if result_future is not None:
                            result_future.put(e)

            if dist.is_initialized():
                dist.barrier()

        logging.info("API server shutting down...")