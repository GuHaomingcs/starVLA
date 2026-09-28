# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Configured deployments implement per-connection reset and sampler negotiation.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        from deployment.model_server.sampling_session import SamplingSession

        contract = self._metadata.get("serving_contract")
        session = SamplingSession(contract) if contract is not None else None

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                ret = self._route_message(msg, session)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict, session=None) -> dict:
        """Native RPC dispatch; configured sessions require an explicit reset."""
        req_id = msg.get("request_id", "default")
        kind = msg.get("type", "infer")
        response_type = "inference_result" if kind in {"infer", "predict_action"} else kind
        if kind not in {"infer", "predict_action", "ping", "reset"}:
            response_type = "unknown"
        response = {"request_id": req_id, "type": response_type, "ok": True, "status": "ok"}
        try:
            if kind == "ping":
                return response
            if kind == "reset":
                if session is None:
                    raise ValueError("This deployment has no session/reset contract")
                session.reset(msg.get("session_id"))
                return {**response, "session_id": session.session_id}
            if kind not in {"infer", "predict_action"}:
                raise ValueError(f"Unsupported message type {kind!r}")
            payload = msg.get("payload", msg)
            if not isinstance(payload, dict):
                raise ValueError("Payload must be a dict")
            sampling = payload.get("sampling", {"mode": "default"})
            if session is not None:
                if session.session_id is None or msg.get("session_id") != session.session_id:
                    raise ValueError("Inference session has not been reset or does not match")
                parameters = session.parameters(sampling)
                payload = dict(payload)
                if parameters["mode"] == "default":
                    payload.pop("sampling", None)
                else:
                    payload["sampling"] = parameters
                if payload.get("unnorm_key") != session.contract["unnorm_key"]:
                    raise ValueError("unnorm_key differs from the deployment contract")
            result = self._policy.predict_action(**payload)
            if session is not None:
                result = session.finish(result, sampling)
            return {**response, "data": result}
        except Exception as error:
            logging.exception("Policy request failed (request_id=%s)", req_id)
            return {**response, "ok": False, "status": "error", "error": {"message": str(error)}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
