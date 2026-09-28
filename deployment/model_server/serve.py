"""Serve a pinned deployment recipe through StarVLA's native WebSocket protocol."""

import argparse
import logging
from pathlib import Path

import yaml

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def serve(config):
    recipe = dict(config)
    checkpoint = Path(recipe["checkpoint_path"]).expanduser().resolve(strict=True)
    recipe["checkpoint_path"] = str(checkpoint)
    wrapper = PolicyServerWrapper(
        ckpt_path=str(checkpoint),
        device=recipe.get("device", "cuda"),
        use_bf16=recipe.get("use_bf16", False),
        unnorm_key=recipe["unnorm_key"],
        config_overrides=recipe.get("config_overrides", []),
        deployment=recipe,
    )
    server = WebsocketPolicyServer(wrapper, host=recipe["host"], port=recipe["port"], metadata=wrapper.metadata)
    logging.info("StarVLA native deployment ready: %s", wrapper.metadata)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
