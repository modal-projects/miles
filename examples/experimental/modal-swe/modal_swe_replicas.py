"""Discover direct upstream replicas behind the targeted Stitch router."""

from __future__ import annotations

import argparse
import asyncio
import json

from modal._server import _Server
from modal.client import _Client
from modal_proto import api_pb2


async def discover(app_name: str, server_name: str, environment: str) -> dict:
    """Return the service-function ID and every live Flash container."""
    client = await _Client.from_env()
    server = _Server.from_name(
        app_name,
        server_name,
        environment_name=environment,
    )
    service_function = server._get_service_function()
    await service_function.hydrate(client=client)
    response = await client.stub.FlashContainerList(
        api_pb2.FlashContainerListRequest(
            function_id=service_function.object_id,
        )
    )
    return {
        "function_id": service_function.object_id,
        "containers": [
            {
                "host": container.host,
                "task_id": getattr(container, "task_id", ""),
            }
            for container in response.containers
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--environment", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(discover(args.app, args.server, args.environment)),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
