"""Ask `ksay --mcp` to speak, as an AI assistant would, and print its answer.

    uv run python scripts/mcp_speak.py "Hello from an assistant."

Run it from a checkout with the mcp extra installed: it starts the server with the
interpreter that runs it. It exits 0 when the server spoke, and 1 when the server
answered with an error, which it prints. CI uses it to play through a fake sound card
and to see that a machine without audio gets a message, not a crash. It is also the
quickest way to hear that playback works on a machine.
"""

import os
import sys

import anyio
from mcp import Client, StdioServerParameters


async def speak(text: str) -> bool:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kokoro_say", "--mcp"],
        env=dict(os.environ),  # the model folder and the home of the sound settings
    )
    async with Client(server) as client:
        result = await client.call_tool("speak", {"text": text})
    print("\n".join(block.text for block in result.content))
    return not result.is_error


if __name__ == "__main__":
    spoken = anyio.run(speak, " ".join(sys.argv[1:]) or "Hello from an assistant.")
    sys.exit(0 if spoken else 1)
