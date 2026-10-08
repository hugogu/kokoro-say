# Speak on a Mac for an assistant on another machine

`ksay` speaks through the machine it runs on. To let an assistant on another machine talk
to you through your Mac, such as an agent like [OpenClaw](https://docs.openclaw.ai) on a
Linux server, run `ksay` on the Mac as an HTTP server and point the assistant at it.

```text
OpenClaw on Linux ──HTTP, token──►  ksay --mcp --listen, on the Mac  ──►  Mac speakers
 calls speak("…")                   Kokoro-82M, offline
```

The assistant does the thinking and `ksay` does the speaking. `ksay` does not listen: the
voice that goes the other way has to come from somewhere else.

## On the Mac

Install `ksay` with the `mcp` extra, and fetch the model now, so that the first spoken
answer is not slow. For Chinese voices add the `zh` extra as well, which needs Python
3.12 or older: `uv tool install --force --python 3.12 'kokoro-say[mcp,zh] @ git+...'`.

```sh
uv tool install --force 'kokoro-say[mcp] @ git+https://github.com/hugogu/kokoro-say'
ksay --list-voices
```

Make a token, a long random password that the assistant has to send:

```sh
mkdir -p ~/.config/ksay
openssl rand -hex 24 > ~/.config/ksay/token
chmod 600 ~/.config/ksay/token
```

Start the server:

```sh
ksay --mcp --listen 0.0.0.0:8765 --token-file ~/.config/ksay/token
```

It says `ksay: serving MCP at http://0.0.0.0:8765/mcp, with a token`. `0.0.0.0` is every
network interface of the Mac. To listen on one address only, give that one, such as
`--listen 192.168.1.10:8765`; to listen on this Mac alone, leave the host out
(`--listen 8765`), which is what you want behind an SSH tunnel. Any host but this Mac's
own needs a token, and `ksay` refuses to start without one.

If the macOS firewall is on, it may ask whether to let Python accept incoming connections
the first time. Allow it. The Mac has to be awake and your output device on: `ksay`
speaks through whatever the Mac's default output is at that moment.

### Start it at login

A LaunchAgent starts the server when you log in, and starts it again if it stops. Save
this as `~/Library/LaunchAgents/local.ksay-mcp.plist`, with your own home folder where it
says `/Users/YOU` (launchd does not expand `~`; `which ksay` gives the first path):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>local.ksay-mcp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOU/.local/bin/ksay</string>
        <string>--mcp</string>
        <string>--listen</string>
        <string>0.0.0.0:8765</string>
        <string>--token-file</string>
        <string>/Users/YOU/.config/ksay/token</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/ksay-mcp.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/ksay-mcp.log</string>
</dict>
</plist>
```

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.ksay-mcp.plist   # start, and at every login
launchctl kickstart -k gui/$(id -u)/local.ksay-mcp                             # restart
launchctl bootout gui/$(id -u)/local.ksay-mcp                                  # stop
```

The log has what `ksay` writes to standard error: the line above, and the reason when a
call fails.

## On the other machine

First check that the Mac can be reached, with the token:

```sh
TOKEN=...                                  # what is in ~/.config/ksay/token on the Mac
curl -s -o /dev/null -w '%{http_code}\n' http://192.168.1.10:8765/mcp                                  # 401
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $TOKEN" http://192.168.1.10:8765/mcp  # 400
```

`401` is the server saying that it wants the token, so it is there. `400` with the token
means that the token is right and that what you sent was not an MCP request. Anything
else is the network: see [Troubleshooting](#troubleshooting).

### OpenClaw

OpenClaw's [documentation](https://docs.openclaw.ai/tools/mcp) keeps MCP servers under
`mcp.servers` in its configuration. `ksay` serves Streamable HTTP, and **`transport`
has to say so**: for an entry with a `url` and no `transport`, OpenClaw uses SSE, which
`ksay` does not serve.

```json
{
  "mcp": {
    "servers": {
      "ksay": {
        "url": "http://192.168.1.10:8765/mcp",
        "transport": "streamable-http",
        "headers": { "Authorization": "Bearer <the token>" },
        "connectionTimeoutMs": 10000,
        "requestTimeoutMs": 120000
      }
    }
  }
}
```

The same as a command, then a check that it connects and lists the tools:

```sh
openclaw mcp set ksay '{"url":"http://192.168.1.10:8765/mcp","transport":"streamable-http","headers":{"Authorization":"Bearer <the token>"},"requestTimeoutMs":120000}'
openclaw mcp doctor ksay --probe
```

OpenClaw's documentation advises against keeping a bearer token in a file that is
committed, and `openclaw mcp doctor` warns about one that is written out literally in
`headers`; use the way to keep secrets that your installation has. As far as that
documentation says, the gateway holds the connections to MCP servers, so the machine that
runs the gateway is the one that has to reach the Mac.

`speak` returns when the speech is over, which takes as long as reading the text aloud,
so give `requestTimeoutMs` more than the longest speech you expect, and keep spoken
answers short.

### What to tell the agent

The agent sees two tools of the server `ksay`: `speak` and `list_voices`. It needs to
know when to use them. For example, in its instructions:

> To answer by voice, call the `ksay` tool `speak` with one to three short sentences. Use
> the voice `af_heart` for English and `zf_xiaobei` for Chinese; `list_voices` has the
> others. `speak` returns when the speech has finished, so do not call it again before it
> has. Do not read out code, links or long lists.

A voice reads its own language only, which is why the voice follows the language. Calls
take turns: a second `speak` waits for the first, so two speeches asked at once are heard
one after the other.

## How the Mac is reached

`ksay` speaks HTTP, not HTTPS: the token and the words to be said can be read by anyone
who can see the traffic. Pick the way that fits the network between the two machines.

| The two machines are | Use |
| --- | --- |
| on a network you trust, such as your home | `--listen 0.0.0.0:8765` with a token, as above |
| joined by a VPN such as Tailscale or WireGuard | The same, with the address that the VPN gives the Mac; the VPN encrypts it |
| on a network you do not trust, or the Mac has no address that the other machine can reach | An SSH tunnel, below |

### An SSH tunnel

`ksay` stays on the Mac alone (`--listen 8765`, still with a token, as other users of the
Mac can reach it too) and SSH carries the traffic, so nothing else listens and everything
is encrypted.

If the other machine can SSH into the Mac (Remote Login is on), it forwards its own port
8765 to the Mac's:

```sh
ssh -N -L 8765:127.0.0.1:8765 you@mac.local
```

If only the Mac can reach the other machine, the Mac pushes its server there instead:

```sh
ssh -N -R 8765:127.0.0.1:8765 you@linux-host
```

Either way the URL in OpenClaw is `http://127.0.0.1:8765/mcp`. A tunnel has to be kept
up; `autossh`, or a systemd unit on Linux, does that.

## What the token protects

With the token, a caller can make the Mac speak and can list the voices, and that is all.
Over HTTP the server does not offer `save_speech`, so a caller cannot choose files to
write on the Mac, and it offers nothing that runs commands or reads files. Without the
token every request is refused with `401`. Two things it does not do: it does not
encrypt (see above), and it does not tell one caller from another: everyone with the token
is the same.

A server that listens on the Mac alone also refuses a request whose `Host` header is
not `127.0.0.1`, `localhost` or `[::1]`, which keeps a web page in your browser from
reaching it through a name that the page points at your Mac (DNS rebinding). A proxy in
front of `ksay` that changes that header gets `421`. Put `ksay` on an address that the
proxy can reach, with a token, and not on the Mac alone.

To change the token, put a new one in the file and restart `ksay`.

## Troubleshooting

| What you see | Why, and what to do |
| --- | --- |
| The connection is refused, or times out | `ksay` is not running, or listens on the Mac alone (`--listen 8765`) while the caller is on another machine, or a firewall is in the way, or the address is wrong. `lsof -nP -iTCP:8765 -sTCP:LISTEN` on the Mac shows who listens, and on which address |
| `401` | The token is missing or is not the one in the file. Check `headers` in OpenClaw against `curl` above |
| `421`, or `Invalid Host header` | A proxy or a name rewrites the `Host` header of a server that listens on the Mac alone. See above |
| OpenClaw connects but finds no tools, or fails at once | `transport` is not `streamable-http`, so OpenClaw is talking SSE |
| A call times out | The speech is longer than `requestTimeoutMs`. Shorten the text, or give it more time |
| The call fails with `cannot open the audio output` | The Mac has no output that `ksay` can open: no device, or a server started where there is no audio. Start it from your own login |
| The call succeeds and nothing is heard | The Mac's output is another device than the one you listen on, or muted. `ksay` follows the default output at the moment of each speech |
| The first call is slow | The model loads on the first call, about a second, and downloads, 354 MB, if it was never fetched: run `ksay --list-voices` on the Mac first |
| Nothing works after a while | The Mac went to sleep. Keep it awake, with `caffeinate -i` or in the Energy settings |

## Limits

- `ksay` only speaks. It does not listen, and has no speech recognition.
- One speech at a time: calls wait for each other, whichever caller they come from.
- There is no encryption and no way to tell callers apart. Use a tunnel or a VPN when the
  network is not yours.
- `save_speech` is for an assistant that you started yourself, over standard input and
  output; it is not offered over HTTP.
