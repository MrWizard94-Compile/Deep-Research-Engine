# Ollama API — endpoint used by `llm_bridge.py`

Default base URL: `http://127.0.0.1:11434` (override via `OLLAMA_URL`).
Ollama runs in Docker here as container **`ollama-engine`** (publishes 11434).

## Custom model: `deepseek-r1-highctx`

Built from [`../Modelfile`](../Modelfile):
```
FROM deepseek-r1:8b
PARAMETER num_ctx 32768
PARAMETER temperature 0.2
```
The bridge defaults `OLLAMA_MODEL=deepseek-r1-highctx`. Create it inside the container
either with the CLI or the API (the API form is the most version-robust):
```
curl http://127.0.0.1:11434/api/create -d '{"model":"deepseek-r1-highctx","from":"deepseek-r1:8b","parameters":{"num_ctx":32768,"temperature":0.2},"stream":false}'
```

## Hardware note (this machine: GTX 1660 Ti, 6 GB VRAM)

A 14B model (~9 GB weights) cannot fit in 6 GB VRAM and is forced ~73% onto the CPU
(~2.4 tok/s — minutes per reasoning call). **deepseek-r1:8b (~5 GB) mostly fits the GPU
and is the chosen base.** Tune via env:
- `OLLAMA_MODEL`, `OLLAMA_URL` — model + endpoint.
- `OLLAMA_TIMEOUT` — per-request timeout in seconds (default 1800; reasoning models are slow).

## POST /api/generate (single-prompt completion)

Request body:
```json
{
  "model": "deepseek-r1-highctx",
  "prompt": "…",
  "stream": false,
  "options": { "num_ctx": 32768, "temperature": 0.2 }
}
```

Response body (with `stream: false`): a single JSON object whose `response` field holds
the generated text.

### Notes
- A `404 Not Found` from `/api/generate` almost always means the named **model is not
  pulled** locally (the endpoint itself is reachable). Pull it with:
  `ollama pull deepseek-r1-highctx` — or point `OLLAMA_MODEL` at an installed model.
- DeepSeek-R1 emits chain-of-thought wrapped in `<think>…</think>`. The bridge strips
  `<think>` / `<thinking>` / `<thought>` blocks before JSON parsing.
- `options.num_ctx` sets the context window for the request; `temperature` 0.2 keeps
  structured-JSON output stable.

## Container

Generated experiment code runs in the `polyglot-research-env` image (see `Dockerfile`):
`docker build -t polyglot-research-env .`
