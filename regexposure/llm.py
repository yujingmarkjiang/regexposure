"""All OpenAI API access. Every processing step that reads text runs through these endpoints.

    Step                                   Endpoint                Model
    extract  (locate + clean Item 1/1A)    POST /v1/responses      GPT-5.6 Terra
    domains  (name the 20 clusters)        POST /v1/responses      GPT-5.6 Terra
    score    (rate rules and sections)     POST /v1/responses      GPT-5.6 Terra
    embed    (rules and 10-K sections)     POST /v1/embeddings     text-embedding-3-large
    Batch mode (openai.use_batch: true): POST /v1/files (upload JSONL, purpose "batch"),
    POST /v1/batches (create), GET /v1/batches/{id} (poll), GET /v1/files/{id}/content (results).

The API key is read from the file named in config.yaml (default: OpenAI.key in the
repository root), falling back to the OPENAI_API_KEY environment variable.
Every request is logged with its endpoint and the identifiers OpenAI returns (response id
or batch id), so each stored number can be matched to the API call that produced it.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
from openai import OpenAI

from .textutil import JsonlStore, n_tokens, token_chunks

RESPONSES = "/v1/responses"
EMBEDDINGS = "/v1/embeddings"

# which endpoint each step uses (printed by `python run.py endpoints`, documented in the README and appendix)
STEP_ENDPOINTS = {
    "extract": (RESPONSES, "model"),
    "domains": (RESPONSES, "model"),
    "score": (RESPONSES, "model"),
    "embed": (EMBEDDINGS, "embedding_model"),
}
BATCH_ENDPOINTS = ["POST /v1/files", "POST /v1/batches", "GET /v1/batches/{id}", "GET /v1/files/{id}/content"]


def describe_endpoints(cfg) -> str:
    rows = [f"  {step:<8} POST {ep:<16} {getattr(cfg.openai, key)}" for step, (ep, key) in STEP_ENDPOINTS.items()]
    mode = "Batch API: " + ", ".join(BATCH_ENDPOINTS) if cfg.openai.use_batch else "synchronous requests"
    return "OpenAI API endpoints by step:\n" + "\n".join(rows) + f"\n  mode: {mode}"


# ------------------------------------------------------------------------------- client
def read_key(cfg) -> str:
    path = cfg.root / cfg.openai.key_file
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if "=" in key and key.split("=", 1)[0].strip().upper() == "OPENAI_API_KEY":
            key = key.split("=", 1)[1].strip().strip('"').strip("'")
        if key:
            return key
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise FileNotFoundError(f"No API key: create {path} (one line with the key) or set OPENAI_API_KEY.")


def get_client(cfg) -> OpenAI:
    return OpenAI(api_key=read_key(cfg), max_retries=cfg.openai.max_retries, timeout=cfg.openai.timeout_s)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------- structured requests
def json_request(cfg, step: str, instructions: str, user_input: str, schema: dict, schema_name: str) -> dict:
    """Body for one Responses API call that must return JSON matching `schema`."""
    body = {
        "model": cfg.openai.model,
        "instructions": instructions,
        "input": user_input,
        "reasoning": {"effort": getattr(cfg.openai.reasoning_effort, step)},
        "text": {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
        "max_output_tokens": cfg.openai.max_output_tokens,
        "store": cfg.openai.store,
    }
    return body


def output_text(body: dict) -> str:
    """Concatenate the assistant text of a Responses API result (SDK dump or Batch body)."""
    if body.get("status") and body["status"] != "completed":
        raise RuntimeError(f"response status={body['status']} details={body.get('incomplete_details')}")
    parts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "refusal":
                raise RuntimeError("model refusal: " + str(c.get("refusal")))
    if not parts:
        raise RuntimeError("no output_text in response")
    return "".join(parts)


def run_json_requests(cfg, client, requests: dict, store: JsonlStore, parse, meta: dict | None = None):
    """Run {custom_id: body} requests, skipping ids already stored. `parse(dict)->dict` validates output.

    Each stored record holds custom_id, ok, the parsed result or error, token usage, model and time,
    so every number can be traced back to the request that produced it.
    """
    done = store.done_ids()
    todo = {k: v for k, v in requests.items() if k not in done}
    if not todo:
        return
    meta = meta or {}

    def record(cid, body_out=None, error=None, batch_id=None):
        rec = {"custom_id": cid, "endpoint": RESPONSES, "model": cfg.openai.model, "time": now(), **meta.get(cid, {})}
        if batch_id:
            rec["batch_id"] = batch_id
        if body_out:
            rec["response_id"] = body_out.get("id")
        try:
            if error:
                raise RuntimeError(error)
            rec["result"] = parse(json.loads(output_text(body_out)))
            rec["usage"] = body_out.get("usage")
            rec["ok"] = True
        except Exception as e:  # stored, retried on the next run
            rec["ok"] = False
            rec["error"] = str(e)[:500]
        store.append(rec)

    if cfg.openai.use_batch:
        for cid, res in run_batch(cfg, client, RESPONSES, todo, tag=store.path.stem).items():
            resp, err = res.get("response") or {}, res.get("error")
            if err or resp.get("status_code", 200) != 200:
                record(cid, error=str(err or resp.get("body")), batch_id=res.get("batch_id"))
            else:
                record(cid, body_out=resp["body"], batch_id=res.get("batch_id"))
        return

    def one(cid, body):
        try:
            return cid, client.responses.create(**body).model_dump(), None
        except Exception as e:
            return cid, None, str(e)

    with ThreadPoolExecutor(cfg.openai.max_workers) as ex:
        futs = [ex.submit(one, cid, b) for cid, b in todo.items()]
        for i, f in enumerate(as_completed(futs), 1):
            cid, out, err = f.result()
            record(cid, body_out=out, error=err)
            if i % 200 == 0:
                print(f"  {i:,}/{len(todo):,} requests")


# ------------------------------------------------------------------------------- Batch API
def run_batch(cfg, client, endpoint: str, requests: dict, tag: str) -> dict:
    """Submit requests through the Batch API, wait, and return {custom_id: result line}.

    Submitted batch ids are saved in data/logs/batches/<tag>.json, so an interrupted run
    resumes by polling the same batches instead of paying for them twice.
    """
    state_path = cfg.paths.batches / f"{tag}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    submitted = {cid for b in state["batches"] for cid in b["ids"]}
    pending = [(cid, body) for cid, body in requests.items() if cid not in submitted]
    # split into batch files that respect the Batch API limits (request count and file size)
    files, cur, cur_bytes = [], [], 0
    for cid, body in pending:
        line = json.dumps({"custom_id": cid, "method": "POST", "url": endpoint, "body": body}) + "\n"
        nb = len(line.encode("utf-8"))
        if cur and (len(cur) >= cfg.openai.batch_max_requests or cur_bytes + nb > cfg.openai.batch_max_mb * 1e6):
            files.append(cur)
            cur, cur_bytes = [], 0
        cur.append((cid, line))
        cur_bytes += nb
    if cur:
        files.append(cur)
    for n, chunk in enumerate(files):
        path = cfg.paths.batches / f"{tag}_{int(time.time())}_{n}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(line for _, line in chunk)
        with open(path, "rb") as fh:
            fobj = client.files.create(file=fh, purpose="batch")
        b = client.batches.create(input_file_id=fobj.id, endpoint=endpoint, completion_window="24h")
        state["batches"].append({"id": b.id, "ids": [cid for cid, _ in chunk]})
        state_path.write_text(json.dumps(state))
        path.unlink(missing_ok=True)
        print(f"  submitted batch {b.id} ({len(chunk):,} requests)")

    results = {}
    for rec in state["batches"]:
        while True:
            b = client.batches.retrieve(rec["id"])
            if b.status in ("completed", "failed", "expired", "cancelled"):
                break
            print(f"  batch {b.id}: {b.status} {b.request_counts}")
            time.sleep(cfg.openai.batch_poll_seconds)
        for fid in (b.output_file_id, b.error_file_id):
            if fid:
                for line in client.files.content(fid).text.splitlines():
                    if line.strip():
                        o = json.loads(line)
                        o["batch_id"] = b.id
                        results[o["custom_id"]] = o
        for cid in rec["ids"]:
            results.setdefault(cid, {"error": f"batch {b.id} ended with status {b.status}", "batch_id": b.id})
    state_path.unlink(missing_ok=True)
    return results          # includes results of batches left over from an interrupted run


# ------------------------------------------------------------------------------- embeddings
def embed_documents(cfg, client, docs: list[tuple[str, str]]) -> dict:
    """Embed (doc_id, text) pairs with text-embedding-3-large.

    Texts longer than the model's input limit are split into chunks of at most
    embedding.max_tokens_per_chunk tokens; the document vector is the token-weighted mean of
    its chunk vectors, re-normalised to unit length. Returns {doc_id: float32 vector}.
    """
    E = cfg.embedding
    pieces = []  # (doc_id, chunk_text, n_tokens)
    for doc_id, text in docs:
        for ch in token_chunks(text, E.max_tokens_per_chunk, 0, E.tokenizer):
            pieces.append((doc_id, ch, n_tokens(ch, E.tokenizer)))
    reqs, cur, cur_tok = [], [], 0
    for p in pieces:
        if cur and (len(cur) >= E.max_inputs_per_request or cur_tok + p[2] > E.max_tokens_per_request):
            reqs.append(cur)
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += p[2]
    if cur:
        reqs.append(cur)

    log = JsonlStore(cfg.paths.logs / "embeddings_requests.jsonl")

    def call(batch):
        r = client.embeddings.create(model=cfg.openai.embedding_model, input=[t for _, t, _ in batch],
                                     dimensions=cfg.openai.embedding_dim)
        usage = getattr(r, "usage", None)
        log.append({"endpoint": EMBEDDINGS, "model": cfg.openai.embedding_model, "dimensions": cfg.openai.embedding_dim,
                    "doc_ids": sorted({d for d, _, _ in batch}), "n_inputs": len(batch),
                    "tokens": getattr(usage, "total_tokens", None), "time": now()})
        return batch, [np.asarray(d.embedding, dtype=np.float32) for d in sorted(r.data, key=lambda d: d.index)]

    acc = {}
    with ThreadPoolExecutor(cfg.openai.max_workers) as ex:
        for batch, vecs in ex.map(call, reqs):
            for (doc_id, _, k), v in zip(batch, vecs):
                s, w = acc.get(doc_id, (0.0, 0))
                acc[doc_id] = (s + v * k, w + k)
    out = {}
    for doc_id, (s, w) in acc.items():
        v = s / max(w, 1)
        out[doc_id] = (v / max(np.linalg.norm(v), 1e-12)).astype(np.float32)
    return out
