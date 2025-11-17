import os, json, re, asyncio
from typing import List, Dict, Any
from pathlib import Path
from openai import AsyncOpenAI
from asyncio import Semaphore

MODEL = "gpt-5-mini"
MAX_CONCURRENCY = 16          # global cap (tune to your rate limits)
PER_TRACE_BOUNDARY_FANOUT = 8 # per-trace cap (optional, safe default)
TIMEOUT_S = 1200

client = AsyncOpenAI(api_key="")
gate = Semaphore(MAX_CONCURRENCY)

# ---------- I/O (once) ----------
read = lambda p: Path(p).read_text(encoding="utf-8")
system_prompt = read("abstraction_system_prompt.txt")
user_prompt_template = read("abstraction_user_prompt.txt")
boundary_system_prompt = read("boundary_prediction_system_prompt.txt")
boundary_user_prompt_template = read("boundary_prediction_user_prompt.txt")

records = json.loads(Path("/scratch/rst306/verifier_scaling/verifiers/result/aime/Qwen_Qwen3-8B/20251010_115527/record.json").read_text("utf-8"))
traces = [r["solver_full_output"] for r in records if r["solver_correct"]][:20]

# ---------- helpers ----------
def find_subsequence(text: str, search_sequence: str, wildcard: str = "...") -> List[Dict]:
    parts = search_sequence.split(wildcard)
    pattern = ".*?".join(re.escape(p) for p in parts)
    regex = re.compile(pattern, re.DOTALL)
    return [{"match": m.group(0), "start": m.start(), "end": m.end()} for m in regex.finditer(text)]

# Keep these prefixes identical across calls to hit prompt-caching.
ABSTRACTIONS_INSTR = (
    "Return JSON: {\"abstractions\": [{\"name\": str, \"trigger\": str, \"procedure\": [str]}]}.\n"
)
BOUNDARIES_INSTR = (
    "Return JSON: {\"boundaries\": [{\"boundary\": str}]}.\n"
)

def abstraction_user(trace: str) -> str:
    # Only the dynamic tail (trace) varies.
    return user_prompt_template.replace("REASONING_TRACES", trace)

def boundary_user(abstraction_block: str, trace: str) -> str:
    # Template stays identical; only {{ABSTRACTION}} and {{RAW_TEXT}} vary.
    return boundary_user_prompt_template.replace("{{ABSTRACTION}}", abstraction_block).replace("{{RAW_TEXT}}", trace)

def abstraction_block_from(a: Dict[str, Any]) -> str:
    # Stable formatting helps caching + model reliability.
    proc = "\n".join(a["procedure"])
    return (
        f"Abstraction Name: {a['name']}\n\n"
        f"Abstraction Trigger: {a['trigger']}\n\n"
        f"Abstraction Deliverables:\n{proc}\n"
    )

# ---------- API wrappers ----------
async def call_json(instructions: str, user_input: str) -> Dict[str, Any]:
    async with gate:
        resp = await client.responses.create(
            model=MODEL,
            instructions=instructions,         # keep as stable, long prefix
            input=user_input,                  # small varying tail
            # response_format={"type": "json_object"},
            timeout=TIMEOUT_S,
        )
    return json.loads(resp.output_text)

# ---------- pipeline ----------
async def get_abstractions_for_trace(trace: str) -> List[Dict[str, Any]]:
    data = await call_json(
        instructions=system_prompt,
        user_input=abstraction_user(trace),
    )
    return data["abstractions"]

async def get_boundaries_for_abstraction(abstraction: Dict[str, Any], trace: str) -> Dict[str, Any]:
    block = abstraction_block_from(abstraction)
    data = await call_json(
        instructions=boundary_system_prompt,
        user_input=boundary_user(block, trace),
    )
    # post-process: fuzzy match on the raw trace (local CPU, cheap)
    hits = []
    for b in data["boundaries"]:
        s = b["boundary"].replace("«", "").replace("»", "")
        hits.extend(find_subsequence(trace, s))
    abstraction["boundaries"] = data["boundaries"]
    abstraction["matched_subsequences"] = hits
    return abstraction

async def process_trace(trace: str) -> Dict[str, Any]:
    abstractions = await get_abstractions_for_trace(trace)
    # Per-trace fan-out with its own cap to avoid burst spikes
    sem = Semaphore(PER_TRACE_BOUNDARY_FANOUT)
    async def guarded(a):
        async with sem:
            return await get_boundaries_for_abstraction(a, trace)
    enriched = await asyncio.gather(*[guarded(a) for a in abstractions])
    return {"solver_output": trace, "abstractions": enriched}

async def main():
    results = await asyncio.gather(*(process_trace(t) for t in traces))
    out_dir = Path("abstraction_results_async/Qwen/Qwen3-1.7B")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "abstractions.json").write_text(json.dumps(results, ensure_ascii=False), "utf-8")

if __name__ == "__main__":
    asyncio.run(main())
