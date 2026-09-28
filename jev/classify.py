"""Classify every review in the thesis dataset with Jev.

    export TYPESAFE_API_KEY=sk-...
    python classify.py --limit 20      # try a few first, check cost and output
    python classify.py                 # all 1,107 reviews (~2k tokens each, roughly $0.10)
    python classify.py --mock          # offline dry run with fake answers, no key needed

Results are appended to predictions.jsonl, one line per review, so an interrupted run
resumes where it stopped and you never pay twice for the same review.
"""

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from questions import ASPECTS, MODEL, build_questions

HERE = Path(__file__).parent
DATA = HERE.parent / "data"


def load_reviews() -> pd.DataFrame:
    """Join the aspect labels (ac_dataset) and sentiment labels (sa_dataset) by row id."""
    ac = pd.read_csv(DATA / "ac_dataset.csv", index_col=0)
    sa = pd.read_csv(DATA / "sa_dataset.csv", index_col=0)
    assert (ac["Reviews"] == sa["Reviews"]).all(), "ac and sa datasets are out of sync"
    df = pd.DataFrame({"id": ac.index, "review": ac["Reviews"]})
    for label, key in ASPECTS.items():
        df[f"gold_mention_{key}"] = ac[label].astype(bool)
        df[f"gold_sentiment_{key}"] = sa[label].where(sa[label].isin(["positive", "negative"]))
    return df


def answers_to_row(answers) -> dict:
    """Flatten the SDK's typed answers into plain JSON."""
    out = {}
    for name, a in answers.items():
        if a.type == "noul":
            out[name] = {"p_yes": a.noul}
        else:
            out[name] = {"choice": a.choice, "confidence": a.confidence, "probabilities": a.probabilities}
    return out


async def classify_one(client, sem, review_id, text, questions, model):
    async with sem:
        started = time.perf_counter()
        response = await client.system_one(state={"review": text}, questions=questions, model=model)
        return {
            "id": int(review_id),
            "model": response.model,  # the exact version that answered; log it
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "input_tokens": response.usage.input_tokens,
            "answers": answers_to_row(response.answers),
        }


async def main(args):
    df = load_reviews()
    out_path = HERE / ("predictions.mock.jsonl" if args.mock else "predictions.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open()}
    todo = df[~df["id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already classified, {len(todo)} to go -> {out_path.name}")
    if todo.empty:
        return

    client_kwargs = {"model": args.model, "retry": RetryPolicy(max_retries=4), "timeout": 30.0}
    if args.mock:
        client_kwargs["api_key"] = "mock"
        client_kwargs["transport"] = mock_transport()

    questions = build_questions()
    sem = asyncio.Semaphore(args.concurrency)
    failures = 0
    async with AsyncTypeSafeClient(**client_kwargs) as client:
        tasks = [
            asyncio.create_task(classify_one(client, sem, r.id, r.review, questions, args.model))
            for r in todo.itertuples()
        ]
        with out_path.open("a") as f:
            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    row = await task
                except TypeSafeError as e:
                    failures += 1
                    print(f"  request failed: {type(e).__name__}: {e}")
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 100 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}")

    rows = [json.loads(line) for line in out_path.open()]
    tokens = sum(r["input_tokens"] or 0 for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    print(f"\n{len(rows)} reviews classified, {failures} failures")
    print(f"models that answered: {sorted({r['model'] for r in rows})}")
    print(f"input tokens: {tokens:,}  ->  ${tokens * 0.042 / 1e6:.4f} at $0.042/M (output is free)")
    print(f"latency per call: p50 {latencies[len(latencies) // 2]} ms, p95 {latencies[int(len(latencies) * 0.95)]} ms")


# ---------------------------------------------------------------------------------------
# Offline mock. Fakes Jev's response *shape* with crude keyword rules so you can check the
# plumbing and the evaluation without a key. Its numbers say nothing about Jev.
# ---------------------------------------------------------------------------------------
MOCK_KEYWORDS = {
    "general": r"\b(phone|device|iphone|galaxy|love|great|best|worst|disappoint|amazing|good)\b",
    "feature": r"\b(battery|chip|fast|slow|ram|storage|sd|ios|android|os|face ?id|fingerprint|jack|feature|size|big)\b",
    "price": r"\b(price|expensive|cheap|overpric|cost|\$|eur|worth|money|budget)",
    "camera": r"\b(camera|photo|picture|lens|zoom|selfie|mp|video)\b",
    "design_screen": r"\b(design|ugly|ugliest|look|notch|bezel|screen|display|720p|hd|lcd|oled|beautiful)",
}
MOCK_NEGATIVE = r"\b(no|not|ugly|ugliest|bad|worst|expensive|overpric|disappoint|terrible|joke|never|poor|slow|too)"


def mock_transport():
    import httpx2

    rng = random.Random(0)

    def handler(request):
        body = json.loads(request.content)
        text = body["state"]["review"].lower()
        negative = bool(re.search(MOCK_NEGATIVE, text))
        answers = {}
        for key, pattern in MOCK_KEYWORDS.items():
            hit = bool(re.search(pattern, text))
            p = min(max((0.85 if hit else 0.1) + rng.uniform(-0.15, 0.15), 0.01), 0.99)
            answers[f"mention_{key}"] = {"type": "noul", "noul": p}
            pos = 0.2 if negative else 0.8
            answers[f"polarity_{key}"] = choice_answer({"positive": pos, "negative": 1 - pos})
            joint = {"not_mentioned": 1 - p, "positive": p * pos, "negative": p * (1 - pos)}
            answers[f"joint_{key}"] = choice_answer(joint)
        payload = {
            "model": "mock-not-jev",
            "answers": answers,
            "usage": {"input_tokens": len(request.content) // 4, "output_tokens": 0},  # ~4 chars/token
        }
        return httpx2.Response(200, json=payload)

    return httpx2.MockTransport(handler)


def choice_answer(probs):
    best = max(probs, key=probs.get)
    return {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only classify this many new reviews")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=8, help="parallel requests (limit is 1,200/min)")
    parser.add_argument("--mock", action="store_true", help="offline dry run with fake answers")
    asyncio.run(main(parser.parse_args()))
