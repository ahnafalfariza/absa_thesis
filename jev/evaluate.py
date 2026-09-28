"""Score Jev against the gold labels and the thesis baseline, on the same rows with the same code.

    python evaluate.py                 # scores predictions.jsonl
    python evaluate.py --mock          # scores predictions.mock.jsonl (plumbing check only)

Writes results.md (the tables below) and label_audit.csv (reviews where Jev confidently
disagrees with the gold label: worth reading, because some of those are labelling errors).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from classify import load_reviews
from questions import ASPECTS

HERE = Path(__file__).parent
KEYS = list(ASPECTS.values())
LABEL = {v: k for k, v in ASPECTS.items()}


def load_jev(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.open():
        r = json.loads(line)
        a = r["answers"]
        row = {"id": r["id"], "model": r["model"], "latency_ms": r["latency_ms"], "input_tokens": r["input_tokens"]}
        for k in KEYS:
            row[f"p_mention_{k}"] = a[f"mention_{k}"]["p_yes"]
            pol = a[f"polarity_{k}"]
            row[f"polarity_{k}"], row[f"polarity_conf_{k}"] = pol["choice"], pol["confidence"]
            jp = a[f"joint_{k}"]["probabilities"]
            row[f"joint_{k}"] = a[f"joint_{k}"]["choice"]
            row[f"joint_conf_{k}"] = a[f"joint_{k}"]["confidence"]
            # joint sentiment when forced to pick a side (for scoring on gold-mentioned rows)
            row[f"joint_side_{k}"] = "positive" if jp["positive"] >= jp["negative"] else "negative"
        rows.append(row)
    return pd.DataFrame(rows)


def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return p, r, f


def tune_threshold(y_true, p):
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return max(grid, key=lambda t: f1_score(y_true, p >= t, zero_division=0))


def table(headers, rows):
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def main(args):
    pred_path = HERE / ("predictions.mock.jsonl" if args.mock else "predictions.jsonl")
    gold = load_reviews()
    jev = load_jev(pred_path)
    base = pd.read_csv(HERE / "baseline_predictions.csv")
    df = gold.merge(jev, on="id").merge(base, on="id")
    out = []
    say = out.append

    if args.mock or any(m.startswith("mock") for m in df["model"].unique()):
        say("> **MOCK RUN.** These numbers come from keyword rules faking Jev's response shape. "
            "They test the plumbing and say nothing about Jev. Run `classify.py` with a real key.\n")

    say(f"# Jev on the ABSA thesis data\n\n{len(df)} reviews, model(s): {', '.join(sorted(df['model'].unique()))}\n")
    tokens = df["input_tokens"].sum()
    say(f"Cost: {tokens:,} input tokens = **${tokens * 0.042 / 1e6:.4f}**. "
        f"Latency per call: p50 {df['latency_ms'].median():.0f} ms, p95 {df['latency_ms'].quantile(0.95):.0f} ms.\n")

    # A dev/test split so tuned thresholds are not scored on the rows they were tuned on.
    rng = np.random.RandomState(0)
    is_dev = rng.rand(len(df)) < 0.2
    dev, test = df[is_dev], df[~is_dev]

    # ---- 1. Aspect classification --------------------------------------------------------
    say("## 1. Aspect classification (is the aspect discussed?)\n")
    say(f"Scored on the held-out {len(test)} test rows. Thresholds for 'Jev split, tuned' are chosen "
        f"per aspect on the other {len(dev)} rows. The thesis baseline is 10-fold out-of-fold, "
        "i.e. trained on ~1,000 labelled reviews; Jev saw none.\n")
    rows, micro = [], {"base": [[], []], "split": [[], []], "tuned": [[], []], "joint": [[], []]}
    thresholds = {}
    for k in KEYS:
        y = test[f"gold_mention_{k}"].to_numpy()
        t = tune_threshold(dev[f"gold_mention_{k}"].to_numpy(), dev[f"p_mention_{k}"].to_numpy())
        thresholds[k] = t
        preds = {
            "base": test[f"pred_mention_{k}"].to_numpy(),
            "split": test[f"p_mention_{k}"].to_numpy() >= 0.5,
            "tuned": test[f"p_mention_{k}"].to_numpy() >= t,
            "joint": test[f"joint_{k}"].to_numpy() != "not_mentioned",
        }
        row = [LABEL[k], int(y.sum())]
        for name, p in preds.items():
            row.append(prf(y, p)[2])
            micro[name][0].append(y), micro[name][1].append(p)
        row.append(t)
        rows.append(row)
    micro_row = ["**micro avg**", sum(r[1] for r in rows)]
    for name in ["base", "split", "tuned", "joint"]:
        micro_row.append(prf(np.concatenate(micro[name][0]), np.concatenate(micro[name][1]))[2])
    rows.append(micro_row + [""])
    say(table(["aspect", "n", "thesis LR F1", "Jev split @0.5 F1", "Jev split tuned F1", "Jev joint F1", "tuned threshold"], rows))

    say("\nPrecision / recall behind the tuned Jev column:\n")
    pr_rows = []
    for k in KEYS:
        y = test[f"gold_mention_{k}"].to_numpy()
        pb, rb, _ = prf(y, test[f"pred_mention_{k}"].to_numpy())
        pj, rj, _ = prf(y, test[f"p_mention_{k}"].to_numpy() >= thresholds[k])
        pr_rows.append([LABEL[k], pb, rb, pj, rj])
    say(table(["aspect", "thesis P", "thesis R", "Jev P", "Jev R"], pr_rows))

    # ---- 2. Sentiment on gold-mentioned rows ------------------------------------------
    say("\n## 2. Sentiment, given the aspect (the thesis setup)\n")
    say("Only reviews whose gold label includes the aspect, as in `sentiment_analysis.py`. All rows "
        "(no tuning is involved). Macro-F1 matters here: #PRICE and #CAMERA are heavily imbalanced, "
        "so accuracy flatters any model that leans to the majority class.\n")
    rows = []
    for k in KEYS:
        s = df[df[f"gold_sentiment_{k}"].notna()]
        y = s[f"gold_sentiment_{k}"]
        balance = f"{(y == 'positive').sum()}+ / {(y == 'negative').sum()}-"
        row = [LABEL[k], balance]
        for col in [f"pred_sentiment_{k}", f"polarity_{k}", f"joint_side_{k}"]:
            row += [accuracy_score(y, s[col]), f1_score(y, s[col], average="macro", zero_division=0)]
        rows.append(row)
    say(table(["aspect", "pos / neg", "thesis acc", "thesis macro-F1", "Jev polarity acc",
               "Jev polarity macro-F1", "Jev joint acc", "Jev joint macro-F1"], rows))

    # ---- 3. End to end ------------------------------------------------------------------
    say("\n## 3. End to end: (aspect, sentiment) pairs from raw text\n")
    say("What a real system has to do: find the aspects *and* their sentiment with no gold help. "
        "A pair counts only if both are right. Test rows. The thesis pipeline cannot be scored "
        "this way from its out-of-fold predictions, which is itself a gap in the original evaluation.\n")
    gold_pairs, split_pairs, joint_pairs = set(), set(), set()
    for r in test.itertuples():
        d = r._asdict()
        for k in KEYS:
            if isinstance(d[f"gold_sentiment_{k}"], str):
                gold_pairs.add((r.id, k, d[f"gold_sentiment_{k}"]))
            if d[f"p_mention_{k}"] >= thresholds[k]:
                split_pairs.add((r.id, k, d[f"polarity_{k}"]))
            if d[f"joint_{k}"] != "not_mentioned":
                joint_pairs.add((r.id, k, d[f"joint_{k}"]))

    def pair_scores(pred):
        tp = len(pred & gold_pairs)
        p, r = tp / max(len(pred), 1), tp / max(len(gold_pairs), 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    say(table(["approach", "precision", "recall", "F1"],
              [["Jev split (tuned mention + polarity)", *pair_scores(split_pairs)],
               ["Jev joint (one 3-way choice)", *pair_scores(joint_pairs)]]))

    # ---- 4. Calibration -----------------------------------------------------------------
    say("\n## 4. Can you trust the confidence?\n")
    say("If calibration holds, answers in the 0.9+ bucket should be right about 90%+ of the time. "
        "Polarity answers on gold-mentioned rows, all aspects pooled.\n")
    recs = []
    for k in KEYS:
        s = df[df[f"gold_sentiment_{k}"].notna()]
        recs.append(pd.DataFrame({"conf": s[f"polarity_conf_{k}"], "ok": s[f"polarity_{k}"] == s[f"gold_sentiment_{k}"]}))
    cal = pd.concat(recs)
    cal["bucket"] = pd.cut(cal["conf"], [0, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    g = cal.groupby("bucket", observed=True)["ok"].agg(["count", "mean"])
    say(table(["confidence", "answers", "accuracy"], [[str(b), int(c), m] for b, (c, m) in g.iterrows()]))

    # ---- 5. Label audit -----------------------------------------------------------------
    # Every disagreement goes to label_diff.csv; the confident ones also go to label_audit.csv.
    diff = []
    for r in df.itertuples():
        d = r._asdict()
        for k in KEYS:
            p, g_m = d[f"p_mention_{k}"], d[f"gold_mention_{k}"]
            if (p >= 0.5) != g_m:
                diff.append({"id": r.id, "aspect": LABEL[k], "kind": "mention",
                             "direction": "jev_missed" if g_m else "jev_extra", "gold": g_m,
                             "jev": round(p, 3), "confident": p >= 0.9 or p <= 0.1, "review": r.review})
            gs = d[f"gold_sentiment_{k}"]
            if isinstance(gs, str) and d[f"polarity_{k}"] != gs:
                conf = d[f"polarity_conf_{k}"]
                diff.append({"id": r.id, "aspect": LABEL[k], "kind": "sentiment",
                             "direction": f"{gs}->{d[f'polarity_{k}']}", "gold": gs,
                             "jev": f"{d[f'polarity_{k}']} ({conf:.2f})", "confident": conf >= 0.9,
                             "review": r.review})
    diff = pd.DataFrame(diff).sort_values(["aspect", "kind", "direction", "id"])
    diff.to_csv(HERE / "label_diff.csv", index=False)
    audit = diff[diff["confident"]][["id", "aspect", "kind", "gold", "jev", "review"]]
    audit.to_csv(HERE / "label_audit.csv", index=False)
    say(f"\n## 5. Label audit\n\n{len(audit)} confident disagreements written to `label_audit.csv`. "
        "Read a sample before blaming the model: on a dataset labelled by one person, a share of "
        "these will be gold-label mistakes.\n")
    if len(audit):
        sample = audit.sample(min(8, len(audit)), random_state=0)
        say(table(["aspect", "kind", "gold", "jev", "review"],
                  [[a.aspect, a.kind, a.gold, a.jev, a.review[:110].replace("|", "/")] for a in sample.itertuples()]))

    report = "\n".join(out)
    (HERE / ("results.mock.md" if args.mock else "results.md")).write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    main(parser.parse_args())
