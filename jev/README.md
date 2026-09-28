# Jev on the ABSA thesis

Your thesis pipeline, redone with TypeSafe's Jev: same GSMArena reviews, same five aspects,
same positive/negative sentiment, scored with the same metrics against your original models.

## How the thesis maps onto Jev

| Thesis step | Thesis model | Jev version |
|---|---|---|
| Preprocessing | `CleanText`: lowercase, strip punctuation, stopwords, lemmatise, stem | none; Jev reads the raw review |
| Aspect classification (multi-label) | CountVectorizer(1-2 gram) + one-vs-rest LogisticRegression, trained on ~1,000 labels | five `Noul`s, `mention_<aspect>`: probability the aspect is discussed |
| Sentiment per aspect | one LogisticRegression per aspect, trained on that aspect's labelled rows | five `Choice`s, `polarity_<aspect>`: positive or negative |
| Both at once | not in the thesis | five `Choice`s, `joint_<aspect>`: not_mentioned / positive / negative |
| Training data needed | ~1,000 labelled reviews | none; the aspect definitions in `questions.py` replace it |

All 15 questions for a review go in **one** request. Jev answers them in parallel, so the extra
questions cost a few tokens and almost no time. That's why it asks both the split and joint versions and
lets the evaluation decide which works better.

## Run it

```bash
cd jev
pip install -r requirements.txt
export TYPESAFE_API_KEY=sk-...            # console.typesafe.ai, or see "Vercel" below

python baseline.py                       # thesis models, out-of-fold  (already included)
python classify.py --limit 20            # try 20 reviews, check the output and cost
python classify.py                       # all 1,107 reviews: ~2.1M tokens, about $0.10
python evaluate.py                       # -> results.md and label_audit.csv
```

`classify.py` appends to `predictions.jsonl` and skips reviews already done, so an interrupted
run resumes and never pays twice. `python classify.py --mock && python evaluate.py --mock` runs
the whole thing offline with fake answers if you want to check the plumbing first.

The thesis code needs NLTK's `stopwords` and `wordnet` corpora for `baseline.py`:
`python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"`.

## What `evaluate.py` reports

1. **Aspect classification.** Per-aspect F1 for the thesis LR, Jev at a 0.5 threshold, Jev with
   per-aspect thresholds tuned on a 20% dev split, and the joint formulation. Scored on the other 80%.
2. **Sentiment given the aspect.** The thesis setup: only rows where gold says the aspect is present.
   Reports accuracy *and* macro-F1, because #PRICE (25 positive vs 110 negative) and #CAMERA are
   imbalanced enough that accuracy hides a model that just predicts the majority class.
3. **End to end.** (aspect, sentiment) pairs from raw text, which is what a real product needs and
   what the thesis evaluation never measured.
4. **Calibration.** Whether Jev's 0.9-confidence answers are actually right ~90% of the time on this data.
5. **Label audit.** `label_audit.csv` lists every review where Jev confidently disagrees with gold.

## Read the results with this in mind

- **The comparison is not like for like, and that's the point.** The thesis model trained on
  ~1,000 of these exact reviews. Jev saw none, only the five definitions in `questions.py`. If
  Jev gets close, the takeaway is "zero-shot from a written rubric nearly matches a supervised
  model", not "Jev is better".
- **The gold labels are noisy.** For example, "Great specs, except of it's golden price" is labelled
  #PRICE *positive*, "higher price 64GB $999" has no #PRICE label, and "The front camera looks quite
  small and nice in preview" is #DESIGN#SCREEN *negative*. A model can be right and still be scored
  wrong. Read `label_audit.csv` before concluding Jev failed on a case.
- **When a number is bad, fix the question, not the pipeline.** Jev reads instructions literally. If it
  keeps missing an aspect, look at the misses and tighten that aspect's `true`/`false` description in
  `questions.py`. Your labelling conventions (phone size counts as #FEATURE, a camera bump's looks count
  as design) exist only in your head and in the data until they're written there.
- **Pin the model.** `questions.py` pins `jev-1.13.0`. Tuned thresholds are only valid for the
  version they were tuned on, and every row in `predictions.jsonl` records which version answered.

## No TypeSafe account?

Signups were paused on 22 September. Jev is also on Vercel's AI Gateway as `typesafe-ai/jev` at the
same price, but that route uses the Vercel AI SDK (`experimental_evaluate`), not this Python SDK.
The questions port over directly; the call site changes.
