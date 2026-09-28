# Jev on the ABSA thesis data

1107 reviews, model(s): jev-1.13.0

Cost: 2,417,410 input tokens = **$0.1015**. Latency per call: p50 339 ms, p95 494 ms.

## 1. Aspect classification (is the aspect discussed?)

Scored on the held-out 882 test rows. Thresholds for 'Jev split, tuned' are chosen per aspect on the other 225 rows. The thesis baseline is 10-fold out-of-fold, i.e. trained on ~1,000 labelled reviews; Jev saw none.

| aspect | n | thesis LR F1 | Jev split @0.5 F1 | Jev split tuned F1 | Jev joint F1 | tuned threshold |
|---|---|---|---|---|---|---|
| #GENERAL | 226 | 0.641 | 0.737 | 0.742 | 0.663 | 0.800 |
| #FEATURE | 312 | 0.793 | 0.858 | 0.872 | 0.822 | 0.600 |
| #PRICE | 105 | 0.709 | 0.866 | 0.890 | 0.868 | 0.800 |
| #CAMERA | 94 | 0.744 | 0.848 | 0.848 | 0.861 | 0.500 |
| #DESIGN#SCREEN | 295 | 0.866 | 0.908 | 0.935 | 0.900 | 0.700 |
| **micro avg** | 1032 | 0.770 | 0.843 | 0.863 | 0.807 |  |

Precision / recall behind the tuned Jev column:

| aspect | thesis P | thesis R | Jev P | Jev R |
|---|---|---|---|---|
| #GENERAL | 0.762 | 0.553 | 0.759 | 0.726 |
| #FEATURE | 0.916 | 0.699 | 0.833 | 0.913 |
| #PRICE | 0.910 | 0.581 | 0.828 | 0.962 |
| #CAMERA | 0.821 | 0.681 | 0.808 | 0.894 |
| #DESIGN#SCREEN | 0.944 | 0.800 | 0.900 | 0.973 |

## 2. Sentiment, given the aspect (the thesis setup)

Only reviews whose gold label includes the aspect, as in `sentiment_analysis.py`. All rows (no tuning is involved). Macro-F1 matters here: #PRICE and #CAMERA are heavily imbalanced, so accuracy flatters any model that leans to the majority class.

| aspect | pos / neg | thesis acc | thesis macro-F1 | Jev polarity acc | Jev polarity macro-F1 | Jev joint acc | Jev joint macro-F1 |
|---|---|---|---|---|---|---|---|
| #GENERAL | 206+ / 78- | 0.817 | 0.718 | 0.915 | 0.902 | 0.940 | 0.928 |
| #FEATURE | 199+ / 202- | 0.791 | 0.791 | 0.915 | 0.915 | 0.943 | 0.943 |
| #PRICE | 25+ / 110- | 0.800 | 0.444 | 0.941 | 0.895 | 0.926 | 0.873 |
| #CAMERA | 92+ / 28- | 0.783 | 0.575 | 0.933 | 0.915 | 0.933 | 0.913 |
| #DESIGN#SCREEN | 103+ / 252- | 0.814 | 0.741 | 0.946 | 0.935 | 0.963 | 0.956 |

## 3. End to end: (aspect, sentiment) pairs from raw text

What a real system has to do: find the aspects *and* their sentiment with no gold help. A pair counts only if both are right. Test rows. The thesis pipeline cannot be scored this way from its out-of-fold predictions, which is itself a gap in the original evaluation.

| approach | precision | recall | F1 |
|---|---|---|---|
| Jev split (tuned mention + polarity) | 0.787 | 0.841 | 0.813 |
| Jev joint (one 3-way choice) | 0.664 | 0.913 | 0.769 |

## 4. Can you trust the confidence?

If calibration holds, answers in the 0.9+ bucket should be right about 90%+ of the time. Polarity answers on gold-mentioned rows, all aspects pooled.

| confidence | answers | accuracy |
|---|---|---|
| (-0.001, 0.6] | 78 | 0.577 |
| (0.6, 0.7] | 24 | 0.708 |
| (0.7, 0.8] | 36 | 0.806 |
| (0.8, 0.9] | 51 | 0.804 |
| (0.9, 1.0] | 1106 | 0.967 |

## 5. Label audit

157 confident disagreements written to `label_audit.csv`. Read a sample before blaming the model: on a dataset labelled by one person, a share of these will be gold-label mistakes.

| aspect | kind | gold | jev | review |
|---|---|---|---|---|
| #CAMERA | mention | False | 0.980 | The camera on the XR has a larger sensor than the iPhone cameras from last year, but loses the telephoto lens. |
| #FEATURE | mention | False | 0.960 | best phone I've ever used. swipe up for home button and face id works flawless even in the dark. phone is a mu |
| #GENERAL | mention | False | 0.980 | It works great and from the processor to the camera i had nothing to say. Honestly I recommend this phone it’s |
| #GENERAL | sentiment | positive | negative (1.00) | We somehow defend iPhone X only by good, better, best having no lingual expression of overall performance, but |
| #FEATURE | mention | False | 0.950 | In my opinion it's absolutely crap phone. Huge (even wider than Galaxy Note), heavyweight with low resolution  |
| #GENERAL | mention | False | 0.930 | on top of that the A12, the great camera and you have a clear winner. No wonder it's gonna sell great. |
| #GENERAL | sentiment | positive | negative (0.98) | I used it, great phone but needs more features! Not worth the money imho! |
| #FEATURE | sentiment | positive | negative (0.99) | for the iOS,no one would think Apple is actually selling a good products to consumers. |
