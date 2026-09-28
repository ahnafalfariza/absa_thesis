"""Every question Jev is asked about a review lives in this file.

With Jev, the question text *is* the model. The thesis trained a LogisticRegression on
~900 labelled reviews; here there is no training at all, so these descriptions carry
the whole definition of each aspect. Edit these, not the pipeline, when accuracy is off.

For every review we ask 15 questions in ONE call (speculative fan-out: questions run in
parallel, so extra questions cost a few tokens but almost no latency):

  mention_<aspect>   Noul    Is this aspect discussed?          -> aspect classification
  polarity_<aspect>  Choice  positive | negative                -> sentiment, if mentioned
  joint_<aspect>     Choice  not_mentioned | positive | negative -> both at once

The split (mention + polarity) and joint formulations answer the same thing two ways,
so the evaluation can tell you which framing Jev handles better on this data.
"""

from typesafe_sdk import Choice, Noul

# Pin the version you tuned against. `jev-latest` moves when TypeSafe ships a release.
MODEL = "jev-1.13.0"

# Thesis label -> short key used in question names.
ASPECTS = {
    "#GENERAL": "general",
    "#FEATURE": "feature",
    "#PRICE": "price",
    "#CAMERA": "camera",
    "#DESIGN#SCREEN": "design_screen",
}

# What counts as each aspect. Written from reading the labelled data, including its edge
# cases (e.g. phone size and battery were labelled #FEATURE; the display counts as design).
ASPECT_DEFINITIONS = {
    "general": {
        "name": "the phone overall",
        "true": (
            "The review gives an overall verdict on the phone or brand as a whole, "
            "e.g. 'amazing device', 'not a good phone', 'really disappointed', 'best iPhone ever'."
        ),
        "false": (
            "The review only judges specific parts (battery, price, camera, screen, design) "
            "without an overall verdict on the phone."
        ),
    },
    "feature": {
        "name": "features, hardware and software",
        "true": (
            "The review discusses functionality, specs or performance other than the camera and "
            "the screen: battery, chipset, speed, RAM, storage, SD card slot, OS or iOS/Android, "
            "updates, Face ID, fingerprint sensor, headphone jack, size or weight, or new/missing features."
        ),
        "false": "None of those functional aspects are discussed.",
    },
    "price": {
        "name": "price and value for money",
        "true": (
            "The review discusses what the phone costs or whether it is worth the money: "
            "expensive, cheap, overpriced, a price figure, value for money."
        ),
        "false": "Cost and value for money are not discussed.",
    },
    "camera": {
        "name": "the camera",
        "true": (
            "The review discusses the camera's photos, video or capabilities: image quality, "
            "megapixels, number of lenses, zoom, night mode, selfie camera, ultrawide."
        ),
        "false": (
            "The camera's photo or video capability is not discussed. A remark only about how "
            "the camera bump or lens layout looks is about design, not the camera."
        ),
    },
    "design_screen": {
        "name": "design, looks and screen",
        "true": (
            "The review discusses how the phone looks or is built, or its display: design, "
            "ugly or beautiful, colours, notch, bezels, camera bump appearance, build materials, "
            "screen or display quality, resolution (720p, HD, LCD, OLED)."
        ),
        "false": "Appearance, build and the display are not discussed.",
    },
}


def build_questions() -> dict:
    questions = {}
    for key, d in ASPECT_DEFINITIONS.items():
        questions[f"mention_{key}"] = Noul(
            instructions=f"Does the review express an opinion about {d['name']}?",
            criteria={"true": d["true"], "false": d["false"]},
        )
        questions[f"polarity_{key}"] = Choice(
            instructions=(
                f"Considering only what the review says about {d['name']}, is the reviewer's "
                "sentiment towards it positive or negative? Sarcasm counts as negative."
            ),
            criteria={
                "positive": f"Praises, likes or is satisfied with {d['name']}.",
                "negative": f"Criticises, dislikes, mocks or is disappointed by {d['name']}.",
            },
        )
        questions[f"joint_{key}"] = Choice(
            instructions=f"What does the review say about {d['name']}?",
            criteria={
                "not_mentioned": d["false"],
                "positive": f"{d['true']} The reviewer is positive about it.",
                "negative": f"{d['true']} The reviewer is negative about it, including sarcasm.",
            },
        )
    return questions
