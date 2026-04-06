from utils import *

def evaluate_analogies(models, tests, test_type="Semantic"):
    print(f"\n--- {test_type} Analogy Tests ---")
    model_names = list(models.keys())
    header = f"{'Test #':<8} | {'Test format':<40} | {'Expected':<15} | {model_names[0]:<15} | {model_names[1]:<15} | {model_names[2]:<15}"
    print(header)
    print("-" * len(header))

    stats = {name: {"top1": 0, "top5": 0} for name in models.keys()}

    for i, (a, b, c, expected) in enumerate(tests, 1):
        test_str = f"{a}:{b} :: {c}:?"
        results_str = []

        for name, model in models.items():
            predictions = predict_analogy(model, a, b, c, k=5)

            top1_result = predictions[0] if len(predictions) > 0 else "N/A"
            results_str.append(top1_result)

            if len(predictions) > 0 and predictions[0].lower() == expected.lower():
                stats[name]["top1"] += 1
            if any(p.lower() == expected.lower() for p in predictions):
                stats[name]["top5"] += 1

        print(
            f"{i:<8} | {test_str:<40} | {expected:<15} | {results_str[0]:<15} | {results_str[1]:<15} | {results_str[2]:<15}")

    return stats
def semantic_test():
    semantic_tests = [
    ("poet", "poem", "novelist", "novel"),
    ("director", "movie", "playwright", "play"),
    ("composer", "music", "architect", "building"),
    ("comedian", "joke", "magician", "trick"),

    ("actor", "stage", "musician", "concert"),
    ("ballet", "dance", "opera", "singing"),
    ("choir", "voices", "orchestra", "instruments"),
    ("jazz", "saxophone", "rock", "guitar"),
    ("mozart", "piano", "paganini", "violin"),
    ("rhyme", "poetry", "melody", "music")
]
    return semantic_tests
def syntactic_test():
    syntactic_tests = [
        ("rich", "richest", "poor", "poorest"),
        ("heavy", "heaviest", "light", "lightest"),
        ("smart", "smartest", "dumb", "dumbest"),
        ("bright", "brightest", "dark", "darkest"),
        ("hard", "hardest", "soft", "softest"),
        ("good", "best", "bad", "worst"),
        ("hard", "hardest", "soft", "softest"),
        ("far", "farthest", "near", "nearest"),
        ("easy", "easiest", "hard", "hardest"),
        ("happy", "happiest", "sad", "saddest")
    ]

    return syntactic_tests