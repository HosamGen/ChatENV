import json
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import sys

# Define keyword groups
KEYWORD_GROUPS = {
    "increase": ["increase", "increased", "increases", "increasing"],
    "decrease": ["decrease", "decreased", "decreases", "decreasing"],
    "rise": ["rise", "rises", "rose", "rising"],
    "drop": ["drop", "drops", "dropped", "dropping"],
    "high": ["high", "higher", "elevated", "elevation", "elevates"],
    "low": ["low", "lower", "reduce", "reduced", "reduces", "reducing"],
    "gain": ["gain", "gains", "gained", "gaining"],
    "fall": ["fall", "falls", "fell", "falling"],
    "vibrant": ["vibrant", "lush", "greener", "brighter", "saturated"],
    "muted": ["muted", "dormant", "faded", "duller", "washed out"],
    "color_richness": ["richer", "contrasted", "defined", "clearer"],
    "temperature": ["warmer", "cooler", "hot", "cold", "milder"],
    "humidity": ["humid", "moist", "drier"],
    "change": ["change", "changed", "changes", "changing"],
    "shift": ["shift", "shifts", "shifted", "shifting"],
    "transition": ["transition", "transitions", "transitioned", "transitioning"],
}


def evaluate_grouped_keywords(data, keyword_groups):
    group_metrics = {}
    y_true_all, y_pred_all = [], []
    #this is how the key is formatted based on the model used for evaluation (qwen or video models)
    key = "true" if any(s in args.file for s in ["Video-LLaVA", "LLaVA-NeXT-Video"]) else "ground_truth" 

    for group, variants in keyword_groups.items():
        y_true_group, y_pred_group = [], []

        for item in data:
            pred_text = item["generated"].lower()

            
            ref_text = item[key].lower()

            true_match = any(v in ref_text for v in variants)
            pred_match = any(v in pred_text for v in variants)

            y_true_group.append(int(true_match))
            y_pred_group.append(int(pred_match))

        y_true_all.extend(y_true_group)
        y_pred_all.extend(y_pred_group)

        group_metrics[group] = {
            "precision": precision_score(y_true_group, y_pred_group, zero_division=0),
            "recall": recall_score(y_true_group, y_pred_group, zero_division=0),
            "f1": f1_score(y_true_group, y_pred_group, zero_division=0),
            "accuracy": accuracy_score(y_true_group, y_pred_group),
            "support": sum(y_true_group)
        }

    macro_precision = np.mean([m["precision"] for m in group_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in group_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in group_metrics.values()])
    micro_precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    micro_recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    micro_f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    micro_accuracy = accuracy_score(y_true_all, y_pred_all)

    group_metrics["__overall__"] = {
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "accuracy": micro_accuracy
        }
    }

    return group_metrics


def format_scores(scores_dict):
    return {
        k: int(v) if k == "support" else round(float(v), 4)
        for k, v in scores_dict.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate grouped keyword metrics from a JSON file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the JSON results file.")
    parser.add_argument("--output", type=str, default=None, help="Path to the output log file if needed.")

    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = evaluate_grouped_keywords(data, KEYWORD_GROUPS)

    # Redirect stdout to file if output is specified
    if args.output:
        sys.stdout = open(args.output, "w")

    print(f"\nResults for file: {args.file}")
    print("Overall Macro Average:", format_scores(results["__overall__"]["macro_avg"]))
    print("Overall Micro Average:", format_scores(results["__overall__"]["micro_avg"]))

    print("\nKeyword Group Metrics:")
    for word, scores in results.items():
        if word.startswith("__"):
            continue
        print(f"{word}: {format_scores(scores)}")

    if args.output:
        sys.stdout.close()
