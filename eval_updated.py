import json
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from multiprocessing import Pool
import torch
from evaluate import load
from bert_score import score

# COMET
from comet import download_model, load_from_checkpoint

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
args = parser.parse_args()
file_path = args.model_path

# File path assumed to be defined before this snippet
file_format = file_path.split('/')[-1].replace(".json", ".csv")

# Load the JSON data
with open(file_path, 'r') as f:
    json_data = json.load(f)

# Initialize metrics: ROUGE, SBERT, BERTScore
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
# bertscore = load("bertscore")
# bert_score_model = "microsoft/deberta-v3-large"

bert_score_model = "microsoft/deberta-v3-large"


# Load COMET model
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

# Function to calculate ROUGE and BLEU
def compute_metrics(sample):
    true_desc = sample["true"]
    gen_desc = sample["generated"]

    # ROUGE
    rouge_scores = scorer.score(true_desc, gen_desc)
    # BLEU
    smoothing_function = SmoothingFunction().method1
    bleu_score_val = sentence_bleu(
        [true_desc.split()],
        gen_desc.split(),
        weights=(0.5, 0.5),
        smoothing_function=smoothing_function
    )

    return {
        "id": sample["id"],
        "video": sample["video"],
        "rouge1": rouge_scores['rouge1'].fmeasure,
        "rouge2": rouge_scores['rouge2'].fmeasure,
        "rougeL": rouge_scores['rougeL'].fmeasure,
        "bleu": bleu_score_val
    }


# Modified SBERT section
def compute_sbert_similarity(true_descriptions, generated_descriptions):
    # Initialize model inside function
    sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reduce batch size to 32 for lower memory usage
    true_embs = sbert_model.encode(true_descriptions, convert_to_tensor=True, batch_size=32)
    gen_embs = sbert_model.encode(generated_descriptions, convert_to_tensor=True, batch_size=32)
    
    # Compute similarities while keeping memory footprint low
    similarities = []
    for true_emb, gen_emb in zip(true_embs, gen_embs):
        similarities.append(util.pytorch_cos_sim(true_emb, gen_emb).item())
    
    # Explicit cleanup
    del sbert_model, true_embs, gen_embs
    torch.cuda.empty_cache()
    
    return similarities

# Compute COMET scores
def compute_comet_scores(srcs, preds, refs):
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
    comet_outputs = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    return comet_outputs.scores

# Prepare descriptions
true_descriptions = [sample['true'] for sample in json_data]
generated_descriptions = [sample['generated'] for sample in json_data]
src_inputs = ["N/A"] * len(json_data)  # Placeholder source since you only have ref and pred

# Run metrics
sbert_similarities = compute_sbert_similarity(true_descriptions, generated_descriptions)


# bertscore_results = bertscore.compute(
#     predictions=generated_descriptions,
#     references=true_descriptions,
#     lang="en",
#     model_type=bert_score_model
# )

P, R, F1 = score(
    generated_descriptions,
    true_descriptions,
    model_type=bert_score_model,
    lang="en",
    use_fast_tokenizer=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# if 'bertscore' in globals():
#     del bertscore
# torch.cuda.empty_cache()

comet_scores = compute_comet_scores(src_inputs, generated_descriptions, true_descriptions)


with Pool() as pool:
    metric_results = pool.map(compute_metrics, json_data)

for i, result in enumerate(metric_results):
    result['sbert_similarity'] = sbert_similarities[i]
    result['bertscore_f1'] = F1[i].item()
    result['bertscore_precision'] = P[i].item()
    result['bertscore_recall'] = R[i].item()
    result['comet_score'] = comet_scores[i]

# Save with average row
df = pd.DataFrame(metric_results)
avg_results = df.mean(numeric_only=True)
avg_results_row = pd.DataFrame(avg_results).transpose()
avg_results_row['id'] = 'average'
avg_results_row['video'] = 'average'
df_with_avg = pd.concat([df, avg_results_row], ignore_index=True)
df_with_avg.to_csv(file_format, index=False)
