import argparse
import json

import code_bert_score
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("result_file", type=str, help="Path to the result file")
args = parser.parse_args()

# Load the saved file
with open(args.result_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract predictions and references
predictions = [item["generated"] for item in data]
references = [item["target"] for item in data]

# Initialize metrics
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# Compute BLEU
bleu_results = bleu.compute(
    predictions=predictions,
    references=[[ref] for ref in references]  # BLEU requires list of lists for references
)

# Compute BERTScore
bertscore_results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"  # Specify language
)

codebertscore_results = code_bert_score.score(
    cands=predictions,
    refs=references,
    lang='python')

# Display results
print("BLEU Score:", bleu_results["bleu"])
print("BERTScore:")
print("Precision:", sum(bertscore_results["precision"]) / len(bertscore_results["precision"]))
print("Recall:", sum(bertscore_results["recall"]) / len(bertscore_results["recall"]))
print("F1 Score:", sum(bertscore_results["f1"]) / len(bertscore_results["f1"]))
print("CodeBERTScore:")
print("Precision:", float(codebertscore_results[0].sum()) / len(codebertscore_results[0]))
print("Recall:", float(codebertscore_results[1].sum()) / len(codebertscore_results[1]))
print("F1 Score:", float(codebertscore_results[2].sum()) / len(codebertscore_results[2]))
print("F3 Score:", float(codebertscore_results[3].sum()) / len(codebertscore_results[3]))
