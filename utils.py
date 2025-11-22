from numpy import extract
import pandas as pd
from collections import Counter
import json
import pandas as pd
from pathlib import Path

model_mapping = {
    'Llama-3.1-8B-Instruct': 'llama_3_1_8B_instruct',
    'Mistral-7B-Instruct-v0.3': 'mistral_7B_instruct',
    'Qwen2.5-3B-Instruct': 'qwen_2_5_3B_instruct',
    'TinyLlama-1.1B-Chat-v1.0': 'tinyllama_1.1B_chat_v1_0'
}

def count_categories(csv_file, col1="gpt-4o-mini", col2="llama_3_1_70b"):
    df = pd.read_csv(csv_file)

    counts_1 = Counter(df[col1].astype(str))
    counts_2 = Counter(df[col2].astype(str))

    print("\n===== CATEGORY COUNTS FOR MODEL 1:", col1, "=====")
    for cat, num in counts_1.items():
        print(f"{cat}: {num}")

    print("\n===== CATEGORY COUNTS FOR MODEL 2:", col2, "=====")
    for cat, num in counts_2.items():
        print(f"{cat}: {num}")

    print("\n==============================================\n")

    return counts_1, counts_2

def save_data(rows, domain, model):
    path = f"assets/{domain}_{model_mapping[model.split('/')[1]]}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)
    print("Saved json")


def extract_original(json_file, model):
    data = json.load(open(json_file, "r", encoding="utf-8"))
    rows = []

    for entry in data:
        original_query = entry["original_query"]
        for orig_resp in entry["original_responses"]:
            response_model = orig_resp["response_model"]
            response_text = orig_resp["response_text"]
            if response_model == model:
                rows.append({
                    "id": entry["id"],
                    "query": original_query,
                    "response": response_text
                })
    save_data(rows, "original", model)


def extract_domain(json_file, dom, model):
    data = json.load(open(json_file, "r", encoding="utf-8"))

    rows = []

    for entry in data:
        for conv in entry["conversions"]:
            domain = conv["domain"]
            if domain == dom:
                converted_query = conv.get("converted_query")
                # For each response model under each conversion
                for resp in conv["responses"]:
                    response_model = resp["response_model"]
                    response_text = resp["response_text"]

                    if response_model == model:
                        rows.append({
                            "id": entry["id"],
                            "query": converted_query,
                            "response": response_text
                        })
    save_data(rows, dom, model)


models = ['meta-llama/Llama-3.1-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'Qwen/Qwen2.5-3B-Instruct']
domains = ['cybersecurity', 'legal', 'psychology']

for model in models:
    extract_original('judge_responses_all.json', model)
    for dom in domains:
        extract_domain('judge_responses_all.json', dom, model)

    


