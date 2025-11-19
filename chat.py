import openai
import json
import sys
from tqdm import tqdm
from datasets import load_dataset



def call_llm(model_name, port, query):
    openai.api_base = f"http://localhost:{port}/v1"
    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
    )
    return resp["choices"][0]["message"]["content"]

def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(path, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def main():
    MODEL_NAME = sys.argv[1]
    INPUT_FILE = sys.argv[2]
    PORT = sys.argv[3]

    OUTPUT_FILE = f"{INPUT_FILE.rsplit('.', 1)[0]}_{MODEL_NAME}_{MODEL_NAME.rsplit('/', 1)[-1]}.json"

    queries = load_queries(INPUT_FILE)
    results = []   

    for item in tqdm(queries):
        answer = call_llm(MODEL_NAME, PORT, item["query"])
        results.append({
            "id": item["id"],
            "query": item["query"],
            "response": answer
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()





    
