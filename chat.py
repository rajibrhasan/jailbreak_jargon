import openai
import json
import sys
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse



def call_llm(model_name, query):
    response: ChatResponse = chat(model=model_name, messages=[{
        'role': 'user',
        'content': query,
    }])
    return response['message']['content']

def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(path, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def main():
    MODEL_NAME = sys.argv[1]
    INPUT_FILE = sys.argv[2]

    OUTPUT_FILE = f"{INPUT_FILE.rsplit('.', 1)[0]}_{MODEL_NAME}_{MODEL_NAME.replace(':', '')}.json"

    queries = load_queries(INPUT_FILE)
    results = []   

    for item in tqdm(queries):
        answer = call_llm(MODEL_NAME, item["query"])
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





    
