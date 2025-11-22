
import json
import sys
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import os

load_dotenv()

def init_client(api_src: str) -> OpenAI:
    """
    Initialize an OpenAI-compatible client for either Ollama or OpenRouter.
    Reads base_url and api_key from environment variables.
    """
    if api_src == 'ollama':
        base_url = os.environ.get("OLLAMA_URL")
        api_key = os.environ.get("OLLAMA_KEY", "not-needed")  # Ollama often doesn't require key
    elif api_src == 'openrouter':
        base_url = os.environ.get("OPENROUTER_URL")
        api_key = os.environ.get("OPENROUTER_KEY")
    else:
        raise NotImplementedError(f"Unknown api_src: {api_src}")

    if not base_url:
        raise ValueError(f"Missing base_url for '{api_src}'. Check environment variables.")

    return OpenAI(base_url=base_url, api_key=api_key)


def call_llm(client, model_name, query):
    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}]
    )
    return res.choices[0].message.content

def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(path, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



def parse_args():
    parser = argparse.ArgumentParser(description="Run model on input file")
    parser.add_argument("-src", type = str, default='ollama', help = "Source of the api")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file")

    return parser.parse_args()

def main():

    args = parse_args()

    OUTPUT_FILE = f'files/{args.model.replace(':', "_")}_{args.input.split('/')[-1]}'

    client = init_client(args.src)

    queries = load_queries(args.input)
    results = []   

    for item in tqdm(queries):
        answer = call_llm(client, args.model, item["query"])
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





    
