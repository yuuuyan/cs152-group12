import vertexai
from vertexai.generative_models import GenerativeModel
from together import Together
import csv
import json
import os
import argparse
import time
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--data-file", required=True)
parser.add_argument("--model", required=True, help="gemini, llama8b, llama70b")
parser.add_argument("--output-file", required=True)
args = parser.parse_args()

num_save = 50

token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    TOGETHER_AI_API_KEY = tokens['together-ai']
    GCP_PROJECT_ID = tokens['gcp']

prompt_fn = "LLM_prompt.txt"

with open(prompt_fn, 'r', encoding='utf-8') as file:
    prompt = file.read()

def query_together_ai(model: str, text: str) -> dict:
    client = Together(api_key=TOGETHER_AI_API_KEY)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}]
    )

    return response.choices[0].message.content


def query_gcp(text: str) -> dict:
    vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
    model = GenerativeModel(model_name="gemini-1.0-pro-002")

    response = model.generate_content(
        text
    )

    return response.text

outputs = defaultdict(dict)

with open(args.data_file, newline='', encoding='utf-8') as csvfile:
    csvreader= csv.reader(csvfile, delimiter=',')

    # skip header
    next(csvreader)

    for i, row in enumerate(csvreader):
        message = row[0]
        model_input = prompt+message

        outputs[i]["message"] = message
        outputs[i]["gt"] = int(row[1])

        success = False
        retries = 3
        count = 0
        while not success and count < retries:
            try:
                if args.model == "llama70b":
                    response = query_together_ai("meta-llama/Llama-3-70b-chat-hf", model_input)
                elif args.model == "llama8b":
                    response = query_together_ai("meta-llama/Llama-3-8b-chat-hf", model_input)
                elif args.model == "gemini":
                    response = query_gcp(model_input)
                outputs[i]["response"] = response
                response = response.lower()
                outputs[i]["prediction"] = 1 if "yes" in response else 0 
                success = True
            except:
                wait = retries * 30
                time.sleep(wait)
                count += 1
                outputs[i]["prediction"] = None

        if i % num_save == 0 and i != 0:
            with open(args.output_file, 'w') as f:
                json.dump(outputs, f, indent=6)

with open(args.output_file, 'w') as f:
    json.dump(outputs, f, indent=6)

