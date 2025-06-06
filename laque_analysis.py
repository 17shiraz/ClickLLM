import os
import pickle
import pandas as pd
import ast
import copy
import random
import argparse
from ollama import chat
from ollama import ChatResponse

random.seed(42)

# === Argument Parser ===
#parser = argparse.ArgumentParser()
#parser.add_argument("--model_name", type=str, default="qwen3")
#parser.add_argument("--n_entities", type=int, default=5)
#args = parser.parse_args()

model_name = "llama4" #args.model_name
N = None#args.n_entities

# === Hardcoded entity-level irrelevance files ===
irrelevant_files = {
    "3": "/mnt/data/msaliminabi/entity/llmrel/outputs/zero_by_3_llms.txt",
    "4": "/mnt/data/msaliminabi/entity/llmrel/outputs/zero_by_4_llms.txt"
}

# === Entity text source ===
tsv_base_path = "/mnt/data/msaliminabi/entity/llmrel/laque_data"

# === Prompt Template (Entity Judgement Task) ===
'''
entity_judgement_prompt = [
    {
        "role": "system",
        "content": (
            "You are ClickLLM, an intelligent assistant that updates a list of atomic reasons \
why a clicked wikipedia entity may appear irrelevant to the user's search query. \
Each reason should be atomic, generalizable, and interpretable."
        )
    },
    {
        "role": "user",
        "content": ( 'Update the list of atomic reasons (1-12 words), if needed, '
        'so they best describe why the user-clicked entity might appear irrelevant to the user query. '
        'Leverage only the initial list of atomic reasons (if exists). '
        'Return only the final list in valid Python format (even if no updates). '
        'Ensure no redundant or overly specific reasons. Keep at least 5 and at most 30 of the most vital ones. '
        'Use only generalizable, conceptual language.'
        'Search Query: {}'
        'Entity Title or ID: {}'
        'Initial Assessment Criteria List: {}'
        'Initial Assessment Criteria List Length: {}'
        'Only output the updated list in Python format (e.g., ["a", "b"]). Do not explain.' )
    }
]
'''

'''
# why was it clicked separate
entity_judgement_prompt = [
    {
        "role": "system",
        "content": (
            "You are ClickLLM, an intelligent assistant that updates a list of atomic reasons \
why a wikipedia entity was clicked by a user despite being irrelevant to the user's search query. \
Each reason should be atomic, generalizable, and interpretable."
        )
    },
    {
        "role": "user",
        "content": ( 'Create list of atomic reasons (1-12 words)'
        'so they best describe why the entity was clicked on by the user despite appearing to be irrelevant to the user query. '
        'Return only the list in valid Python format. '
        'Ensure no redundant or overly specific reasons. Keep at most 5 of the most vital ones. '
        'Use only generalizable, conceptual language.'
        'Search Query: {}'
        'Entity Title or ID: {}'
        'Only output thelist in Python format (e.g., ["a", "b"]). Do not explain.' )
    }
]
'''

#'''
# why was it clicked separate psychology
entity_judgement_prompt = [
    {
        "role": "system",
        "content": (
            "You are ClickLLM, an expert in human behavior and psychology. Your job is to update a list of atomic reasons \
why a wikipedia entity was clicked by a user despite being irrelevant to the user's search query. \
Each reason should be atomic, generalizable, and interpretable."
        )
    },
    {
        "role": "user",
        "content": ( 'Create list of atomic reasons (4-8 words)'
        'so they best describe why the wikipedia entity was clicked on by the user despite appearing to be irrelevant to the user query. '
        'Return only the list in valid Python format. '
        'Consider the nature of the entity and why is it irrelevant to the query.'
        'Base your answers on your knowledge of human behavior, psychology, and clicking patterns.'
        'Ensure no redundant or overly specific reasons. Keep at most 5 of the most vital ones. '
        'Search Query: {}'
        'Entity Title or ID: {}'
        'Only output thelist in Python format (e.g., ["a", "b"]). Do not explain.' )
    }
]
#'''

'''
# separate list
entity_judgement_prompt = [
    {
        "role": "system",
        "content": (
            "You are ClickLLM, an intelligent assistant that updates a list of atomic reasons \
why a clicked wikipedia entity may appear irrelevant to the user's search query. \
Each reason should be atomic, generalizable, and interpretable."
        )
    },
    {
        "role": "user",
        "content": ( 'Create list of atomic reasons (1-12 words)'
        'so they best describe why the user-clicked entity might appear irrelevant to the user query. '
        'Each response must compare some property of the query and some property of the entity'
        'that are incompatible.'
        'Return only the final list in valid Python format. '
        'Ensure no redundant or overly specific reasons. Keep at most 3 of the most vital ones. '
        'Use only generalizable, conceptual language.'
        'Search Query: {}'
        'Entity Title: {}'
        'Only output the list in Python format (e.g., ["a", "b"]). Do not explain.' )
    }
]
'''
def load_irrelevant_entities(file_path):
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, _, docid = parts[0], parts[1], parts[2]
                entries.append((qid, docid))
    return entries

def main():
    if model_name not in ['llama4', 'qwen3']:
        raise ValueError("Model not available")

    print(f"\n=== Processing dataset: 0 ===")
    pkl_path = os.path.join(tsv_base_path, f"triples.run.queries.train.0.bm25.top1000.tsv.pkl")
    with open(pkl_path, 'rb') as f:
        triple_dic = pickle.load(f)

    queries = {qid: triple_dic[qid]['query'] for qid in triple_dic}

    for vote_count, file_path in irrelevant_files.items():
        entity_subset = load_irrelevant_entities(file_path)
        if N is not None:
            entity_subset = entity_subset[:min(N, len(entity_subset))]

        output_file = f'/mnt/data/msaliminabi/entity/llmrel/outputs/click_separete_reasons.{model_name}.vote_{vote_count}_all_psyche.txt'
        seen_set = set()
        previous_list = []

        print(f"Processing {len(entity_subset)} entities from {file_path}")

        for qid, docid in entity_subset:
            if qid not in queries:
                continue
            query = queries[qid]
            key = (qid, docid)
            if key in seen_set:
                continue
            seen_set.add(key)

            local_message = copy.deepcopy(entity_judgement_prompt)
            local_message[1]['content'] = local_message[1]['content'].format(query, docid)#, previous_list, len(previous_list))

            try:
                if model_name == 'llama4':
                    response: ChatResponse = chat(
                        model='hf.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q2_K_XL',
                        messages=local_message,
                        options={"temperature": 0}
                    )
                elif model_name == 'qwen3':
                    response: ChatResponse = chat(
                        model='qwen3:8b',
                        messages=local_message,
                        options={"temperature": 0}
                    )

                reply = response.message.content.strip()
                data = reply.split('</think>')[-1].strip()
                parsed = ast.literal_eval(data)
                previous_list = parsed

                print(f"[{vote_count} LLMs] {qid} - {docid} → {parsed[:3]}...")
                with open(output_file, 'a') as output:
                    output.write(f"0\t{qid}\t{docid}\t{query}\t{parsed}\n")

            except Exception as e:
                print(f"Error on {qid}, {docid}: {e}")
                continue

        print(f">>> Saved clues to: {output_file}")

if __name__ == "__main__":
    main()
