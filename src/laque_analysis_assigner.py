#["Name or lexical similarity", "Misspelling or typing error", "Category or topical association", "Geographic name confusion", "High‑ranked result bias", "Exploratory curiosity", "Related entity association"]
# ["Keyword or name similarity confusion", "Geographic or temporal name confusion", "Acronym or abbreviation misinterpretation", "Clicked broader parent topic for details", "Prominence or familiarity result bias"]
# ["Name or lexical similarity", "Category or topical association", "Geographic name confusion", "High‑ranked result bias", "Exploratory curiosity", "Clicked broader parent topic for details"]



# run2 ["Name or lexical similarity", "Clicked broader parent topic for details", "Geographic name confusion", "Prominent Result Bias", "Exploratory curiosity", "Familiarity/Recognition Bias"]

# run1 ["Name or lexical similarity", "Category or topical association", "Geographic name confusion", "Prominent Result Bias", "Exploratory curiosity", "Familiarity/Recognition Bias"]
import os
import pickle
import pandas as pd
import ast
import copy
import random
from ollama import chat, ChatResponse

random.seed(42)

# === Model Configuration ===
model_name = "qwen3"  # or "qwen3"

# === Hardcoded entity-level irrelevance files ===
irrelevant_files = {
    "3": "/mnt/data/msaliminabi/entity/llmrel/outputs/zero_by_3_llms.txt",
    "4": "/mnt/data/msaliminabi/entity/llmrel/outputs/zero_by_4_llms.txt"
}

# === Entity text source ===
tsv_base_path = "/mnt/data/msaliminabi/entity/llmrel/laque_data"

# === Fixed List of Reasons for One-Shot Labeling ===
reasons = ["Name or lexical similarity", "Category or topical association", "Geographic name confusion", "Prominent Result Bias", "Exploratory curiosity", "Familiarity/Recognition Bias"]

reasons_with_desc = [
    "Name or lexical similarity – The entity’s title closely matches the query’s wording or spelling",
    "Category or topical association – The entity sits in the same broad subject area as the query",
    "Geographic name confusion – The entity shares a place name with the query and is mistaken for it",
    "Prominent Result Bias – The entity is clicked mainly because it ranks highly or is typically among the most clicked for similar queries",
    "Exploratory curiosity – The user clicks out of interest in a tangential or loosely related topic",
    "Familiarity/Recognition Bias – The entity looks familiar or well-known, prompting a recognition-driven click"
]

def load_irrelevant_entities(file_path):
    """
    Reads a file where each line has at least: qid, some_unused_field, docid.
    Returns a list of (qid, docid) tuples.
    """
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, _, docid = parts[0], parts[1], parts[2]
                entries.append((qid, docid))
    return entries

def main():
    # 1. Load the BM25 top1000 triples for dataset index "0"
    print("=== Loading BM25 triples for dataset 0 ===")
    pkl_path = os.path.join(
        tsv_base_path,
        "triples.run.queries.train.0.bm25.top1000.tsv.pkl"
    )
    with open(pkl_path, "rb") as f:
        triple_dic = pickle.load(f)
    # Build a map from qid -> query text
    queries = {qid: triple_dic[qid]["query"] for qid in triple_dic}

    # 2. Iterate over each irrelevant_entities file (vote_count = "3" or "4")
    for vote_count, file_path in irrelevant_files.items():
        entity_subset = load_irrelevant_entities(file_path)
        output_file = (
            f"/mnt/data/msaliminabi/entity/llmrel/outputs/"
            f"run1_desc_multi_hot.{model_name}2.vote_{vote_count}.txt"
        )
        seen_set = set()
        print(f"\nProcessing {len(entity_subset)} entity pairs from {file_path}")

        # 3. For each (qid, docid), send a one-shot labeling prompt to the LLM
        for qid, docid in entity_subset:
            if qid not in queries:
                continue
            query_text = queries[qid]
            key = (qid, docid)
            if key in seen_set:
                continue
            seen_set.add(key)

            # Construct the system + user messages for one-shot labeling
            system_prompt = {
                "role": "system",
                "content": (
                    "You are ClickLLM, an intelligent assistant that assigns "
                    "binary labels (0 or 1) to the elements of a fixed list of atomic reasons. "
                    "Each label indicates whether the given reason explains why the "
                    "entity was clicked despite being irrelevant to the user's search query."
                    "/No_Think"
                )
            }


            user_prompt_description = {
                                "role": "user",
                "content": (
                    f"Based on the search query and the clicked entity, assign each of the "
                    f"{len(reasons_with_desc)} reasons a label of 1 (if it is a vital reason for the entity being clicked despite irrelevance) or 0 "
                    f"(if it is not). Return only a Python list of 0/1 values in the "
                    f"same order as the reasons below. Do not include any explanation.\n\n"
                    f"Assess each reason based on how applicable it is to the query-entity relationship.\n"
                    f"Search Query: {query_text}\n"
                    f"Entity Title or ID: {docid}\n"
                    f"Reasons: {reasons_with_desc}\n"
                    "Labels:"
                )
            }


            user_prompt = {
                "role": "user",
                "content": (
                    f"Based on the search query and the clicked entity, assign each of the "
                    f"{len(reasons)} reasons a label of 1 (if it is a vital reason for the entity being clicked despite irrelevance) or 0 "
                    f"(if it is not). Return only a Python list of 0/1 values in the "
                    f"same order as the reasons below. Do not include any explanation.\n\n"
                    f"Assess each reason based on how applicable it is to the query-entity relationship.\n"
                    f"Search Query: {query_text}\n"
                    f"Entity Title or ID: {docid}\n"
                    f"Reasons: {reasons}\n"
                    "Labels:"
                )
            }

            messages = [system_prompt, user_prompt_description]

            # 4. Call the appropriate model via ollama.chat()
            try:
                if model_name == "llama4":
                    response: ChatResponse = chat(
                        model="hf.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q2_K_XL",
                        messages=messages,
                        options={"temperature": 0}
                    )
                elif model_name == "qwen3":
                    response: ChatResponse = chat(
                        model="qwen3:32b",
                        messages=messages,
                        options={"temperature": 0}
                    )
                else:
                    raise ValueError(f"Model '{model_name}' not supported.")

                # 5. Extract only the bracketed list of 0/1 labels
                reply = response.message.content.strip()
                data = reply.split("</think>")[-1].strip()
                labels = ast.literal_eval(data)

                if len(labels) != len(reasons):
                    raise ValueError(
                        f"Expected {len(reasons)} labels but got {len(labels)}"
                    )

                # 6. Append to output file
                # Format: 0 <tab> qid <tab> docid <tab> query <tab> [labels]
                with open(output_file, "a") as out_f:
                    out_f.write(f"0\t{qid}\t{docid}\t{query_text}\t{labels}\n")

                # Print a brief confirmation (first three labels shown)
                print(f"[{vote_count} LLMs] {qid} - {docid} → {labels[:3]}...")

            except Exception as e:
                print(f"Error on ({qid}, {docid}): {e}")
                # Skip this pair on failure
                continue

        print(f">>> Saved one-shot labels to: {output_file}")

if __name__ == "__main__":
    main()
