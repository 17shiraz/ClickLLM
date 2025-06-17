import os
import json
import re
import openai
import ast
import pickle
import pandas as pd
import random
import argparse
from ollama import chat
from ollama import ChatResponse 
random.seed(42)

def main(dataset, model_name, openai_api_key=None):
    # Set to True to use abstracts, False to use entity titles (docid)
    use_abstract = True
    if model_name not in  ['llama4','qwen3']:
        raise ValueError("Model not available")

    for year in dataset:
        # === Config ===
        queries_path = f'/mnt/data/msaliminabi/entity/llmrel/dbpedia/queries-v{year}.txt'
        qrels_path = f'/mnt/data/msaliminabi/output.txt'
        N = 50 # Set to None to load all samples
        
        # Modify output filename based on mode
        mode_suffix = "abstract" if use_abstract else "entity_title"
        output = open(f'/mnt/data/msaliminabi/entity/llmrel/gold/graded_{model_name}_{mode_suffix}_gold_standard_50.txt', 'a')

        # === Load queries ===
        queries = {}
        with open(queries_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid, qtext = parts[0], parts[1]
                    queries[qid] = qtext

        qid_list = list(queries.keys())
        if N is not None:
            qid_list = random.sample(qid_list, min(N, len(qid_list)))

        # === Load qrels as docid list ===
        qrels_df = pd.read_csv(qrels_path, sep='\t', header=None)
        qrels_df.columns = ['qid', '_', 'docid','abstract', '_']
        qrels_grouped = qrels_df.groupby('qid')['docid'].apply(list).to_dict()

        # === Build text_col dict ===
        all_docids = qrels_df['docid'].unique()
        
        if use_abstract:
            # Use abstracts from qrels file
            text_col = {}
            for _, row in qrels_df.iterrows():
                docid = row['docid']
                abstract = row['abstract']
                text_col[docid] = abstract
        else:
            # Use docid as entity title (original behavior)
            text_col = {docid: docid for docid in all_docids}

        for qid in qid_list:
            if qid not in queries:
                continue
            query = queries[qid]
            pos_entities = qrels_grouped.get(qid, [])
            for docid in pos_entities:
                if docid not in text_col:
                    continue  # Skip if abstract not found

                #doctext = text_col[docid]

                messages = [
                    {
                        "role": "system",
                        "content": (
                    "Given a query and the abstract of a knowledge entity, you must choose one option:\
                    0 : the entity seems irrelevant the query, \
                    1 : the entity seems relevant to the query and but does not directly match it, \
                    2 : the entity seems highly relevant to the query or is an extact match .\
                    Break down each query into these steps:\
                            Consider what information the user is likely searching for with the query.\
                            Measure how well the abstract matches a likely intent of the query (M) as an integer 0-2.\
                            Assess whether the entity matches any reasonable interpretation of the query (I) as an integer 0-2\
                            Consider the aspects above and the relative importance of each, and decide on a final score (O) as an integer 0-2.\
                    "
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f" \
                            Query: {queries[qid]}\n \
                            Entity: {text_col[docid]} \n \
                            IMPORTANT: Your response must only be in the format of ""Final score: #""\
                            Relevant?\
                            \
                            "
                        )
                    }
                ]
                try:
                    if model_name =='llama4':
                        response: ChatResponse = chat(model='hf.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q2_K_XL', 
                        messages = messages,options={ "temperature": 0})
                    elif model_name =='qwen3':
                        response: ChatResponse = chat(model='qwen3:8b', 
                        messages = messages,options={ "temperature": 0})
                        system_prompt = 1
                        user_prompt = system_prompt
                        system_prompt = user_prompt
                except:
                    continue
                if model_name =='llama4':
                    rate = response.message.content  
                elif model_name =='qwen3':
                    rate = response.message.content  

                match = re.search(r'final score:\s*(\d)', rate.lower())
                if match:
                    rate = int(match.group(1))
                else:
                    # Fallback or error handling
                    rate = 0 
                print('binary', model_name, year, qid, docid, rate)
                output.write(f"{qid} 0 {docid} {rate}\n")
        output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', nargs='+', type=str, default=['20','21','antique','19'], help='name the collections')
    parser.add_argument('--model_name', type=str, default='qwen3', help='Model name to use')
    #parser.add_argument('--api_key', type=str, default=key,help='OpenAI API key for model gpt-4o')

    args = parser.parse_args()
    
    main(dataset=args.dataset, model_name=args.model_name)  # , api_key=args.api_key)