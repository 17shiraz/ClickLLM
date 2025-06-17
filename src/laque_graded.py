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
    if model_name not in  ['llama4','qwen3']:
        raise ValueError("Model not available")

    for year in dataset:

        # === Config ===
        pkl_path = f'/mnt/data/msaliminabi/entity/llmrel/laque_data/triples.run.queries.train.{year}.bm25.top1000.tsv.pkl'
        abstract_path = f'/mnt/data/msaliminabi/entity/llmrel/laque_data/LaQuE_collection.tsv'
        N = 15000 # Set to None to load all samples
        output = open(f'graded_{model_name}_laque_abstract_15k.txt', 'a')

        # === Load triple_dic.pkl ===
        with open(pkl_path, 'rb') as f:
            triple_dic = pickle.load(f)

        # === Select N samples or all ===
        qid_list = list(triple_dic.keys())
        if N is not None:
            qid_list = random.sample(qid_list, min(N, len(qid_list)))

        # === Build queries dict ===
        queries = {qid: triple_dic[qid]['query'] for qid in qid_list}

        # === Load abstract text file into dict ===
        abstract_df = pd.read_csv(abstract_path, sep='\t', header=None, names=['docid', 'abstract'])
        text_col = dict(zip(abstract_df['docid'], abstract_df['abstract']))
        #text_col = dict(
        #    zip(
        #        abstract_df['docid'],
        #        [f"Title: {title}\nAbstract: {abstract}" for title, abstract in zip(abstract_df['docid'], abstract_df['abstract'])]
        #    )
        #)

        for qid in qid_list:
            query = queries[qid]
            pos_entities = triple_dic[qid].get('pos', [])
            neg_entities = triple_dic[qid].get('neg', [])
            all_entities = pos_entities + neg_entities

            for docid in pos_entities:
                if docid not in text_col:
                    continue  # Skip if abstract not found

                #doctext = text_col[docid]

                messages = [
                    {
                        "role": "system",
                        "content": (
                    "Given a query and the abstract of a knowledge entity, you must choose one option:\
                    0 : that the entity seems irrelevant the query, \
                    1 : that the entity seems relevant to the query but does not directly match it, \
                    2 : that the entity seems highly relevant to the query or is an extact match .\
                    Break down each query into these steps:\
                            Consider what information the user is likely searching for with the query.\
                            Measure how well the abstract matches a likely intent of the query (M) as an integer 0-2.\
                            Assess whether the entity matches any reasonable interpretation of the query (I) as an integer 0-2\
                            Consider the aspects above and the relative importance of each, and decide on a final score (O) as an integer 0-2.\
                    /No_think\
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
    parser.add_argument('--dataset',nargs='+', type=str, default=['20','21','antique','19'], help='name the collections')
    parser.add_argument('--model_name', type=str, default='qwen3', help='Model name to use')
    #parser.add_argument('--api_key', type=str, default=key,help='OpenAI API key for model gpt-4o')

    args = parser.parse_args()

    main(dataset=args.dataset, model_name=args.model_name)#, api_key=args.api_key)