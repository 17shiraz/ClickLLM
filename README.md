# ClickLLM

## Scripts
Our project files consist of three stages: Relevance Judgement, Reason Nuggetization, and Label Assignment
### Relevance Judgement

#### Dbpedia
The script ``gold_standard_abstract_graded.py`` prompts an LLM of choice ``Qwen3:8b`` or ``LLama4:Scout`` to perform a judgement task on Dbpedia query-entity pairs.
The ``use_abstract`` flag determines whether the script uses **entity titles** or **entity abstracts** for the judgement task.

#### LaQuE
``laque_graded.py``

### Reason Nuggetization
``laque_analysis.py

### Label Assignment
``laque_analysis_assigner.py``

## Prompt
```
Given a query and the abstract of a knowledge entit you must choose one option:\n"
  0: The entity seems irrelevant to the query.\n"
  1: The entity seems relevant to the query but does not directly match it.\n"
  2: The entity seems highly relevant to the query or is an exact match.\n\n"
Break down each query into these steps:\n"
  1. Consider what information the user is likely searching for with the query.\n"
  2. Measure how well the abstract matches a likely intent of the query (M), scored 0–2.\n"
  3. Assess whether the entity matches any reasonable interpretation of the query (I), scored 0–2.\n"
  4. Based on M and I, decide on a final score (O), scored 0–2.\n"

"Query: {}\n"
"Entity: {}\n"
"IMPORTANT: Your response must only be in the format of "Final score: #" \n"
"Relevant?"
 ```
