# -- ML vs. LLM in Pipelines --

# Q1:
# In a comment block, explain the difference between what the ML classifier produces and what the LLM produces in this week's pipeline.
# Why does each tool do what it does? What would go wrong if you tried to swap them — using the LLM to make the binary good/skip prediction
# and the ML model to write the recommendation?

# Answer:
# the ML classifier produces a binary classification while the LLM produces a short phrase that summarises the conclusions on the data. the ML model is good
# for deterministic outputs, while the LLM is good for language generation, if we swapped them, the binary prediction would take much longera, the costs
# associated with the API usage would be much higher, and we would risk getting non deterministic answers from the LLM, meanwhile using the ML model to process
# natural langage would simply fail.

# Q2:
# For each task below, write one sentence in a comment block stating whether you would use a trained ML model, an LLM, or deterministic code, and why:

# 1. Converting a date string like "2023-07-04" to day-of-week
# 2. Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# 3. Predicting customer churn given 15 numeric features and a labeled training dataset
# 4. Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# 4. Summing a column of revenue figures

# Answer:
# 1. Deterministic code, as it excels with fixed, unambiguous rules.
# 2. ML model because text classification over predefined categories maps directly to supervised learning strengths.
# 3. ML model as structured tabular data with a labeled training dataset is the standard use case for supervised algorithms.
# 4. LLM because it naturally handles semantic variations, abbreviations, and messy formatting without rigid manual rules.
# 5. Deterministic code, because basic arithmetic requires absolute mathematical precision.

# Q3
# what is incremental processing, and why is it important for this pipeline?
# What would happen — in terms of cost and data correctness — if the transform script re-processed all 365 records every time it ran?

# Answer:
# Incremental processing is when a script run multiple times,processes only the data that is new, modified or that has not been processed yet.
# without this feature, everytime that the script has to re-run would means starting from scratch, losing valuable time,
# driving up the costs of the operation, and compromising the integrity of the data with duplicate entries.

# -- Prompt Design --

# Q1:
# The lesson prompt asks the LLM for exactly one sentence. Write an alternative system prompt that asks for a two-sentence recommendation where 
# the first sentence states the prediction and the second sentence explains the reasoning. 
# In a comment, describe: what would you need to change in the validation logic to accommodate two sentences instead of one?

# Answer:

SYSTEM_PROMPT = (
    "You are writing a two-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction about whether the day is good for running. "
    "The first sentence should state the prediction, direct, practical, and specific to the conditions. "
    "The second sentence should explain the reasoning behind the prediction."
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

# to update the validation logic to accomodate two sentences instead of one we would need to parse the response, split it into sentences and
# validate that there is two sentences available, add a positional check to make sure sentence 1 contains the prediction and sentence 2 contains
# the reasonning.

# Q2
# Write a function call_with_retry(client, messages, max_retries=3) that calls client.chat.completions.create() and retries up to max_retries times
# on any exception, with a 2-second wait between attempts. On final failure, return None. 
# In a comment, describe when you would use this in a production pipeline.

# Answer:

import time

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

# Production use case: 
# I would use this in a production pipeline to handle network glitches or API outages, preventing single intermittent errors
# from crashing the entire pipeline.