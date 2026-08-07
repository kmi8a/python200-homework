from openai import OpenAI
from dotenv import load_dotenv
import json

# --- Completions API --

# API Q1

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f'Response text: {response.choices[0].message.content}')
print(f'Model used:{response.model}')
print(f'Total tokens: {response.usage.total_tokens}')

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "user",
                                                          "content":prompt}],
                                                          temperature= temp)
    print(f'Response text (temp: {temp}): {response.choices[0].message.content}')

# As the temperature gets bigger, the output gets more creative.
# I would use 0.7 or lower as a setting for consistent reproducible output.


# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, n in enumerate(response.choices, start=1):
    print(f'Responce choice {i}: {n.message.content}')

# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print(f'Response text: {response.choices[0].message.content}')

# Output was cut short due to the 'max_tokens' limit set up.
# You would like to use this limitation for budget control, or for UI/UX limitations.

# --- System Messages and Personas --

# System Q1

# personality 1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system",
         "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
        },
        {"role": "user",
         "content": "I don't understand what a list comprehension is.",
        },
    ],
)

print(f'Personality 1: {response.choices[0].message.content}')

# personality 2

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system",
         "content": "You are an hyper-enthusiastic, high-energy tech cheerleader who treats every programming concept like it's the greatest discovery in human history. Use lots of exclamation points, caps for emphasis, and pure unbridled excitement, always ending with a massive pump-up cheer!",
        },
        {"role": "user",
         "content": "I don't understand what a list comprehension is.",
        },
    ],
)

print(f'Personality 2: {response.choices[0].message.content}')

# you can really notice the influence of the pesonality definition between the two responses.

# System Q2

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Jordan and I'm learning Python."},
        {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
        {"role": "user", "content": "Can you remind me what my name is?"}
    ]
)

print(f'Q2 model answer: {response.choices[0].message.content}')

# The model knows Jordan's name because it was on the list of messages sent on the API call.

# --- Prompt Engineering --

def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

# Prompt Q1 — Zero-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
    Classify the sentiment of each review below as positive, negative, or mixed.
    Output each classifiction on a new line.

    Reviews:
    1. {reviews[0]}
    2. {reviews[1]}
    3. {reviews[2]}
"""

result = get_completion(prompt)
print(result) 


# Prompt Q2 — One-Shot

prompt = f"""
    Classify the sentiment of each review below as positive, negative, or mixed.
    Example:
        Review: "Fast shipping but the item arrived damaged."
        Sentiment: mixed
    Output each classifiction on a new line.

    Reviews:
    1. {reviews[0]}
    2. {reviews[1]}
    3. {reviews[2]}
"""

result = get_completion(prompt)
print(result) 

# Prompt Q3 — Few-Shot

prompt = f"""
    Classify the sentiment of each review below as positive, negative, or mixed.
    Example 1:
        Review: "Fast shipping but the item arrived damaged."
        Sentiment: mixed
    Example 3:
        Review: "Fast shipping and top notch customer service."
        Sentiment: positive
    Example 2:
        Review: "Unresponsive customer support, the item was never delivered!"
        Sentiment: negative
    Output each classifiction on a new line.

    Reviews:
    1. {reviews[0]}
    2. {reviews[1]}
    3. {reviews[2]}
"""

result = get_completion(prompt)
print(result) 

# 1. Zero-Shot Prompting:
#    - Choose for straightforward, general tasks where the model already has strong baseline
#      knowledge (e.g., standard sentiment classification, basic translation, summarizing text).
#    - Pros/Cons: Fast and uses the fewest tokens, but less reliable for specialized formatting or nuanced rules.
#
# 2. One-Shot Prompting:
#    - Choose when you need to guide the model toward a very specific output format, 
#      style, or classification that a zero-shot prompt might misinterpret.
#    - Pros/Cons: Balances token efficiency with a clear demonstration of expectations.
#
# 3. Few-Shot Prompting:
#    - When to choose: For complex, ambiguous, or highly specialized tasks.
#    - Pros/Cons: Highly accurate and reliable at enforcing consistency, but consumes more tokens and costs more per request.

# Prompt Q4 — Chain of Thought

prompt = """
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?

Please solve this problem by showing your reasoning step by step before giving a final answer. Clearly label the final answer.
"""

result = get_completion(prompt)
print(result)

# Why asking the model to reason step by step improves accuracy on problems like this?:
# It splits a multi-step problem into smaller, manageable chunks, reducing the chances of hallucinations, errors and the posibility of context overload.

# Prompt Q5 — Structured Output

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the review below and return the result only as valid JSON.
Do not include any markdown formatting blocks (like ```json), commentary, or extra text outside the JSON object.
The JSON must have the following keys:
- sentiment (string: positive, negative, or mixed)
- confidence (float from 0 to 1)
- reason (string: exactly one sentence)

Review: {review}
"""

result = get_completion(prompt)
print(result)

try:
    parsed_data = json.loads(result)

    print(f"Sentiment: {parsed_data.get('sentiment')}")
    print(f"Confidence: {parsed_data.get('confidence')}")
    print(f"Reason: {parsed_data.get('reason')}")

except json.JSONDecodeError:
    print("Error: Failed to parse response as valid JSON.")
    print(f"Raw response for debugging:{result}")


# Prompt Q6 — Delimiters

# text with instructions

user_text = """First boil a pot of water. Once boiling, add a handful of salt and the
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."""

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

result = get_completion(prompt)
print(result)

# text without instructions

user_text = """i want to travel all over the world so i can meet new people, new cultures, new foods and learn new languages"""

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

result = get_completion(prompt)
print(result)

# Delimiters help prevent prompt injection, it also helps creating boundaries for the data to be processed.


# --- Local Models with Ollama --

# Ollama Q1

# Ollama output

"""
Ollama Terminal Output:
Thinking...
Okay, the user wants me to explain what a large language model is in two sentences. Let me start by recalling what I know about them. First, large language models are big artificial intelligence models. They can 
understand and generate text, right? They're used in various fields like AI, NLP, etc.

Wait, but how to make that two sentences? Maybe start with the definition. "A large language model is a type of artificial intelligence that can understand and generate human-like text, making it useful for 
various tasks such as language processing and creative writing." Then, add another sentence to expand. Maybe mention their capabilities like summarizing or answering questions. So, "These models are trained on 
massive datasets and can handle complex tasks, making them powerful tools in various fields including science, technology, and daily life." That should cover it in two sentences.
...done thinking.

A large language model is a type of artificial intelligence that can understand and generate human-like text, enabling tasks such as language processing and creative writing. These models are trained on massive 
datasets and can handle complex tasks, making them powerful tools in various fields like science, technology, and daily life.
"""

#OpenAI Output

prompt = "Explain what a large language model is in two sentences."

result = get_completion(prompt)
print(f'OpenAI response: {result}')

# the definition by the OpenAI model is more technical while the definition by the Ollama model uses more accesible vocabulary.
# Running a model locally has the advantage of not having any attached costs to it, the main disadvantage of running a model locally would limitations on the processing power.