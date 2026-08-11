# --- Step 1: Setup ---

from dotenv import load_dotenv
import os
import string
from pathlib import Path
from pypdf import PdfReader
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# --- Step 2: Load the Documents ---

docs = SimpleDirectoryReader(docs_dir).load_data()

print(f'Documents loaded: {len(docs)}')

for d in docs:
    print(d.metadata.get("file_name", "Unknown file"))


# --- Step 3: Build the Index and Query Engine ---

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

if index:
    print('Index built successfully. Ready to answer questions.')


# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes[:1]:
        print(f"Document Name: {node_with_score.node.metadata.get('file_name', 'Unknown Document')}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

# Question: did the assistant sound confident and accurate? Did any of the answers surprise you?
# Answer: Yes the assistant sounded very confident, accurate and gave short and to-the-point answers, none of the answers really surprised me.

# --- Step 5: Find a Failure ---

q2 = 'If working as a barista, what benefits are offered as an employee of Groundwork Coffee?'

query_engine = index.as_query_engine(similarity_top_k=3)
print(f"\nQ: {q2}")
response = query_engine.query(q2)
print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Document Name: {node_with_score.node.metadata.get('file_name', 'Unknown Document')}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

# Question: What you asked and why you expected it to be hard.
# Answer: I asked 'If working as a barista, what benefits are offered as an employee of Groundwork Coffee?', and expected it to be hard because
# the information is not anywhere on the documents provided.

# Question: What went wrong — wrong retrieval, missing information, the model guessed anyway?
# Answer: missing information, and because of that the model retrieved the wrong text, taking a description of the company ethic and vision and turning it into the answer

# Question: When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
# Answer: the model kept it's confident tone, even tho the informatin was not right, this suggest that AI generated content con be misleading and that we have
# to always review and confront it with external information on the subject, we cannot take ai generated content as the only source of information.

# Question: What you would change about the system to improve it
# I would train the model to simply state that it doesnt knows the answer when it is the case, that ends up being way more helpful.


# --- Step 6: Reflection ---

# Question: The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code.
# How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?
# Answer: the Llama Index implementation took just 2 lines of code, this reduction in the amount of code demonstrates that using a framework to abstract complex,
# repetitive tasks allows developers to focus in application of the logic instead of focusing and spending so much time in theimplementation.

# Question: You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would
# add genuine value to a business or organization.
# Answer: These system can be used by any business to provide interactive information to it customers in a manner that is  instant, accurate and backed by exact references
# to official company documents.

# Question: What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?
# Answer: a RAG pipeline cannot fully prevent hallucination.