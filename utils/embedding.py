import openai
import numpy as np
from joblib import Memory

# Caching directory
memory = Memory("cache_dir", verbose=0)

client = openai.OpenAI()

@memory.cache
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

@memory.cache
def get_batch_embeddings(chunks, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=chunks, model=model)
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [r.embedding for r in sorted_data]

