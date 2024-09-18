import os
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_distances
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key= os.environ['OPENAI_KEY'])


def get_last_n_sentences(text, num_sentences):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[-num_sentences:])


def get_first_n_sentences(text, num_sentences):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:num_sentences])


def calculate_cosine_distance(emb1, emb2):
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)
    return cosine_distances(emb1, emb2)[0][0]


def calculate_embeddings(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_distances_to_next(file_path):
    """
    Calculate the cosine distance between the last and first sentences of consecutive chunks in a document for 1 and 5
    sentences.

    Updates:
        The JSON file will be updated with new fields for cosine distances.
    """
    with open(file_path, 'r') as file:
        source_documents = json.load(file)

    n_sent = [1, 5]

    for n in n_sent:
        for i in tqdm(range(len(source_documents) - 1), desc=f"Processing {file_path}"):
            current_sentences = get_last_n_sentences(source_documents[i]['content'], n)
            next_sentences = get_first_n_sentences(source_documents[i + 1]['content'], n)

            current_vector = calculate_embeddings(current_sentences)
            next_vector = calculate_embeddings(next_sentences)

            distance = calculate_cosine_distance(current_vector, next_vector)
            source_documents[i][f'cosine_distance_to_next_{n}'] = distance

    for i in tqdm(range(len(source_documents) - 1), desc=f"Processing {file_path}"):
        current_vector = calculate_embeddings(source_documents[i]['content'])
        next_vector = calculate_embeddings(source_documents[i + 1]['content'])
        distance = calculate_cosine_distance(current_vector, next_vector)
        source_documents[i][f'cosine_distance_to_next_full_text'] = distance

    with open(file_path, 'w') as f:
        json.dump(source_documents, f, indent=4)


if __name__ == "__main__":
    document_types = [
        '../../data/agentic_full_text/chunks_agentic_full_text.json',
        '../../data/fixed_length/chunks_fixed_length.json',
        '../../data/agentic_breakpoint/chunks_agentic_breakpoint.json',
        '../../data/semantic/chunks_semantic.json'
    ]

    for file_path in document_types:
        get_distances_to_next(file_path)
