import os
import random

import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


load_dotenv()

# Initialize OpenAI Chat model
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.environ['OPENAI_KEY'])

system_prompt = """

Your task is to write a single factual question and answer given a context.
Your factual question should be answerable with a specific, concise piece of factual information from the context.
Your factual question should be formulated in the same style as questions users might ask in a search engine.
Your factual question should not be too long.
This means that your factual question should NOT contain anything like "according to the passage" or "context".

Additionally, return the answer to the factual question. 
Also, return the exact sentence in the text on which the QUESTION is based. Don't add text on which the answer is based.
    
"""


class QuestionGeneration(BaseModel):
    """
    This class defines the schema for the generated output, including the factual question, answer,
    and the exact sentence from the context on which the question is based, ensuring a structured output.
    """
    factual_question: str = Field(description="The factual question given the context.")
    answer: str = Field(description="The answer to the factual question.")
    exact_text: str = Field(description="The exact sentence in the text on which the answer is based.")


def generate_output(input_text):
    """
    Make the API call to the model. Given an input text (context), generates a factual question, its answer, and the
    exact sentence from the context.

    :return: A tuple containing the factual question, answer, and the exact sentence.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input_text}")
    ])

    chain = prompt | llm.with_structured_output(QuestionGeneration)
    output = chain.invoke({"input_text": input_text})
    return output.factual_question, output.answer, output.exact_text


def get_random_docs(n_samples):
    """
    Randomly extracts 500-word samples from a text file containing Paul Graham's essays.

    :return: A list of text samples, each containing 500 words.
    """
    with open('../data/paul_graham_essays.txt', 'r') as file:
        text: str = file.read()

    words = text.split()
    sampled_docs = []

    for i in range(n_samples):
        start_index = random.randint(0, len(words) - 501)
        upcoming_words = words[start_index:start_index + 500]

        result_string = ' '.join(upcoming_words)
        sampled_docs.append(result_string)
    return sampled_docs


def generate_dataset(n_samples):
    """
    Generates a dataset of factual questions, answers, and exact text segments based on random 500-word samples from a
    text file.

    :return: A pandas DataFrame containing the generated questions, answers, and the exact sentences from the context.
    """
    dataset = pd.DataFrame(columns=["question", "answer", "exact_text"])
    docs = get_random_docs(n_samples)
    with tqdm(total=len(docs), desc="Generating Questions") as pbar:
        for doc in docs:
            question, answer, exact_text = generate_output(doc)
            dataset = dataset._append({
                "context": doc,
                "question": question,
                "answer": answer,
                "exact_text": exact_text
            }, ignore_index=True)
            pbar.update(1)
    return dataset

