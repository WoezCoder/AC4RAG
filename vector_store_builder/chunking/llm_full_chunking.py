import os
import ast
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import nltk
import tiktoken
import openai  # Import OpenAI for exception handling
from tqdm import tqdm

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.environ['OPENAI_KEY'])

system_prompt = """
    You are an advanced language model tasked with semantically splitting up documents.
    Your goal is to divide the FULL input text into meaningful, coherent chunks. That is: aim to find chunks which are semantically apart from each other. 
    Try to find chunks of between 750 and 1500 words. 
    Return the output as list. 

    ################################

    Example Output:

    {{
        "Chunk 1 text": "first chunk",
        "Chunk 2 text": "second chunk",
        "Chunk 3 text": "third chunk"
    }}

    Now here is the input text:

    {input}
"""


def clean_response(response_content):
    """Remove backticks and the "json" label."""
    cleaned_content = response_content.strip().strip('```').replace('json\n', '').strip()
    return cleaned_content


def split_text_to_chunks(large_string, response_content):
    """
    Extract chunks of text based on the response content and the original large string.

    Returns:
        list: A list of extracted text chunks.
    """
    json_data = json.loads(response_content)
    extracted_chunks = []
    for chunk in json_data:
        start = chunk['chunk_text']['start']
        end = chunk['chunk_text']['end']

        start_index = large_string.find(start)
        end_index = large_string.find(end, start_index + len(start))

        if start_index != -1 and end_index != -1:
            extracted_chunk = large_string[start_index:end_index + len(end)]
            extracted_chunks.append(extracted_chunk.strip())

    return extracted_chunks


def run_chunker(input):
    """
    Process the input text to split it into meaningful chunks using the language model.

    Returns:
        list: A list of chunks if successful; otherwise, None.
    """
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    response = chain.invoke({'input': input})

    json_data = clean_response(response.content)
    json_data = json.loads(json_data)

    chunks = []

    if response and response.content:
        for chunk in json_data.values():
            chunks.append(chunk)
        return chunks
    else:
        print("Received empty response from the model.")
        return None


def split_text_into_input_texts(text, max_tokens=3500):
    """
    Split the large text into smaller texts suitable for processing by the language model, such that the max context
    window is satisfied.

    Returns:
        list: A list of smaller text chunks, each within the token limit.
    """
    tokenizer = tiktoken.get_encoding('cl100k_base')

    sentences = nltk.sent_tokenize(text)

    input_texts = []
    current_input_text = []
    current_input_text_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_token_count = len(sentence_tokens)

        if current_input_text_token_count + sentence_token_count > max_tokens:
            input_texts.append(tokenizer.decode(current_input_text))
            current_input_text = sentence_tokens
            current_input_text_token_count = sentence_token_count
        else:
            current_input_text.extend(sentence_tokens)
            current_input_text_token_count += sentence_token_count

    if current_input_text:
        input_texts.append(tokenizer.decode(current_input_text))

    return input_texts


def full_text_chunking(files):
    """
    Run the full chunking operation.

    Returns:
        list: A list of chunks resulting from the text processing.
    """
    inputs = split_text_into_input_texts(files)
    full_chunk_list = []

    for input in tqdm(inputs):
        chunks = run_chunker(input)
        if chunks:
            full_chunk_list.extend(chunks)
        else:
            print("Error processing chunk.")

    return full_chunk_list
