import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from .llm_breakpoint_sentence import breakpoint_sentence_chunking
from .llm_full_chunking import full_text_chunking

load_dotenv()


def chunker(method, files):
    """
    Determines the chunking method to be used based on the `method` argument
    and calls the appropriate chunking function.

    Returns:
    list: A list of chunks produced by the specified chunking method.
    """
    if method == "agentic_breakpoint":
        return agentic_breakpoint_chunker(files)
    if method == "agentic_full_text":
        return agentic_full_text_chunker(files)
    if method == "semantic":
        return semantic_chunker(files)
    if method == "fixed_length":
        return fixed_length_chunker(files)


def agentic_full_text_chunker(files):
    """
    Uses agentic_full_text chunking to split the input text files into chunks using a
    custom full operation function.

    Returns:
    list: A list of text chunks produced by the agentic_breakpoint chunker.
    """
    return full_text_chunking(files)


def agentic_breakpoint_chunker(files):
    """
    Uses agentic_breakpoint chunking to split the input text files into chunks using a
    custom full operation function.

    Returns:
    list: A list of text chunks produced by the agentic_breakpoint chunker.
    """
    return breakpoint_sentence_chunking(files)


def semantic_chunker(files):
    """
    Uses semantic chunking with OpenAI embeddings to split the input text files
    into chunks based on semantic similarity.

    Parameters:
    files (str): The input text files to be chunked.

    Returns:
    list: A list of semantically meaningful chunks produced by the chunker.
    """
    openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.environ['OPENAI_KEY'])
    text_splitter = SemanticChunker(embeddings=openai_embeddings, breakpoint_threshold_type='gradient', breakpoint_threshold_amount=97.5)
    docs = text_splitter.create_documents([files])

    chunks = []
    for doc in docs:
        chunks.append(doc.page_content)
    return chunks


def fixed_length_chunker(files, chunk_size=1024, chunk_overlap=20):
    """
    Uses fixed-length chunking to split the input text files into chunks of a
    specified size with optional overlap.

    Parameters:
    files (str): The input text files to be chunked.
    chunk_size (int): The maximum number of words in each chunk. Default is 1024.
    chunk_overlap (int): The number of overlapping words between consecutive chunks. Default is 20.

    Returns:
    list: A list of fixed-length text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
        is_separator_regex=False,
    )

    documents = text_splitter.create_documents([files])
    chunks = [doc.page_content for doc in documents]
    return chunks


if __name__ == "__main__":

    with open('../../data/paul_graham_essays.txt', 'r') as file:
        full_input_text: str = file.read()

    chunks = semantic_chunker(full_input_text)
