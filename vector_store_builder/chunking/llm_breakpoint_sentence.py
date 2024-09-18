import os
import re
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

# Initialize OpenAI Chat model
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.environ['OPENAI_KEY'])


system_prompt = """

The presented text needs to be split into two parts at a single point. 
Your task is to find the optimal breakpoint in the text where the two resulting sections are semantically as distinct 
from each other as possible. If possible the sections should be understandable without each other. To do this:

Read the entire text carefully.
Analyze the content and meaning of different sections.
Identify the single point in the text where dividing it into two parts would maximize the semantic difference between the two chunks.
Consider factors like shifts in topic, differences in subject matter, or transitions in context that might cause the two sections to have distinct meanings.
Do not start the new chunk with an answer to a question of the previous chunk. 
Your task is to return **the first sentence of the second chunk**, while keeping the sentence EXACTLY the same. This sentence will serve as the boundary where the text is split.

Key instructions:
1. The breakpoint should occur at a sentence that contains **at least 5 words**.
2. **Do not alter** the content of any sentence in the input text.
3. Return **only full sentences**, and avoid splitting within a sentence.
4. The final sentence of the input text **cannot** serve as the breakpoint.
5. Aim to find breakpoints such that the two texts are semantically as distinct as possible. 
6. If there is a spelling mistake in one of the words or characters which should not be there or punctuation is missing or capitalization is not correct, do not fix that in your response. Keep it exactly as it is. 

"""


class BreakpointSentence(BaseModel):
    breakpoint_sentence: str = Field(description="The first sentence after the point where the input text is split.")


def run_chunker(input_text: str):
    """
    Run the chunking model on the input text to generate semantically meaningful chunks using structured output.

    Returns:
        A list of chunks extracted from the input text.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input_text}")
    ])

    chain = prompt | llm.with_structured_output(BreakpointSentence)
    breakpoint = chain.invoke({"input_text": input_text})
    return breakpoint.breakpoint_sentence


def filter_words(remaining_text_list, breakpoint_sentence):
    """
    Filters the text to find the correct split point based on the breakpoint sentence.

    Args:
        remaining_text_list (list): List of words from the remaining text.
        breakpoint_sentence (list): List of words from the detected breakpoint sentence.

    Returns:
        tuple: Two lists - the first contains the chunk before the breakpoint, and the second contains the chunk after.
    """
    len_long = len(remaining_text_list)
    len_short = len(breakpoint_sentence)

    for i in range(len_long - len_short + 1):
        sublist = remaining_text_list[i:i + len_short]

        # Calculate the allowed mismatches based on length of breakpoint_sentence
        if len_short <= 4:
            tolerance = 0  # Allow at most 1 mismatch
        elif len_short <= 7:
            tolerance = 0  # Allow 1 or 2 mismatches depending on length 5 or 6
        else:
            tolerance = 0  # Allow more mismatches as length grows

        mismatches = sum(1 for a, b in zip(sublist, breakpoint_sentence) if a != b)

        if mismatches <= tolerance:
            return remaining_text_list[:i], remaining_text_list[i:]

    return [], remaining_text_list


def breakpoint_sentence_chunking(full_text):
    """
    Splits the full text into chunks based on semantic breakpoints identified. This is the full chunking operation.

    Returns:
        list: A list of text chunks separated at the optimal semantic breakpoints.
    """
    remaining_text_list = re.split(r'\s+', full_text)
    input_text = ' '.join(remaining_text_list[:2000])
    chunks = []

    while len(remaining_text_list) > 2000:
        breakpoint_sentence = run_chunker(input_text)
        breakpoint_sentence_list = re.split(r'\s+', breakpoint_sentence)
        chunked_text_list, remaining_text_list = filter_words(remaining_text_list, breakpoint_sentence_list)
        input_text = ' '.join(remaining_text_list[:2000])
        chunks.append(' '.join(chunked_text_list))
    if remaining_text_list:
        chunks.append(' '.join(remaining_text_list))

    return chunks


if __name__ == "__main__":
    with open('../../data/paul_graham_essays.txt', 'r') as file:
        full_input_text = file.read()

    chunks = breakpoint_sentence_chunking(full_input_text)
