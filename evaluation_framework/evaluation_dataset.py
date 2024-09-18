import os
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

load_dotenv()

# Initialize OpenAI Chat model
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.environ['OPENAI_KEY'])

evaluation_prompt = """
    ###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, 
    and a score rubric representing a evaluation criteria are given.
    
    Write a score that is an integer between 1 and 5. You should refer to the score rubric.

    ###The instruction to evaluate:
    {instruction}

    ###Response to evaluate:
    {response}

    ###Reference Answer (Score 5):
    {reference_answer}

    ###Score Rubrics:
    [Is the response correct, accurate, and factual based on the reference answer?]
    Score 1: The response is completely incorrect and/or inaccurate. ("Unfortunately, I cannot help you with this.")
    Score 2: The response is mostly incorrect and/or inaccurate.
    Score 3: The response is somewhat incorrect and/or inaccurate.
    Score 4: The response is mostly correct and accurate.
    Score 5: The response is completely correct and accurate.
    """


class Tagging(BaseModel):
    """This class is used to define a schema for tagging the ratings of the answers, ensuring a structured output."""
    rating: Optional[str] = Field(
        description="Assign the answer the score which it belongs to. 1 is the lowest, 5 is the highest.",
        enum=['1', '2', '3', '4', '5']
        )


def run_evaluation(question, answer_truth, rag_answer):
    """
    Evaluate the response based on the instruction, reference answer, and provided response using a language model.

    Returns:
        str: The evaluation rating assigned to the response, from '1' to '5'.
    """
    prompt = ChatPromptTemplate.from_template(evaluation_prompt)
    chain = prompt | llm.with_structured_output(Tagging)
    response = chain.invoke({'instruction': question, 'response': rag_answer, 'reference_answer': answer_truth})
    return response.rating


def run_llm_evaluation(dataframe):
    """
    Evaluate responses in a dataframe using the language model.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'llm_evaluation' containing the evaluation scores.
    """
    dataframe['llm_evaluation'] = " "

    with tqdm(total=len(dataframe), ncols=100, desc='LLM Evaluation', position=0) as pbar:
        for index, row in dataframe.iterrows():
            dataframe.loc[index, "llm_evaluation"] = run_evaluation(row['question'], row['answer'], row['RAG_answer'])
            pbar.update(1)
    return dataframe


def check_search(retrieved_doc_ids, exact_text, source_documents):
    """
    Check if the exact text is found in any of the retrieved documents.

    Returns:
        bool: True if the exact text is found in any of the retrieved documents, otherwise False.
    """
    for item in source_documents:
        for id in retrieved_doc_ids:
            if item['id'] == id:
                if exact_text in item['content']:
                    return True
    return False


def run_search_evaluation(dataframe, method):
    """
    Evaluate the search effectiveness based on retrieved documents and exact text.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'search_evaluation' indicating if the exact text was found.
    """
    dataframe["search_evaluation"] = " "

    file_path = f'data/{method}/chunks_{method}.json'
    with open(file_path, 'r') as file:
        source_documents = json.load(file)

    with tqdm(total=len(dataframe), ncols=100, desc='Search Evaluation', position=0) as pbar:
        for index, row in dataframe.iterrows():
            retrieved_doc_ids = row["retrieved_docs"]
            exact_text = row["exact_text"]
            dataframe.loc[index, "search_evaluation"] = check_search(retrieved_doc_ids, exact_text, source_documents)
            pbar.update(1)
    return dataframe


def run_full_evaluation(df, method):
    """
    Perform both LLM and search evaluations on the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional columns for both LLM and search evaluations.
    """
    df = run_llm_evaluation(df)
    df = run_search_evaluation(df, method)
    return df

