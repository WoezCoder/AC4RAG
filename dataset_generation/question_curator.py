import os
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from question_generator import generate_dataset


load_dotenv()

# Initialize OpenAI Chat model
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.environ['OPENAI_KEY'])


class Tagging(BaseModel):
    """This class is used to define a schema for tagging the ratings of the questions, ensuring a structured output."""
    rating: Optional[str] = Field(
        description="""Assign the question the score which it belongs to.
                    1 is the worst, 5 is the best. Do not be too strict.""",
        enum=['1', '2', '3', '4', '5']
        )


def evaluate_groundedness(question, context):
    """
    Evaluates how well a question can be answered given a specific context.

    :return: A score from 1 (not grounded) to 5 (highly grounded).
    """

    question_groundedness_critique_prompt = """
        You will be given a context and a question.
        Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with 
        the given context. Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at 
        all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.
        
        Question: {question}
        
        Context: {context}
    """

    prompt = ChatPromptTemplate.from_template(question_groundedness_critique_prompt)
    chain = prompt | llm.with_structured_output(Tagging)
    response = chain.invoke({"question": question, "context": context})
    return response.rating


def evaluate_relevance(question):
    """
    Evaluates how relevant or useful a question is for gaining information from an essay on the topic.

    :return: A score from 1 (not useful) to 5 (highly useful).
    """
    question_relevance_critique_prompt = """
        You will be given a question.
        Your task is to provide a 'total rating' representing how useful this question can be for gaining information 
        from an essay in which the topic is discussed. Give your answer on a scale of 1 to 5, where 1 means that the 
        question is not useful at all, and 5 means that the question is extremely useful.
        
        Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(question_relevance_critique_prompt)
    chain = prompt | llm.with_structured_output(Tagging)
    response = chain.invoke({"question": question})
    return response.rating


def evaluate_standalone(question):
    """
    Evaluates how context-independent a question is. If the question is meaningful without any external context, it will score higher.

    :return: A score from 1 (dependent on context) to 5 (makes sense independently).
    """
    question_standalone_critique_prompt = """
        You will be given a question.
        Your task is to provide a 'total rating' representing how context-independant this question is.
        
        Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be 
        understood, and 5 means that the question makes sense by itself. For instance, if the question refers to a 
        particular setting, like 'in the context' or 'in the document', the rating must be 1. 
        
        The questions can contain obscure technical nouns or acronyms and still be a 5: it must simply be clear to an 
        operator with access to documentation what the question is about.

        For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, 
        since there is an implicit mention of a context, thus the question is not independant from the context.
        
        Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(question_standalone_critique_prompt)
    chain = prompt | llm.with_structured_output(Tagging)
    response = chain.invoke({"question": question})
    return response.rating


def generate_critique(dataframe):
    """
    Generate a critique for each question in the dataset by evaluating groundedness, relevance, and standalone nature.

    :return: DataFrame with added critique columns for 'groundedness', 'relevance', and 'standalone'.
    """
    dataframe['groundedness'] = ""
    dataframe['relevance'] = ""
    dataframe['standalone'] = ""

    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Evaluating"):
        evaluations = {
            "groundedness": int(evaluate_groundedness(row["question"], row["context"])),
            "relevance": int(evaluate_relevance(row["question"])),
            "standalone": int(evaluate_standalone(row["question"]))
        }

        dataframe.at[index, 'groundedness'] = evaluations["groundedness"]
        dataframe.at[index, 'relevance'] = evaluations["relevance"]
        dataframe.at[index, 'standalone'] = evaluations["standalone"]

    return dataframe


def curate_dataframe(dataframe):
    """
    Filter the dataframe to only retain high-quality questions with scores of 4 or above for groundedness, relevance,
    and standalone nature.

    :return: Filtered dataframe with only high-quality questions.
    """
    dataframe = dataframe.loc[
        (dataframe["groundedness"] >= 4)
        & (dataframe["relevance"] >= 4)
        & (dataframe["standalone"] >= 4)
        ]
    return dataframe


def process_and_save_dataset(batch_size, total_samples, file_path):
    """
    Process and save the dataset in batches. For each batch, generate questions, evaluate their quality, curate the
    high-quality ones, and append them to a CSV file.
    """

    for start in range(0, total_samples, batch_size):
        try:
            df = generate_dataset(batch_size)
            df = generate_critique(df)
            df = curate_dataframe(df)

            if start == 0:
                df.to_csv(file_path, index=False, mode='w')
            else:
                df.to_csv(file_path, index=False, mode='a', header=False)

        except Exception as e:
            print(f"Error in batch {start // batch_size + 1}: {e}")
            continue


if __name__ == "__main__":
    number_of_samples = 2000
    batch_size = 200
    output_file = '../data/QA_dataset.csv'

    process_and_save_dataset(batch_size, number_of_samples, output_file)
