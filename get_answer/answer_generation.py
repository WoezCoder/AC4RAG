import os
import pandas as pd
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from langchain_openai import ChatOpenAI
from azure.core.credentials import AzureKeyCredential
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.environ['OPENAI_KEY'])


class AnswerGeneration:
    def __init__(self):
        self.endpoint = os.environ['AZURE_SEARCH_SERVICE_ENDPOINT']
        self.credential = AzureKeyCredential(os.environ['AZURE_SEARCH_ADMIN_KEY'])
        self.index_name = os.environ['AZURE_SEARCH_INDEX']
        self.openai_api_key = os.getenv('OPENAI_KEY')
        self.llm_chat = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.search_client = SearchClient(self.endpoint, self.index_name, credential=self.credential)

    def generate_answer(self, question, number_retrieved_docs):
        """
        Generates an answer to the given question using the provided number of retrieved documents.

        Args:
            question (str): The question to be answered.
            number_retrieved_docs (int): The number of documents to retrieve for context.

        Returns:
            tuple: A tuple containing the generated answer and a list of retrieved document IDs.
        """
        context, sources = self.perform_search(question, number_retrieved_docs=number_retrieved_docs)
        system_prompt = """
                You are a helpful assistant answering questions about multiple written essays. 
                Your task is to answer questions on these topics. 
                Use only the context to answer the question and NEVER make up information yourself. 
                Use only the part of the context that is relevant to the question when formulating your answer. 
                Try to stay as close to the original text as possible. Do not reformulate the text if it is unnecessary. 
                If you do not know the answer, respond with: Unfortunately, I cannot help you with this.

                Question: {question}

                Context: {context}
                """

        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm_chat
        response = chain.invoke({'question': question, 'context': context})

        return response.content, sources

    def perform_search(self, query, number_retrieved_docs, hybrid=True):
        """
        Performs a search on the Azure Search index to retrieve relevant documents based on the query.

        Args:
            query (str): The search query.
            number_retrieved_docs (int): The number of documents to retrieve.
            hybrid (bool): Whether to use hybrid search (vector and text) or just vector search.

        Returns:
            tuple: A tuple containing the context string and a list of retrieved document IDs.
        """
        embedding = client.embeddings.create(input=query, model='text-embedding-3-small').data[0].embedding
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=number_retrieved_docs, fields="contentVector")

        if hybrid:
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "content"],
                top=number_retrieved_docs
            )
        else:
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["id", "content"],
                top=number_retrieved_docs
            )

        sources = []

        response = "Context: \n\n"
        for i, result in enumerate(results):
            response += f"{result['content']}\n\n"
            sources.append(result["id"])

        return response, sources


def generate_answer_dataset(dataframe, number_of_docs):
    """
    Generates answers for a dataframe containing questions and stores the results in the dataframe.

    Returns:
        pd.DataFrame: The dataframe with added columns for generated answers and retrieved documents.
    """
    answerer = AnswerGeneration()
    dataframe["RAG_answer"] = " "
    dataframe["retrieved_docs"] = pd.Series(dtype=object)

    with tqdm(total=len(dataframe), ncols=100, desc='Generating Answers', position=0) as pbar:
        for index, row in dataframe.iterrows():
            query = row["question"]
            rag_answer, retrieved_docs = answerer.generate_answer(query, number_of_docs)
            dataframe.loc[index, "RAG_answer"] = rag_answer
            dataframe.at[index, "retrieved_docs"] = retrieved_docs
            pbar.update(1)

    return dataframe
