import pandas as pd
import os
from dotenv import load_dotenv
from vector_store_builder.search_index import SearchIndexConfiguration
from vector_store_builder.upload import Uploader
from get_answer.answer_generation import generate_answer_dataset
from evaluation_framework.evaluation_dataset import run_full_evaluation
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential


load_dotenv()

endpoint = os.environ['AZURE_SEARCH_SERVICE_ENDPOINT']
credential = AzureKeyCredential(os.environ['AZURE_SEARCH_ADMIN_KEY'])
indexer_name = os.environ['AZURE_SEARCH_INDEX']

df = pd.read_csv("data/QA_dataset.csv")
methods = ["semantic"]  # ["agentic_breakpoint", "agentic_full_text", "semantic", "fixed_length"]
number_of_docs = [1, 3, 5]

client = SearchIndexClient(endpoint=endpoint, credential=credential)
client.delete_index(indexer_name)

for method in methods:
    # create search index
    search_index = SearchIndexConfiguration()
    search_index.create_search_index()

    # upload chunks
    upload = Uploader()
    upload.upload_files(method=method)

    # run full pipeline
    for amount in number_of_docs:
        print(f"{method}:")
        df = generate_answer_dataset(df, amount)

        # run evaluation
        df = run_full_evaluation(df, method)
        df.to_csv(f'data/{method}/evaluated_{amount}.csv', index=False)
        print("\n")

    # clear vector store
    client = SearchIndexClient(endpoint=endpoint, credential=credential)
    client.delete_index(indexer_name)
