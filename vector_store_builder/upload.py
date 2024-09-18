import os
import json
from vector_store_builder.chunking.chunkers import chunker
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


client = OpenAI(api_key=os.environ['OPENAI_KEY'])

with open('data/paul_graham_essays.txt', 'r') as file:
    files: str = file.read()


class Uploader:
    def __init__(self):
        self.endpoint = os.environ['AZURE_SEARCH_SERVICE_ENDPOINT']
        self.credential = AzureKeyCredential(os.environ['AZURE_SEARCH_ADMIN_KEY'])
        self.index_name = os.environ['AZURE_SEARCH_INDEX']

    def upload_files(self, method):
        """
        Uploads text chunks to Azure Search Service and saves them locally as a JSON file.

        Args:
            method (str): The method for chunking the text.
        """
        chunks = chunker(method=method, files=files)
        chunks = [chunk for chunk in chunks if chunk.strip() != '']
        content_response = client.embeddings.create(input=chunks, model='text-embedding-3-small')
        content_embeddings = [item.embedding for item in content_response.data]

        documents = []
        local_documents = []

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append({
                    'id': str(i),
                    'content': chunk,
                    'contentVector': content_embeddings[i]
                })

                local_document = {
                    'id': str(i),
                    'content': chunk,
                }
                local_documents.append(local_document)

        search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
        search_client.upload_documents(documents)
        print(f"Uploaded {len(documents)} documents")

        output_path = os.path.join(f'data/{method}/', f'chunks_{method}.json')
        output_directory = os.path.dirname(output_path)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(local_documents, f, ensure_ascii=False, indent=4)
        print(f"Saved documents locally to {output_path}")


if __name__ == "__main__":
    method = "fixed_length"  # agentic_breakpoint, semantic, fixed_length
    upload = Uploader()
    upload.upload_files(method=method)
