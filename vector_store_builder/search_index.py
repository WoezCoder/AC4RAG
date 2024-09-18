import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    SearchIndex
)

load_dotenv()


class SearchIndexConfiguration:

    def __init__(self):
        """
        Initializes the SearchIndexConfiguration object by loading the Azure endpoint, credentials,
        and setting up the search fields and index name.
        """
        self.endpoint = os.environ['AZURE_SEARCH_SERVICE_ENDPOINT']
        self.credential = AzureKeyCredential(os.environ['AZURE_SEARCH_ADMIN_KEY'])
        self.index_client = SearchIndexClient(self.endpoint, self.credential)
        self.search_fields = self.define_search_fields()
        self.index_name = os.environ['AZURE_SEARCH_INDEX']

    def define_search_fields(self):
        """
        Defines the fields for the search index, including a vector search field for embedding-based queries.

        Returns:
        list: A list of SearchField and SimpleField objects specifying the fields for the index.
        """
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True),
            SearchField(name="content", type=SearchFieldDataType.String, sortable=False, filterable=False,
                        facetable=False, searchable=True),
            SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        vector_search_dimensions=1536, vector_search_profile_name="myEKNNProfile")
        ]
        return fields

    def config_vector_search(self):
        """
        Configures the vector search capability using an exhaustive K-Nearest Neighbor (KNN) algorithm
        with cosine similarity as the distance metric.

        Returns:
        VectorSearch: The vector search configuration object.
        """
        vector_search = VectorSearch(
            algorithms=[
                ExhaustiveKnnAlgorithmConfiguration(
                    name="myEKNN",
                    parameters=ExhaustiveKnnParameters(
                        metric="cosine"
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myEKNNProfile",
                    algorithm_configuration_name="myEKNN"
                )
            ]
        )
        return vector_search

    def create_search_index(self):
        """
        Creates or updates the search index with the specified fields and vector search configuration.

        The method also prints the name of the created or updated index upon success.
        """
        vector_search = self.config_vector_search()

        index = SearchIndex(name=self.index_name, fields=self.search_fields, vector_search=vector_search)
        result = self.index_client.create_or_update_index(index)
        print(f"{result.name} created")


if __name__ == "__main__":
    search_index = SearchIndexConfiguration()
    search_index.create_search_index()
