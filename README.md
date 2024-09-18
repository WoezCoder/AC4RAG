# AC4RAG: Agentic Chunking for Retrieval Augmented Generation 


## Description
This repository is made for a master thesis in Econometrics at the Erasmus University Rotterdom. We investigate how Retrieval Augmented Generation (RAG) systems can be improved to perform more effectively. 
We propose a novel chunking technique (AC4RAG V1 and V2) for dividing input text into chunks for RAG. The novel method uses GPT 4-o to perform chunking. 
We introduce AC4RAG, with two versions:

  - AC4RAG V1: Divides the full text into chunks and returns them.
  - AC4RAG V2: Utilizes GPT-4 to find a breakpoint sentence every 2000 words, which is then used to filter the text into chunks.


## Installation
To get started, you'll need to set up a few services and configurations:

1. **Create an Azure AI Search Resource:**
   - Follow the [Azure documentation](https://docs.microsoft.com/en-us/azure/search/) to set up a resource.

2. **Create an OpenAI Account:**
   - Sign up at [OpenAI](https://www.openai.com/).

3. **Setup Environment Variables:**
   - Create a `.env` file in the root directory of the project with the following content:
     ```
     AZURE_SEARCH_SERVICE_ENDPOINT=your_azure_search_endpoint
     AZURE_SEARCH_ADMIN_KEY=your_azure_search_key
     AZURE_SEARCH_INDEX=your_azure_search_index_name
     OPENAI_KEY=your_openai_api_key
     ```

4. **Install Dependencies:**
   - Ensure you have all necessary Python packages installed, run:
     ```bash
     pip install -r requirements.txt
     ```
**Note:** Be aware of the costs associated with setting up and running the project. 

## Usage 
1. Run main.py
2. Navigate to the evaluation_framework/metrics directory and run the metric evaluation scripts.
3. Navigate to the evaluation_framework/plots directory and run the plotting scripts to visualize the results.

