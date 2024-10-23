# LangChain-based FastAPI API with Wikipedia, Arxiv, and LangSmith Tools

This project is a FastAPI-based application that leverages LangChain, OpenAI, Wikipedia, Arxiv, and LangSmith for building a robust natural language querying service. The service can fetch information from Wikipedia, Arxiv, and LangSmith documentation using an intelligent agent backed by GPT-3.5.

## Features

- **Query Wikipedia**: Use a pre-configured tool to search Wikipedia for the most relevant answers.
- **Arxiv Search**: Fetch the most relevant AI-related papers from the Arxiv API.
- **LangSmith Retriever**: Retrieve information from LangSmith documentation.
- **GPT-3.5**: The application uses the OpenAI GPT-3.5 model to generate responses based on inputs.
- **FastAPI**: Provides RESTful APIs for easy interaction with the LLM and tools.

## Prerequisites

Before setting up the project, ensure that the following are installed:

- Python 3.8+
- [OpenAI API Key](https://beta.openai.com/signup/)
- [LangChain](https://github.com/hwchase17/langchain) Python SDK
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage
- An active internet connection for API calls (Wikipedia, Arxiv, etc.)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-repo/langchain-fastapi-app.git
    cd langchain-fastapi-app
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory and add your OpenAI API key:

    ```env
    OPENAI_PROJECT_API_KEY=your-openai-api-key
    ```

## Usage

### Running the Application

You can run the FastAPI app using Uvicorn:

```bash
uvicorn main:app --reload

