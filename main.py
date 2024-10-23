from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Setup API Key for OpenAI
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_PROJECT_API_KEY")

# Define LLM and tools
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Wikipedia Tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# LangSmith Retriever Tool
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                       "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

# Arxiv Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Tools List
tools = [wiki, arxiv, retriever_tool]

# Prompt from LangChain Hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Agent Setup
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Input Model for Request
class QueryRequest(BaseModel):
    input: str


# FastAPI Endpoints

@app.post("/query")
def query_agent(request: QueryRequest):
    """
    Query the agent with input and return the response.
    """
    try:
        response = agent_executor.invoke({"input": request.input})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_arxiv")
def search_arxiv():
    """
    Search the Arxiv database using the API.
    """
    try:
        response = requests.get(
            'https://export.arxiv.org/api/query?search_query=artificial+intelligence&sortBy=relevance&sortOrder=descending&start=0&max_results=100',
            verify=False
        )
        return {"arxiv_response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
