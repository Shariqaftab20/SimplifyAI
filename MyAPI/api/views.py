from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .serializers import QuerySerializer
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_tools_agent , AgentExecutor
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables (e.g., API keys) and set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up logging for error tracking
logger = logging.getLogger(__name__)

# Initialize Ollama Llama 2 embeddings for vector search
embeddings = OllamaEmbeddings(model="llama2")

# Load FAISS vector store if it exists; otherwise, create and save it
FAISS_INDEX = 'faiss_index'
if os.path.exists(FAISS_INDEX):
    vectordb = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
else:
    # Split PDF into chunks and create a FAISS vector store for it
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    loader = PyPDFLoader("api/test.pdf")
    pages = loader.load_and_split()
    documents = text_splitter.split_documents(pages)
    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(folder_path=FAISS_INDEX)

# Set up the retriever tool to search the knowledge base
retriever = vectordb.as_retriever()
knowledgeBaseSearchTool = create_retriever_tool(
    retriever, "KnowledgeBase_Search",
    "Search for information about Shariq. For any question about Shariq, you must use this tool."
)

# Define the language model (OpenAI's GPT) and tools (retriever)
tools = [knowledgeBaseSearchTool]
llm = ChatOpenAI()

# Define the prompt template for the AI system's response behavior
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the following question based only on the provided context. Think step by step before providing a detailed answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Set up the agent to manage interactions with the LLM and tools (retriever)
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agentExe = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Set up chat memory to track conversation history across queries
memory = ChatMessageHistory()
chatAgent = RunnableWithMessageHistory(
    agentExe,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Define the API endpoint for processing customer queries
@api_view(['POST'])
def query_ai(request):
    serializer = QuerySerializer(data=request.data)
    if serializer.is_valid():
        query = serializer.validated_data['query']
        
        try:
            # Invoke the AI system to process the user's query and return a response
            result = chatAgent.invoke(
                {"input": query},
                config={"configurable": {"session_id": "<foo>"}}
            )
            return Response({"response": result}, status=status.HTTP_200_OK)

        except Exception as e:
            # Log and return an error if AI processing fails
            logger.error("Error processing query: %s", str(e))
            return Response({"error": "Error generating AI response"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Return validation errors if the request data is invalid
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
