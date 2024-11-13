import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import HumanMessage, BaseMessage
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set GOOGLE_API_KEY in your .env file.")


# Streamlit UI setup
st.title("Question Answering Model (RAG)🌳🚀")

# Status and PDF upload area
st.write("### Status & Processing:")

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    # Show PDF upload status and other info
    st.write("Processing your PDF document...")

    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())  # Save file contents
        temp_file_path = temp_file.name  # File path for processing

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()

    # Split the documents into chunks for better indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Initialize the Google Generative AI embedding model
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Initialize FAISS vector store using the embeddings
    faiss_index = FAISS.from_documents(docs, embedding)

    # Use FAISS to perform similarity search
    retriever = faiss_index.as_retriever(search_kwargs={"k": 20})

    # Display file information in the status section
    st.write(f"File Name: {uploaded_file.name}")
    st.write(f"File Size: {uploaded_file.size} bytes")
    st.write("Status: PDF is being processed...")

    # Temperature slider for adjusting the model's randomness
    temperature = st.slider(
        "Set Temperature (for model randomness)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    st.write(f"Current Temperature: {temperature}")

    # Create a ChatGoogleGenerativeAI instance with the specified temperature
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temperature,
        max_tokens=None,
        timeout=None,
    )

    # Ask the user for a question about the PDF
    query = st.text_input("Ask a question about your PDF:")

    # Add a submit button to trigger the question processing
    submit_button = st.button("Submit Question")

    if submit_button:
        if query:
            # Define the prompt template for the system
            system_prompt = (
                """
                You are a helpful assistant tasked with answering questions based on the content of the provided document. 
                Use the relevant context from the document to formulate your answer. 

                Your response should be:
                1. Clear and concise: Limit your response to a maximum of three sentences.
                2. Informative: Ensure the answer is directly based on the content retrieved from the document.
                3. If the answer is unclear or cannot be found in the document, politely say "I don't know" or state that the information is not available.

                Remember, your goal is to assist the user by providing accurate information quickly.

                Here is the context from the document:

                {context}
                """
            )

            # Use the ChatPromptTemplate with dynamic context
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            rag_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)

            )
            response = rag_chain.invoke({"input": "query"})
            st.write("### Answer:")
            st.write(response['answer'])

        else:
            st.warning("Please enter a question.")
else:
    st.write("Please upload a PDF file to get started.")
