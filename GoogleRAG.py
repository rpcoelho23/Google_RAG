import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import find_dotenv,load_dotenv
dotEnvPath=find_dotenv()
load_dotenv(dotEnvPath)  # This will load the .env file variables into the environment
st.set_page_config(page_title="Talk to Your PDF", layout="wide")
#Fixed by Rodrigo Pinto Coelho
st.markdown("""
## Talk to your PDFs: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-1.5-pro. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key https://makersuite.google.com/app/apikey.

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

# This is the first API key input; no need to repeat it in the main function.
#Gapi_key = os.getenv('GOOGLE_API_KEY')
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
#genai.configure(api_key=Gapi_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
    #                                          google_api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #vector_store = FAISS.from_documents(text_chunks[:100], embeddings)
    vector_store = FAISS.from_texts(text_chunks[:100], embeddings)
    for i in range(100, len(text_chunks), 100):
        vector_store.add_documents(documents=text_chunks[i+1:i+100], embeddings=embeddings)
        #print(i)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ")
    st.write(response["output_text"])

def main():
    st.header("AI Chatbot💁")
    with st.sidebar:
        st.title("Menu:")
        if os.getenv('GOOGLE_API_KEY') == None:
            Gapi_key = st.text_input(
                "Input your Google API Key:",
                type="password",
                placeholder="insert your API key",
            )
        else:
            Gapi_key = os.getenv('GOOGLE_API_KEY')
            genai.configure(api_key=Gapi_key)
            st.write("Google Key loaded from .env")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and Gapi_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, Gapi_key)
                st.success("Done")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and Gapi_key:  # Ensure API key and user question are provided
        user_input(user_question, Gapi_key)

if __name__ == "__main__":
    main()

#conda activate py3.11.7
#cd C:\Users\Luzia\Documents\pasta\Arquivos\Code\LLM_OS
#streamlit run GoogleRAG.py