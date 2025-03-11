import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize session state
if "vectordbs" not in st.session_state:
    st.session_state.vectordbs = []

# Embedding creation
def create_embeddings(url: str):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for script in soup(["script", "style", "meta", "link"]):
        script.decompose()

    text = soup.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb

# Get context from all vector DBs
def get_context(query: str):
    context = []
    for vectordb in st.session_state.vectordbs:
        retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 4})
        docs = retriever.invoke(query)
        for doc in docs:
            context.append(doc.page_content)
    return str(context)

# Chat with LLM
def chat(query: str, context: str):
    prompt = """You are a professional chatbot that can answer any queries related to the context provided. Make sure your
    response should be highly professional and up to the mark. You are not supposed to generate anything harmful or that may hurt anyone's sentiments.
    Context: {context}
    Query: {query}
    """
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)

    PROMPT = PromptTemplate(
        template=prompt,
        input_variables=["context", "query"],
    )

    llm_chain = PROMPT | llm | StrOutputParser()
    response = llm_chain.invoke({"query": query, "context": context})
    return response

# Streamlit UI
st.set_page_config(page_title="Web Chatbot", layout="wide")
st.title("🌐 Web Page Chatbot")

with st.expander("Step 1: Add URLs for scraping"):
    urls = st.text_area("Enter one or more URLs (one per line):", height=150)
    if st.button("Create Embeddings"):
        url_list = [u.strip() for u in urls.split("\n") if u.strip()]
        if not url_list:
            st.warning("Please enter at least one valid URL.")
        else:
            st.session_state.vectordbs.clear()
            with st.spinner("Scraping and generating embeddings..."):
                for url in url_list:
                    try:
                        vectordb = create_embeddings(url)
                        st.session_state.vectordbs.append(vectordb)
                    except Exception as e:
                        st.error(f"Failed to process {url}: {e}")
            st.success("Embeddings created successfully!")

st.markdown("---")

if st.session_state.vectordbs:
    st.subheader("Step 2: Ask Your Questions")
    query = st.text_input("Ask something based on the scraped pages:")
    ask_button = st.button("Ask")
    if ask_button and query:
        with st.spinner("Generating response..."):
            context = get_context(query)
            response = chat(query, context)
            st.markdown("### 🤖 Chatbot Response")
            st.success(response)
else:
    st.info("Please create embeddings first by entering URLs above.")
