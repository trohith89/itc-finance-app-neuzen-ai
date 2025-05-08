import streamlit as st
import zipfile

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser

# API key
key = "AIzaSyCL5TCEbUCfidn-f6cMSa7Lb6P9IyyLGUw"

import base64

# Streamlit setup
st.set_page_config(page_title="ITC Financial Analysis with AI Scraping & LLM Integration", layout="centered")

def set_background(local_image_path):
    with open(local_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    bg_image_style = f"""
    <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Call this function with the path to your image
set_background(r"bkg.jpeg")



st.title("📊 ITC Financial Analysis with AI Scraping & LLM Integration")

# Memory buffer for chat history
memory_buffer = {"chat_history": []}

# End chat button
if st.button("End Chat 🛑"):
    memory_buffer["chat_history"] = []

# Load Chroma vector DB from zip
with zipfile.ZipFile('CHROMA_DB_BACKUP.zip', 'r') as zip_ref:
    zip_ref.extractall('chroma_db')

# Embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=key)

vector_store = Chroma(
    persist_directory='chroma_db',
    embedding_function=embeddings
)
mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 1})

# Helper functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_docs_and_context(question):
    docs = mmr_retriever.get_relevant_documents(question)
    return {"question": question, "docs": docs, "context": format_docs(docs)}

# Chain parts
parallel_chain = RunnableLambda(lambda x: {
    "question": x["input"],
    **get_docs_and_context(x["input"])
})

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a domain-specific AI financial analyst focused on company-level performance evaluation.

     Your task is to analyze and respond to user financial queries *strictly based on the provided transcript data*: {context}.

     Rules:
     1. ONLY extract facts, figures, and insights that are explicitly available in the transcript.
     2. If data is *missing or partially available*, clearly state: "The required data is not available in the current transcript." Then provide a generic but relevant explanation based on standard financial principles.
     3. Maintain numerical accuracy and avoid interpretation beyond data boundaries.
     4. Prioritize answers relevant to *ITC Ltd.*, but keep response format adaptable to other firms and fiscal years.
     5. Clearly present year-wise or metric-wise insights using bullet points or structured formats if applicable.

     Your goals:
     - Ensure 100% fidelity to source transcript.
     - Do not assume or hallucinate missing numbers.
     - Use clear, reproducible reasoning steps (e.g., show which line items support your conclusion).
     - Output should be modular enough to scale across other companies and time periods.

     Respond only to this question from the user.
     """),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}")
])

llm = ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash-exp", temperature=1)
parser = StrOutputParser()

# Chat memory logic
def get_history_from_buffer(_):
    return memory_buffer['chat_history']

runnable_get_history_from_buffer = RunnableLambda(get_history_from_buffer)

main_chain = (
    parallel_chain |
    RunnableLambda(lambda x: {
        "llm_input": {"input": x["question"], "context": x["context"]},
        "docs": x["docs"]
    }) |
    RunnableLambda(lambda x: {
        "result": (chat_prompt | llm | parser).invoke(x["llm_input"]),
        "source_documents": x["docs"]
    })
)

chain = RunnablePassthrough.assign(chat_history=runnable_get_history_from_buffer) | main_chain

# Display previous chat messages
for msg in memory_buffer["chat_history"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Ask a question about ITC's financials...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add user input to memory
    memory_buffer["chat_history"].append(HumanMessage(content=user_input))

    # Call the chain
    output = chain.invoke({"input": user_input})
    ai_response = output["result"]

    # Add AI response to memory
    memory_buffer["chat_history"].append(AIMessage(content=ai_response))

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

        # Show sources
        if output.get("source_documents"):
            st.markdown("**Sources:**")
            for doc in output["source_documents"]:
                st.markdown(f"- {doc.metadata.get('source', 'Unknown document')}")