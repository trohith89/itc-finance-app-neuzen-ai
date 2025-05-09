import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser

# API key
key = "****YOUR GOOGLE GEMINI API KEY*****"

# Memory buffer
memory_buffer = {"chat_history": []}

# Chat history
for msg in memory_buffer["chat_history"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input
user_input = st.chat_input("Ask a question about ITC's financials...")

if st.button("New Chat"):
    memory_buffer["chat_history"] = []

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    memory_buffer["chat_history"].append(HumanMessage(content=user_input))

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=key
    )

    vector_store = Chroma(
        persist_directory='****Your ChromaDB Folder Directory****',
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 1})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_docs_and_context(question):
        docs = retriever.get_relevant_documents(question)
        return {"question": question, "docs": docs, "context": format_docs(docs)}

    parallel_chain = RunnableLambda(lambda x: {
        "question": x["input"],
        **get_docs_and_context(x["input"])
    })

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """
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

    def get_history_from_buffer(_):
        return memory_buffer["chat_history"]

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

    chain = RunnablePassthrough.assign(chat_history=RunnableLambda(get_history_from_buffer)) | main_chain

    output = chain.invoke({"input": user_input})
    ai_response = output["result"]
    memory_buffer["chat_history"].append(AIMessage(content=ai_response))

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # Show sources separately
    if output.get("source_documents"):
        with st.container():
            st.markdown("### 🗂️ Source Documents", unsafe_allow_html=True)
            for i, doc in enumerate(output["source_documents"], 1):
                source_name = doc.metadata.get("source", f"Document {i}")
                st.markdown(f"**{i}. {source_name}**")
