from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
import streamlit as st
import pickle
import os
from concurrent.futures import ThreadPoolExecutor

# Streamlit app configuration
st.set_page_config(page_title="Cloudflare SiteGPT", page_icon="🔅")

# API Key 입력
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai_api_key or not openai_api_key.startswith("sk-"):
        st.warning("Please provide a valid OpenAI API key to proceed.")
        st.stop()

output_file = "cloudflare_sitemap.pkl"

# Initialize ChatOpenAI model with streaming
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key,
    streaming=True
)

# Initialize memory for conversation
memory = ConversationBufferMemory(return_messages=True)

# Prompt templates
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                   
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                   
    Examples:
                                                   
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                   
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                   
    Your turn!

    Question: {question}
    """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

# Answer generation function
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", "Unknown"),
            }
            for doc in docs
        ],
    }

# Answer selection function
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

# Preprocess sitemap with batch processing
def preprocess_cloudflare_sitemap(output_file="cloudflare_sitemap.pkl", batch_size=500):
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping preprocessing.")
        return

    url = "https://developers.cloudflare.com/sitemap-0.xml"
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100
    )
    loader = SitemapLoader(url)
    loader.requests_per_second = 2

    docs = []
    for i, doc in enumerate(loader.load()):
        docs.extend(splitter.split_text(doc.page_content))
        if (i + 1) % batch_size == 0 or i == len(loader.load()) - 1:
            with open(output_file, "ab") as f:
                pickle.dump(docs, f)
            docs = []  # Clear batch to free memory

    print(f"Cloudflare sitemap data saved to {output_file}.")

# Load preprocessed Cloudflare sitemap data
@st.cache_data(show_spinner="Loading Cloudflare sitemap data...")
def load_cloudflare_data(output_file="cloudflare_sitemap.pkl"):
    with open(output_file, "rb") as f:
        docs = []
        while True:
            try:
                docs.extend(pickle.load(f))
            except EOFError:
                break
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

# Preprocess and load sitemap data
preprocess_cloudflare_sitemap(output_file=output_file)
retriever = load_cloudflare_data(output_file=output_file)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat interface
st.markdown("### Cloudflare SiteGPT")
query = st.text_input("Ask a question about Cloudflare documentation:")

with st.sidebar:
    st.markdown("---")
    st.markdown("[View this project on GitHub](https://github.com/mc-rodrigo22/langchain_study_2/blob/main/pages/03_SITEGPT.py)")

if query:
    # Add query to memory
    memory.save_context({"user": query}, {})

    # Process question
    chain = (
        {
            "docs": retriever,
            "question": query,
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = chain.invoke(query)

    # Add bot response to memory
    memory.save_context({}, {"bot": result.content})

    # Update session state
    st.session_state["messages"].append({"user": query, "bot": result.content})

    # Stream bot response
    for chunk in result.content:
        st.markdown(chunk)

# Display chat history
for msg in st.session_state["messages"]:
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**SiteGPT:** {msg['bot']}")
