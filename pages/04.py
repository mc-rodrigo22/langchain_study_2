from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="SiteGPT", page_icon="🔅")

# OpenAI API key setup
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai_api_key or not openai_api_key.startswith("sk-"):
        st.warning("Please provide a valid OpenAI API key to proceed.")
        st.stop()

# Initialize ChatOpenAI model
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key
)

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

# Parse and clean sitemap pages
def parse_page(soup):
    for tag in ["header", "footer", "nav"]:
        if (element := soup.find(tag)):
            element.decompose()
    return soup.get_text(separator=" ", strip=True)

# Caching the website loading process
@st.cache_data(show_spinner="Loading and processing website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

# Sidebar input for sitemap URL
st.sidebar.markdown("### Provide Sitemap URL")
url = st.sidebar.text_input("Sitemap URL", placeholder="https://example.com/sitemap.xml")

# Session state for storing chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if url:
    if not url.endswith(".xml"):
        st.error("Please enter a valid sitemap URL ending with .xml.")
    else:
        retriever = load_website(url)

        st.markdown("### Ask a question")
        query = st.text_input("Type your question here...")

        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)

            st.session_state["messages"].append({"user": query, "bot": result.content})

        # Display chat history
        for msg in st.session_state["messages"]:
            st.markdown(f"**You:** {msg['user']}")
            st.markdown(f"**SiteGPT:** {msg['bot']}")
