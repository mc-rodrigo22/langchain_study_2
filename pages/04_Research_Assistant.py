import streamlit as st
from typing import Any, Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
from langchain.schema import SystemMessage

# ============================================================
# Define custom tools for the OpenAI Assistant.

class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGoSearchTool"
    description: str = """
    Use this tool to perform web searches using the DuckDuckGo search engine.
    It takes a query as an argument.
    Example query: "Latest technology news"
    """
    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = DuckDuckGoSearchToolArgsSchema

    def _run(self, query) -> Any:
        search = DuckDuckGoSearchResults()
        return search.run(query)


class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for on Wikipedia")


class WikipediaSearchTool(BaseTool):
    name: str = "WikipediaSearchTool"
    description: str = """
    Use this tool to perform searches on Wikipedia.
    It takes a query as an argument.
    Example query: "Artificial Intelligence"
    """
    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema

    def _run(self, query) -> Any:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)


class WebScrapingToolArgsSchema(BaseModel):
    url: str = Field(description="The URL of the website you want to scrape")


class WebScrapingTool(BaseTool):
    name: str = "WebScrapingTool"
    description: str = """
    If you found the website link in DuckDuckGo,
    use this to get the content of the link for research.
    """
    args_schema: Type[WebScrapingToolArgsSchema] = WebScrapingToolArgsSchema

    def _run(self, url):
        loader = WebBaseLoader([url])
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
        return text


class SaveToTXTToolArgsSchema(BaseModel):
    text: str = Field(description="The text you will save to a file.")


class SaveToTXTTool(BaseTool):
    name: str = "SaveToTXTTool"
    description: str = """
    Use this tool to save the content as a .txt file.
    """
    args_schema: Type[SaveToTXTToolArgsSchema] = SaveToTXTToolArgsSchema

    def _run(self, text) -> Any:
        with open("research_results.txt", "w") as file:
            file.write(text)
        return "Research results saved to research_results.txt"


# ============================================================
# Streamlit interface setup.

st.set_page_config(page_title="OpenAI Assistant", layout="wide")

# Sidebar setup for API key input and GitHub link.
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
st.sidebar.markdown("[GitHub Repository](https://github.com/mc-rodrigo22/langchain_study_2/blob/main/pages/04_Research_Assistant.py)")

# Ensure API key is provided.
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

# Initialize session state for conversation.
if "assistant" not in st.session_state:
    st.session_state.assistant = ChatOpenAI(
        temperature=0.1, model="gpt-4", openai_api_key=api_key
    )

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "tools" not in st.session_state:
    st.session_state.tools = [
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        WebScrapingTool(),
        SaveToTXTTool(),
    ]

if "agent" not in st.session_state:
    system_message = SystemMessage(
        content="""
        You are a research expert. Use Wikipedia and DuckDuckGo to gather detailed information about user queries. When needed, scrape content from websites and save your findings to a text file. Always provide citations and thorough explanations.
        """
    )

    st.session_state.agent = initialize_agent(
        llm=st.session_state.assistant,
        tools=st.session_state.tools,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs={"system_message": system_message},
    )

# Main UI for conversation.
st.title("OpenAI Assistant")
st.write("Enter your query below and let the assistant perform research for you.")

query = st.text_input("Your query:")
if st.button("Run Query"):
    if query:
        st.session_state.conversation.append({"user": query})
        results = st.session_state.agent.run(query)
        st.session_state.conversation.append({"assistant": results})
    else:
        st.warning("Please enter a query.")

# Display conversation history.
st.subheader("Conversation History")
for entry in st.session_state.conversation:
    if "user" in entry:
        st.write(f"**User:** {entry['user']}")
    elif "assistant" in entry:
        st.write(f"**Assistant:** {entry['assistant']}")
