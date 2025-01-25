import streamlit as st
from openai import OpenAI
import time

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

client = OpenAI(api_key=api_key)

# ============================================================
# Define Function Calling tools.

def duckduckgo_search(query: str) -> str:
    """Simulate a DuckDuckGo search."""
    return f"Search results for '{query}' from DuckDuckGo."

def wikipedia_search(query: str) -> str:
    """Simulate a Wikipedia search."""
    return f"Summary for '{query}' from Wikipedia."

def scrape_web_content(url: str) -> str:
    """Simulate web content scraping."""
    return f"Scraped content from URL: {url}"

def save_to_txt(text: str) -> str:
    """Save content to a .txt file."""
    with open("research_results.txt", "w") as file:
        file.write(text)
    return "Research results saved to 'research_results.txt'."

# ============================================================
# Create Assistant with tools.

assistant = client.beta.assistants.create(
    name="Research Assistant",
    instructions="You are a research assistant that uses tools to answer user queries. Use the available tools to provide accurate and detailed information.",
    model="gpt-4o-mini",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "duckduckgo_search",
                "description": "Search the web using DuckDuckGo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wikipedia_search",
                "description": "Search Wikipedia for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scrape_web_content",
                "description": "Scrape content from a provided URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to scrape content from."
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_to_txt",
                "description": "Save text to a .txt file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to save."
                        }
                    },
                    "required": ["text"]
                }
            }
        }
    ]
)

# ============================================================
# Initialize session state.

if "thread" not in st.session_state:
    st.session_state.thread = client.beta.threads.create()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ============================================================
# Main UI for conversation.

st.title("OpenAI Assistant")
st.write("Enter your query below and let the assistant perform research for you.")

def handle_query(query):
    thread_id = st.session_state.thread.id

    # Check if there are active runs
    active_runs = client.beta.threads.runs.list(thread_id=thread_id)
    for run in active_runs:
        if run.status == "in_progress":
            st.warning("An active run is already in progress. Please wait for it to complete.")
            return

    # Add user message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=query
    )

    # Run the assistant
    run_response = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant.id
    )

    # Poll for run completion
    run_id = run_response.id
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id).status
        if run_status == "completed":
            break
        elif run_status in {"failed", "cancelled"}:
            st.error("Run failed or was cancelled. Please try again.")
            return
        time.sleep(1)  # Wait before checking again

    # Retrieve all messages from the thread
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    latest_message = messages[-1]  # Get the last message from the list

    # Append to conversation history
    st.session_state.conversation.append({"role": "user", "content": query})
    st.session_state.conversation.append({"role": "assistant", "content": latest_message.content})

query = st.text_input("Your query:")
if st.button("Run Query"):
    if query:
        handle_query(query)
    else:
        st.warning("Please enter a query.")

# Display conversation history.
st.subheader("Conversation History")
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"**User:** {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**Assistant:** {message['content']}")
