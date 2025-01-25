import streamlit as st
import openai
from typing import List

# ============================================================
# Streamlit interface setup.

st.set_page_config(page_title="OpenAI Assistant", layout="wide")

# Sidebar setup for API key input and GitHub link.
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
st.sidebar.markdown("[GitHub Repository](https://github.com/your-repo-link)")

# Ensure API key is provided.
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

openai.api_key = api_key

# ============================================================
# Helper functions for managing Assistant, Thread, and Run.

def create_assistant() -> dict:
    """Creates an OpenAI Assistant."""
    response = openai.Assistant.create(name="Research Assistant", description="A research expert that gathers and organizes information.")
    return response

def create_thread(assistant_id: str) -> dict:
    """Creates a new Thread for the given Assistant."""
    response = openai.Assistant.Thread.create(assistant_id=assistant_id, title="User Query Thread")
    return response

def run_thread(thread_id: str, user_message: str) -> dict:
    """Creates a Run within a Thread, sending the user message and returning the assistant's response."""
    response = openai.Assistant.Thread.Run.create(thread_id=thread_id, messages=[{"role": "user", "content": user_message}])
    return response

# ============================================================
# Initialize session state for Assistant, Thread, and conversation history.

if "assistant" not in st.session_state:
    st.session_state.assistant = create_assistant()

if "thread" not in st.session_state:
    assistant_id = st.session_state.assistant["id"]
    st.session_state.thread = create_thread(assistant_id)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Main UI for conversation.
st.title("OpenAI Assistant")
st.write("Enter your query below and let the assistant perform research for you.")

query = st.text_input("Your query:")
if st.button("Run Query"):
    if query:
        thread_id = st.session_state.thread["id"]
        response = run_thread(thread_id, query)

        # Append user query and assistant response to conversation history.
        user_message = {"role": "user", "content": query}
        assistant_message = response["messages"][-1]  # Get the latest assistant message.

        st.session_state.conversation.append(user_message)
        st.session_state.conversation.append(assistant_message)
    else:
        st.warning("Please enter a query.")

# Display conversation history.
st.subheader("Conversation History")
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"**User:** {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**Assistant:** {message['content']}")
