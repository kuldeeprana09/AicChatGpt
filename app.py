from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st

# Set up Streamlit UI
st.set_page_config(page_title="AIC ChatBot", layout="wide")
st.title("🤖 AIC ChatBot")

# Styling for Streamlit UI
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;  /* Fixed invalid color code */
}
</style>
""", unsafe_allow_html=True)

# Create a Sidebar
with st.sidebar:
    st.info("This app uses the DeepSeek model to answer your questions.")

# Define the prompt template with a more formal instruction
template = """You are a helpful internal AI assistant. Please answer the user's question clearly and concisely.

Question: {question}
Answer:"""

# Instantiate the prompt and Ollama model setup
prompt = ChatPromptTemplate.from_template(template)

llm = OllamaLLM(
    model="deepseek-r1:1.5b",  # You can change this model as needed
    base_url="http://localhost:11434"
)

# Combine the prompt with the LLM model
chain = prompt | llm

# Main content: Input field and response display
col1, col2 = st.columns(2)
with col1:
    question = st.text_input("Enter your question")

# Process the question and display the answer
if question:
    with st.spinner('Thinking...'):
        try:
            answer = chain.invoke({"question": question})
            st.success("Done!")
            st.markdown(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("Please enter a question to get an answer.")
