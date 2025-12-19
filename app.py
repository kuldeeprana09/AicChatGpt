# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st



# Set up Streamlit UI
st.title("AIC ChatBOt")
# Styling
st.markdown("""
<style>
.main {
    background-color: #00000;
}
</style>
""", unsafe_allow_html=True)

# Create a Sidebar
with st.sidebar:
    st.info("This app uses deepseek model to answer your questions.")

# Create the prompt template
template = """Question: {question}
Answer: Let's think step by step """
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Please respond to the user queries."),
#     ("user", "Question: {question}")
# ])
prompt = ChatPromptTemplate.from_template(template)

# Ollama LLM model setup
llm = OllamaLLM(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434"
)  # Change to "llama2" for the larger model

chain = prompt | llm

# Main Content
col1, col2 = st.columns(2)
with col1:
    question = st.text_input("Enter your question")

if question:
    with st.spinner('Thinking...'):
        answer = chain.invoke({"question": question})
        st.success("Done!")
    st.markdown(f"**Answer:** {answer}")
else:
    st.warning("Please enter a question to get answer.")



