import os
import streamlit as st
from dotenv import load_dotenv
from github import Github

from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores.tidb_vector import TiDBVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from git.exc import GitCommandError

# Load environment variables
load_dotenv()

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="Codebase Autopsy Agent",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a dark theme
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    /* Buttons */
    .stButton>button {
        border: 2px solid #4A4A5A;
        border-radius: 20px;
        color: #FAFAFA;
        background-color: #262730;
    }
    .stButton>button:hover {
        border-color: #00F6FF;
        color: #00F6FF;
    }
    /* Expander */
    .stExpander {
        border: 1px solid #4A4A5A;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
TIDB_CONNECTION_STRING = os.getenv("TIDB_CONNECTION_STRING")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TIDB_TABLE_NAME = "multi_repo_embeddings" # Use a new table for this version

# Initialize LangChain components
try:
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
    vector_store = TiDBVectorStore(
        embedding_function=embeddings,
        connection_string=TIDB_CONNECTION_STRING,
        table_name=TIDB_TABLE_NAME,
    )
except Exception as e:
    st.error(f"Failed to initialize LangChain components: {e}")
    st.stop()

@st.cache_data(show_spinner="Ingesting repository... this may take a few minutes.")
def ingest_repository(repo_url):
    """
    Clones, chunks, and embeds a repository's code into TiDB,
    tagging each vector with the repo's name for filtering.
    """
    try:
        repo_name = "/".join(repo_url.split("/")[-2:])
        clone_path = f"./temp_repos/{repo_name.replace('/', '_')}"
        
        # --- TRY/EXCEPT LOGIC ---
        try:
            # First, try to load with the 'main' branch
            loader = GitLoader(
                clone_url=repo_url, repo_path=clone_path, branch="main",
                file_filter=lambda file_path: file_path.endswith((".py", ".js", ".ts", ".go", ".rs", ".md", ".yaml", ".txt"))
            )
            documents = loader.load()
        except GitCommandError as e:
            # If 'main' doesn't exist, try 'master'
            if "pathspec 'main' did not match" in str(e):
                loader = GitLoader(
                    clone_url=repo_url, repo_path=clone_path, branch="master",
                    file_filter=lambda file_path: file_path.endswith((".py", ".js", ".ts", ".go", ".rs", ".md", ".yaml", ".txt"))
                )
                documents = loader.load()
            else:
                # If it's a different git error, raise it
                raise e

        # Split Code
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Add Metadata for Scoped Searching
        # tag each document chunk with its source repository.
        for doc in docs:
            doc.metadata["repo"] = repo_name

        # Embed and Store in Batches
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            vector_store.add_documents(batch)
        
        return f"Successfully ingested {repo_name}!"
    except Exception as e:
        return f"Error ingesting repository: {e}"

def create_github_issue(title, body, repo_name):
    """Creates an issue in a GitHub repository."""
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(repo_name)
        issue = repo.create_issue(
            title=f"AI Bug Analysis: {title[:150]}", body=body
        )
        return issue.html_url
    except Exception as e:
        return f"Failed to create GitHub issue: {e}"

# --- Prompt & Chain Definitions ---
template_analyze = """
    You are an expert software developer with deep knowledge of the LangChain library.
    Based on the following error message and relevant code context, provide a concise root cause analysis of the bug.
    Error Message: {error}
    Relevant Code Context: {context}
    Root Cause Analysis:
    """
template_fix = """
    You are an expert software developer with deep knowledge of the LangChain library.
    Based on your root cause analysis and the provided context, generate a code snippet to fix the bug.
    Explain the fix clearly.
    Root Cause Analysis: {analysis}
    Relevant Code Context: {context}
    Suggested Fix (Code and Explanation):
    """
template_ask = """
You are an expert software developer who has memorized the entire repository.
Based on the user's question and the relevant code context provided, answer the question thoroughly.
Provide code snippets from the context where relevant.

Question:
{question}

Relevant Code Context:
{context}

Answer:
"""
prompt_analyze = ChatPromptTemplate.from_template(template_analyze)
prompt_fix = ChatPromptTemplate.from_template(template_fix)
prompt_ask = ChatPromptTemplate.from_template(template_ask)

# # --- Streamlit UI ---

# st.title("Your Personal Codebase Autopsy Agent 👨‍⚕️")
# st.write("Load any public GitHub repository, then provide an error message to get an AI-powered diagnosis and fix.")

# # --- UI Step 1: Load Repository ---
# st.header("1. Load a Repository")
# repo_url_input = st.text_input("Public GitHub URL:", placeholder="https://github.com/user/repository")

# if st.button("Load Repository"):
#     if repo_url_input:
#         result_message = ingest_repository(repo_url_input)
#         if "Successfully" in result_message:
#             st.session_state.current_repo = "/".join(repo_url_input.split("/")[-2:])
#             st.success(result_message)
#         else:
#             st.error(result_message)
#     else:
#         st.warning("Please provide a repository URL.")

# st.divider()

# # --- UI Step 2: Diagnose Bug (only shows if a repo is loaded) ---
# if 'current_repo' in st.session_state:
#     st.header(f"2. Diagnose a Bug in `{st.session_state.current_repo}`")
    
#     # Create a retriever that is FILTERED to the current repository
#     retriever = vector_store.as_retriever(
#         search_kwargs={"filter": {"repo": st.session_state.current_repo}}
#     )

#     # Rebuild chains with the new, filtered retriever
#     chain_analyze = (
#         {"context": retriever, "error": RunnablePassthrough()} | prompt_analyze | llm | StrOutputParser()
#     )
#     chain_fix = prompt_fix | llm | StrOutputParser()

#     error_input = st.text_area("Paste Error Message Here:", height=150)

#     if st.button("Diagnose Bug"):
#         if error_input:
#             with st.spinner("Diagnosing..."):
#                 st.session_state.error_input = error_input
#                 st.session_state.analysis_result = chain_analyze.invoke(error_input)
                
#                 context_for_fix = retriever.invoke(error_input)
#                 fix_input = {
#                     "analysis": st.session_state.analysis_result,
#                     "context": context_for_fix,
#                     "error": error_input
#                 }
#                 st.session_state.fix_result = chain_fix.invoke(fix_input)
#         else:
#             st.warning("Please paste an error message.")

#     # Display results and GitHub button
#     if 'analysis_result' in st.session_state:
#         st.subheader("🔬 Root Cause Analysis")
#         st.markdown(st.session_state.analysis_result)
#         st.subheader("🛠️ Suggested Fix")
#         st.markdown(st.session_state.fix_result)
        
#         st.divider()
        
#         if st.button("Create GitHub Issue"):
#             with st.spinner("Creating issue..."):
#                 issue_body = (
#                     "### 🤖 AI Bug Report\n\n"
#                     "**Original Error:**\n"
#                     f"```\n{st.session_state.error_input}\n```\n\n"
#                     "--- \n\n"
#                     "### 🔬 Root Cause Analysis\n"
#                     f"{st.session_state.analysis_result}\n\n"
#                     "--- \n\n"
#                     "### 🛠️ Suggested Fix\n"
#                     f"{st.session_state.fix_result}"
#                 )
#                 issue_url = create_github_issue(st.session_state.error_input, issue_body, st.session_state.current_repo)
#                 if "Failed" in issue_url: st.error(issue_url)
#                 else: st.success(f"Issue created! [View it here]({issue_url})")


# --- Streamlit UI ---

# Use a sidebar for repository loading
with st.sidebar:
    st.header("Load Repository")
    st.write("Load any public GitHub repository to begin.")
    repo_url_input = st.text_input("Public GitHub URL:", placeholder="https://github.com/user/repository")

    if st.button("Load Repository"):
        if repo_url_input:
            result_message = ingest_repository(repo_url_input)
            if "Successfully" in result_message:
                st.session_state.clear()
                st.session_state.current_repo = "/".join(repo_url_input.split("/")[-2:])
                st.success(result_message)
            else:
                st.error(result_message)
        else:
            st.warning("Please provide a repository URL.")

st.title("Codebase Autopsy Agent 👨‍⚕️")
st.write("An AI agent that diagnoses bugs and answers questions about any codebase.")

# Main app logic only runs if a repository has been successfully loaded
if 'current_repo' not in st.session_state:
    st.info("Please load a repository using the sidebar to get started.")
else:
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"repo": st.session_state.current_repo}}
    )

    tab1, tab2 = st.tabs(["Diagnose a Bug 🐞", "Ask the Codebase 🤔"])

    with tab1:
        st.header(f"Diagnose a Bug in `{st.session_state.current_repo}`")
        error_input = st.text_area("Paste Error Message:", height=150)

        if st.button("Diagnose Bug", key="diagnose"):
            if error_input:
                with st.spinner("Diagnosing..."):
                    st.session_state.error_input = error_input
                    
                    context_docs = retriever.invoke(error_input)
                    with st.expander("View Retrieved Context"):
                        for doc in context_docs:
                            st.code(doc.page_content, language="python")

                    chain_analyze = ({"context": retriever, "error": RunnablePassthrough()} | prompt_analyze | llm | StrOutputParser())
                    analysis_result = chain_analyze.invoke(error_input)
                    st.session_state.analysis_result = analysis_result
                    
                    chain_fix = prompt_fix | llm | StrOutputParser()
                    fix_input = {"analysis": analysis_result, "context": context_docs, "error": error_input}
                    fix_result = chain_fix.invoke(fix_input)
                    st.session_state.fix_result = fix_result
            else:
                st.warning("Please paste an error message.")
        
        if 'analysis_result' in st.session_state:
            st.subheader("🔬 Root Cause Analysis")
            st.markdown(st.session_state.analysis_result)
            st.subheader("🛠️ Suggested Fix")
            st.markdown(st.session_state.fix_result)
            
            st.divider()
            
            if st.button("Create GitHub Issue"):
                with st.spinner("Creating issue..."):
                    issue_body = (
                        "### 🤖 Bug Report\n\n"
                        "**Original Error:**\n"
                        f"```\n{st.session_state.error_input}\n```\n\n"
                        "--- \n\n"
                        "### 🔬 Root Cause Analysis\n"
                        f"{st.session_state.analysis_result}\n\n"
                        "--- \n\n"
                        "### 🛠️ Suggested Fix\n"
                        f"{st.session_state.fix_result}"
                    )
                    issue_url = create_github_issue(st.session_state.error_input, issue_body, st.session_state.current_repo)
                    if "Failed" in issue_url:
                        st.error(issue_url)
                    else:
                        st.success(f"Issue created! 🎉")
                        st.markdown(f"**View it here:** [{issue_url}]({issue_url})")

    with tab2:
        st.header(f"Ask a Question about `{st.session_state.current_repo}`")
        
        chain_ask = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_ask
            | llm
            | StrOutputParser()
        )

        question_input = st.text_area("Ask anything about the codebase:", placeholder="e.g., How does user authentication work?")

        if st.button("Get Answer", key="ask"):
            if question_input:
                with st.spinner("Searching for an answer..."):
                    context_docs = retriever.invoke(question_input)
                    with st.expander("View Retrieved Context"):
                        for doc in context_docs:
                            st.code(doc.page_content, language="python")

                    st.subheader("💬 Answer")
                    # Stream the final answer
                    st.write_stream(chain_ask.stream(question_input))
            else:
                st.warning("Please enter a question.")