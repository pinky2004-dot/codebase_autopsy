# Codebase Autopsy Agent 👨‍⚕️

An AI-powered agent that acts as an expert developer, allowing you to diagnose bugs and ask natural language questions about any public GitHub repository. This project was built for the **TiDB AgentX Hackathon 2025**.

![App Demo GIF](demo.gif)
*(**Note:** You'll need to create this `demo.gif` file yourself. Tools like [ScreenToGif](https://www.screentogif.com/) or [GIPHY Capture](https://giphy.com/apps/giphycapture) make this easy!)*

---
## ## The Problem

Understanding and debugging large, unfamiliar codebases is a significant challenge for developers. It can take hours or even days to trace the source of a bug or understand how a specific feature is implemented, leading to major productivity losses.

## ## The Solution

The **Codebase Autopsy Agent** solves this problem by connecting to any public repository, ingesting its code into a **TiDB Serverless vector database**, and using a multi-step, agentic AI workflow to provide deep, contextual insights. It shifts the burden of code comprehension from the developer to the AI.

---
## ## Key Features

* **💻 On-Demand Repository Ingestion:** Load any public GitHub repository via its URL. The agent automatically handles different default branches (`main` or `master`).
* **🐞 AI-Powered Bug Diagnosis:** Paste a complex error message, and the agent performs a vector search on the codebase in TiDB to find relevant code, analyze the root cause, and suggest a concrete fix.
* **🤔 "Ask Your Codebase" Q&A:** Ask high-level, natural language questions (e.g., "How does user authentication work?") and receive detailed explanations with relevant code snippets.
* **🚀 Automated GitHub Issue Creation:** With one click, the agent can take its full analysis and automatically create a well-formatted issue in the target repository using the GitHub API.

---
## ## Tech Stack & Architecture

This project leverages a modern, agentic AI stack to deliver its features.

### ### Core Technologies
* **Backend:** Python
* **Frontend:** Streamlit
* **AI Framework:** LangChain
* **LLM Provider:** OpenAI
* **Vector Database:** **TiDB Serverless** (via TiDB Cloud)
* **External Tools:** GitHub API

### ### Architecture

The agent follows a multi-step workflow orchestrated by LangChain and powered by TiDB for retrieval.

---
## ## Getting Started

Follow these steps to set up and run the project locally.

### ### 1. Clone the Repository
```
git clone <your-repo-url>
cd <your-repo-folder>
```
### ### 2. Set Up Environment and Install Dependencies
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```
### ### 3. Create your .env file
```
OPENAI_API_KEY="sk-..."
GITHUB_TOKEN="ghp_..."
TIDB_CONNECTION_STRING="mysql+pymysql://user.cluster:password@host:4000/test?..."
```

---
## ## How to Run
```
streamlit run app.py
```