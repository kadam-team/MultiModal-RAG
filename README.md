# -----------------------------------------------------------------------------
<h1 align="center">
  <br>
  <a href="https://github.com/your-username/rag-app"><img src="https://img.shields.io/badge/RAG-App-blue?style=for-the-badge&logo=github"></a>
  <br>
  Context-Aware Document Q&A with RAG
  <br>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/LangChain-Enabled-brightgreen.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/Vector%20DB-Qdrant-darkblue.svg" alt="Qdrant">
  <img src="https://img.shields.io/badge/LLM-Groq-orange.svg" alt="Groq">
</p>

### ✨ What is this project?

This project is a modular and scalable Retrieval-Augmented Generation (RAG) pipeline designed for question-answering on a PDF document. It uses a combination of modern libraries to parse complex documents (including text, tables, and images), store the information in a vector database, and then use that context to provide accurate and insightful answers with a Large Language Model.

The core idea is to move beyond simple summarization by allowing the LLM to "look up" relevant information from a specific document before generating a response.

### 🚀 Key Features

-   **Modular Design**: Code is split into logical modules (`pdf_parser`, `vector_store`, `rag_chain`) for easy maintenance and understanding.
-   **Advanced PDF Parsing**: Extracts not just raw text, but also tables (formatted in Markdown) and image metadata, providing a richer context for the LLM.
-   **Local Vector Database**: Utilizes `Qdrant` to store document embeddings locally, making it fast and easy to run without external dependencies.
-   **Efficient Chunking**: Uses `LangChain`'s `RecursiveCharacterTextSplitter` to create optimal chunks of text, preserving context.
-   **Lightning-Fast LLM**: Integrates with `Groq` to leverage a high-speed LLM for rapid answer generation.

### 📝 How it Works

The RAG pipeline operates in a simple, elegant flow:

1.  **Ingestion (Conditional)**:
    -   The script first checks if the Qdrant vector database has already been populated with data.
    -   If not, a PDF is parsed, extracting text, tables, and image metadata page by page.
    -   This raw data is then saved to a JSON file for persistence.
    -   The text is split into smaller, overlapping `chunks` to capture local context.
    -   Each chunk is converted into a numerical representation (an `embedding`) using `HuggingFaceEmbeddings`.
    -   These embeddings and their metadata are stored in a `Qdrant` vector collection.

2.  **Querying**:
    -   A user asks a question.
    -   The question is embedded into a vector.
    -   The vector store (`Qdrant`) finds the most relevant document chunks based on semantic similarity.
    -   These chunks are passed as `context` to the LLM's prompt.
    -   The LLM generates a well-informed answer based *only* on the provided context and the user's question.

### 🔧 Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/rag-app.git](https://github.com/your-username/rag-app.git)
    cd rag-app
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    -   Create a `.env` file in the root directory.
    -   Add your `GROQ_API_KEY` from the Groq console.
    -   The `.env.example` file provides a template.

    ```ini
    # rag-app/.env
    GROQ_API_KEY="your_groq_api_key_here"
    ```

### 🏃 How to Run

1.  **Prepare your Document:**
    -   Place your PDF document in a folder named `docs` in the project root.
    -   Ensure the `PDF_PATH` in `src/config.py` is set correctly. The default is `docs/v2 WEO Oct 2018.pdf`.

2.  **Start the Application:**
    -   The first run will parse the PDF and ingest the data. Subsequent runs will be much faster as the ingestion step is skipped.

    ```bash
    python main.py
    ```

3.  **Ask a Question:**
    -   The application will prompt you for a question.
    -   Example: `What are the key economic outlooks and challenges discussed in the report?`

### 📚 Technologies Used

* **Python:** The core programming language.
* **LangChain:** For building the RAG pipeline components.
* **Groq:** A high-performance LLM API for fast and efficient responses.
* **Qdrant:** A local vector database for storing and retrieving document embeddings.
* **PyMuPDF & pdfplumber:** Powerful libraries for advanced PDF parsing.
* **Sentence-Transformers:** For generating high-quality embeddings.
* **python-dotenv:** For managing environment variables securely.