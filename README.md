# Multimodal RAG App

A Flask-based **Retrieval-Augmented Generation (RAG)** application that combines **text and image** understanding from PDFs to answer user queries more accurately using a Large Language Model (LLM).

## What It Does

- Accepts PDF uploads (max size: 5MB).
- Extracts **text** and **images** from each page.
- Tags images with reference numbers and appends them to corresponding page text.
- Combines all page-wise content into a unified document.
- Converts the document into **vector embeddings** using **FAISS**.
- On user query:
  - Performs **top-k similarity search** on the vector store.
  - Retrieves relevant content and feeds it to an LLM for **summarized response generation**.
  - Displays **matched images** alongside the answer for context.

## 🛠Tech Stack

- **Flask** – Web backend and API
- **FAISS** – Vector similarity search
- **PyMuPDF** – PDF text and image extraction
- **OpenAI GPT 4o** – LLM for summarization
- **Docker** – Containerization

## Folder Structure

```
Multimodal-RAG-App/
├── app/
│   ├── __init__.py           # App factory
│   ├── routes.py             # Core API routes
│   ├── utils.py              # Helper functions (PDF processing, FAISS, etc.)
│   └── templates/
│       └── index.html        # Frontend UI
├── notebook/
│   └── research.ipynb        # Experiments and prototypes
├── uploads/                  # Uploaded PDFs
├── output_text/
│   └── combined_text.txt     # Merged text with image references
├── vector_store/             # FAISS vector storage
├── Dockerfile                # Docker configuration
├── run.py                    # App entry point
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/multimodal-rag-app.git
   cd multimodal-rag-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python run.py
   ```

4. **Access the app**
   - Open browser: `http://127.0.0.1:5000`

## Run with Docker (Optional)

```bash
docker build -t multimodal-rag .
docker run -p 5000:5000 multimodal-rag
```
