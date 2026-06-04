# Mental Health Chatbot 

A RAG-based mental health support chatbot that provides empathetic, context-aware responses to queries related to anxiety, depression, stress, and crisis support.

---

## Project Structure

```
Mental Health Project/
│
├── language_detection/
│   ├── preprocessor.py
│   ├── vectorizer.py
│   ├── model.py
│   ├── detector.py
│   ├── language_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── emotion_detection/
│   ├── preprocessor.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── classifier.py
│   └── emotion_model.pt
│
├── intent_classifier/
│   ├── prompts.py
│   ├── memory.py
│   └── classifier.py
│
├── rag/
│   ├── data_loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── prompts.py
│   ├── memory.py
│   ├── generator.py
│   ├── qa_records.json         ← auto-generated on first run
│   └── embeddings_q_only.pkl  ← auto-generated on first run
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   └── pipeline_manager.py
│
├── static/
│   ├── css/style.css
│   ├── js/chat.js
│   └── icons/brain.svg
│
├── templates/
│   └── index.html
│
├── run.py
├── pipeline.py
├── requirements.txt
└── .env
```

---

## System Architecture

```
User Message
     │
     ▼
Module 1 — Language Detection
(TF-IDF + Logistic Regression → detects language code e.g. "en", "ar")
     │
     ▼
Module 2 — Emotion Classifier
(DistilBERT → sadness / joy / love / anger / fear / surprise / neutral)
     │
     ▼
Module 3 — Intent Classifier + Router
(LLM zero-shot → greeting / goodbye / gratitude / asking_mental_health_question / out_of_scope)
Also detects language & emotion via LLM if modules 1/2 are not confident
     │
     ├── direct_reply  → LLM responds directly (greeting, goodbye, gratitude, out_of_scope)
     │
     └── rag_pipeline  → Module 4
              │
              ▼
         Module 4 — RAG Pipeline
         (Embed query → Qdrant search → retrieve top-5 counseling Q&A → LLM generates response)
              │
              ▼
         Rolling Memory
         (Summary of old turns + last 3 verbatim → flat token cost)
```


---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/RamyAhmed55/Mental-Health-Chatbot-
cd "Mental Health Project"
```

### 2. Create and activate environment

```bash
conda create -n medibot python=3.10
conda activate medibot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root folder:

```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 5. First run — build vector database

On the first run, the system will:
- Download the mental health counseling dataset from HuggingFace (~3 min)
- Generate embeddings and save them to `rag/embeddings_q_only.pkl`
- Upload all records to Qdrant Cloud

All subsequent runs skip these steps and load from local files.

---

## Running the Project

### Option A — Web Interface (Flask)

```bash
python run.py
```

Then open your browser at:
```
http://localhost:5000
```

### Option B — Terminal Chat

**Clean mode (answers only):**
```bash
python pipeline.py
```

**Debug mode (shows each module output per turn):**
```bash
python pipeline.py --verbose
```


---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |
| `QDRANT_URL` | From [cloud.qdrant.io](https://cloud.qdrant.io) dashboard |
| `QDRANT_API_KEY` | From Qdrant Cloud dashboard |

---

## Dataset

[Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

Real counselor–client Q&A pairs covering anxiety, depression, stress, relationships, and crisis support. Used as the knowledge base for the RAG pipeline.

---

## Models Used

| Module | Model |
|---|---|
| Language Detection | TF-IDF (char n-gram) + Logistic Regression |
| Emotion Classifier | DistilBERT fine-tuned ( 6 classes) |
| Intent Classifier | LLaMA 3.1 8B via Groq (zero-shot) |
| RAG Generation | LLaMA 3.1 8B via Groq |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector DB | Qdrant Cloud (free tier) |

---

## Notes

- The chatbot specializes in mental health support only. Out-of-scope questions receive a polite refusal.
- Conversation memory uses a rolling summary strategy — token cost stays flat regardless of conversation length.
- Models (`.pkl`, `.pt`) are saved locally and loaded on startup — no retraining needed.
