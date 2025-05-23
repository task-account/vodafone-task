# RAG System for Question Answering on Alice in Wonderland

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about "Alice's Adventures in Wonderland" by Lewis Carroll. The system retrieves relevant passages from the book using semantic search and generates answers using a language model.

### Objectives
- Implement efficient hierarchical and semantic chunking strategies
- Create a vector database for fast retrieval of relevant passages
- Compare RAG performance with a baseline approach (no retrieval)
- Evaluate using NarrativeQA dataset questions specific to Alice in Wonderland

## Setup Instructions

### Prerequisites
- Python 3.8+
- ~11GB RAM (peak usage during model inference)
- GPU recommended but not required (CPU mode supported)

### Environment Setup

1. Clone this repository:
```bash
git clone [repository-url]
cd rag-task
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up Hugging Face token for accessing gated models:
```bash
# Create a .env file with your HF token
echo "HF_TOKEN=your_token_here" > .env
```

## Running the System

The system can be run end-to-end or in individual stages:

### End-to-End Execution
```bash
python run.py --stage all
```

### Individual Stages

1. **Download and clean the book**:
```bash
python run.py --stage download
```

2. **Create vector index**:
```bash
python run.py --stage index --chunking semantic  # or --chunking hierarchical
```

3. **Run baseline evaluation** (without retrieval):
```bash
python run.py --stage baseline
```

4. **Run RAG evaluation**:
```bash
python run.py --stage rag --chunking semantic  # or --chunking hierarchical
```

## Technical Details

### Chunking Strategies

The system implements two chunking strategies:

1. **Hierarchical Chunking**: Divides the text into parent chunks (2048 chars) and overlapping child chunks (512 chars with 256 char stride). Child chunks are embedded and indexed, but parent chunks are returned for context.

2. **Semantic Chunking**: Creates chunks based on semantic boundaries like paragraphs, providing more natural context. This is the recommended approach as it preserves narrative flow.

### Embedding Model

- **Model**: sentence-transformers/all-MiniLM-L6-v2 (22M parameters)
- **Embedding Size**: 384 dimensions
- **Characteristics**: Fast, efficient, good semantic understanding despite small size

### Retrieval Logic

- **Vector Database**: Qdrant (local disk-based)
- **Index Size**: <200 MB
- **Retrieval Strategy**:
  - Embed query using the same embedding model
  - Retrieve top-k=5 most similar passages
  - Use hybrid re-ranking combining vector similarity and keyword matching
  - Return parent chunks for better context

### Generation

- **Model**: Google Gemma 3 1B Instruct (1B parameters)
- **Prompt Engineering**: Structured prompt with clear instructions to focus on the provided context
- **Context Format**: Multiple passages with section markers for better reference

## Results

The RAG system significantly outperforms the baseline model that attempts to answer questions without retrieval:

- **Baseline**: BLEU-4: ~0.1, ROUGE-L: ~0.1
- **RAG**: BLEU-4: ~0.2, ROUGE-L: ~0.2
