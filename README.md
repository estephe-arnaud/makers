# MAKERS: Multi Agent Knowledge Exploration & Retrieval System

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

**MAKERS** is an advanced autonomous research system that combines Large Language Models (LLMs) with strategic tool orchestration. It leverages **LangGraph** for workflow orchestration, **LlamaIndex** for Retrieval Augmented Generation (RAG), **ChromaDB** for vector storage, and **SQLite** for persistent state management (LangGraph checkpoints).

### Core Capabilities

*   **Autonomous Research Agent**: Unified ReAct agent that dynamically orchestrates multiple information sources
*   **Multi-Source Retrieval**: Intelligent decision-making between ArXiv search (external) and knowledge base RAG (internal)
*   **Deep Document Analysis**: Specialized CrewAI team for comprehensive PDF analysis
*   **Stateful Workflows**: Persistent, resumable research sessions with SQLite checkpointing
*   **Long-Term Memory**: Conversation summarization prevents prompt explosion while preserving context

## 🏗️ Architecture

### System Architecture

The system implements a **multi-node LangGraph workflow** with separated concerns for agent reasoning, tool execution, and memory management:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                      USER REQUEST (CLI/API)                               ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║                    LANGGRAPH WORKFLOW                                     ║
║         StateGraph with SQLite Checkpointing                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  ╔═════════════════════════════════════════════════════════════════════╗  ║
║  ║                      AGENT NODE                                     ║  ║
║  ║                    (Entry Point)                                    ║  ║
║  ╠═════════════════════════════════════════════════════════════════════╣  ║
║  ║    Input:                                                           ║  ║
║  ║     • conversation_summary (long-term memory)                       ║  ║
║  ║     • recent messages (last 3, immediate context)                   ║  ║
║  ║                                                                     ║  ║
║  ║    Process:                                                         ║  ║
║  ║     • ReAct Agent analyzes context                                  ║  ║
║  ║     • Decision: tool_calls OR final_answer                          ║  ║
║  ║                                                                     ║  ║
║  ║    Output:                                                          ║  ║
║  ║     • AIMessage with tool_calls[] OR content (final answer)         ║  ║
║  ╚═════════════════════════════════════════════════════════════════════╝  ║
║                              │                                            ║
║                              │ route_after_agent                          ║
║                              │                                            ║
║              ┌───────────────┴───────────────┐                            ║
║              │                               │                            ║
║              ▼                               ▼                            ║
║  ╔═══════════════════════╗      ╔═══════════════════════╗                 ║
║  ║      TOOL NODE        ║      ║         END           ║                 ║
║  ║   (if tool_calls)     ║      ║   (if final_answer)   ║                 ║
║  ╠═══════════════════════╣      ╠═══════════════════════╣                 ║
║  ║  • Extract tool_calls ║      ║  • Return final_state ║                 ║
║  ║  • Get from Registry  ║      ║  • Output: final_     ║                 ║
║  ║  • Execute tools:     ║      ║    output             ║                 ║
║  ║    - arxiv_search     ║      ╚═══════════════════════╝                 ║
║  ║    - knowledge_base   ║                                                ║
║  ║    - document_analysis║                                                ║
║  ║  • Return ToolMessage ║                                                ║
║  ╚═══════════════════════╝                                                ║
║              │                                                            ║
║              │ route_after_tool                                           ║
║              │                                                            ║
║              └───────────┬───────────────┐                                ║
║                          │               │                                ║
║                          ▼               ▼                                ║
║          ╔═══════════════════════╗  ╔═══════════════════════╗             ║
║          ║     SUMMARY NODE      ║  ║      AGENT NODE       ║             ║
║          ║   (if msg_count ≥ 20) ║  ║   (if msg_count < 20) ║             ║
║          ╠═══════════════════════╣  ╠═══════════════════════╣             ║
║          ║  • Take: messages +   ║  ║  • Continue workflow  ║             ║
║          ║    existing summary   ║  ║  • Process tool       ║             ║
║          ║  • Generate condensed ║  ║    results            ║             ║
║          ║    summary            ║  ║                       ║             ║
║          ║  • Preserve findings  ║  ║                       ║             ║
║          ║  • Clear old msgs     ║  ║                       ║             ║
║          ║  • Keep last 3 msgs   ║  ║                       ║             ║
║          ╚═══════════════════════╝  ╚═══════════════════════╝             ║
║                      │                                                    ║
║                      │ route_after_summary                                ║
║                      │                                                    ║
║                      └───────────┬                                        ║
║                                  │                                        ║
║                                  ▼                                        ║
║                          ╔═══════════════╗                                ║
║                          ║   AGENT NODE  ║                                ║
║                          ║  (Loop back)  ║                                ║
║                          ╚═══════════════╝                                ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                          ╔═══════════════════════╗
                          ║      FINAL OUTPUT     ║
                          ║  GraphState with      ║
                          ║  final_output         ║
                          ╚═══════════════════════╝
```

### Key Components

1. **Agent Node** (`agentic/workflow/nodes/agent_node.py`):
   - **Input**: `conversation_summary` (string, long-term memory) + `messages` (list, recent context)
   - **Process**: ReAct agent (LangChain) analyzes context and decides on action
   - **Output**: `AIMessage` with `tool_calls` (list) OR `content` (final answer string)
   - **State Updates**: `messages`, `next_action`, `final_output`, `iteration_count`
   - **Safety**: Maximum iteration limit (50) to prevent infinite loops

2. **Tool Node** (`agentic/workflow/nodes/tool_node.py`):
   - **Input**: `messages` (extracts `tool_calls` from last `AIMessage`)
   - **Process**: Retrieves tools from `ToolRegistry`, executes each tool call
   - **Tools Available**: `arxiv_search_tool`, `knowledge_base_retrieval_tool`
   - **Note**: `document_deep_dive_analysis_tool` is not used for security reasons (prevents automatic PDF downloads)
   - **Output**: `ToolMessage` list with execution results
   - **Error Handling**: Graceful failure with error messages in `ToolMessage`
   - **Next Step**: Always routes to either `summarize` (if message_count >= 20) or `agent` (continue)

3. **Summary Node** (`agentic/workflow/nodes/summary_node.py`):
   - **Trigger**: Only after `tool_node` when `len(messages) >= SUMMARY_THRESHOLD` (20 messages)
   - **Input**: Recent messages + existing `conversation_summary`
   - **Process**: LLM-based summarization (temperature: 0.1 for factual accuracy)
   - **Output**: Condensed summary preserving key findings
   - **Memory Management**: Clears old messages, keeps last 3 for immediate context
   - **Next Step**: Always routes back to `agent` with updated summary

4. **Routing Logic** (`agentic/workflow/routing.py`):
   - **route_after_agent**: Routes to `tool` (if tool_calls), `continue` (if unclear), or `end` (if final_answer)
   - **route_after_tool**: Routes to `summarize` (if message_count >= 20) or `agent` (if message_count < 20)
   - **route_after_summary**: Always routes back to `agent` with updated summary
   - **Note**: Summary node can only be reached after tool node execution, never in parallel

5. **State Management** (`core/state.py`):
   - **GraphState**: TypedDict with `messages`, `conversation_summary`, `user_query`, `final_output`, `next_action`, `error_message`, `iteration_count`
   - **Checkpointing**: SQLite-based persistence via `services/storage/checkpointer.py`
   - **Resumability**: Thread-based state recovery for long-running sessions

### Tool Registry Architecture

The system uses a **modular Tool Registry** pattern:
- Tools are registered in `src/agentic/tools/registry.py`
- `tool_node.py` retrieves tools dynamically (no hardcoding)
- Easy to add/remove tools without modifying workflow code

### Information Retrieval Strategy

**Hybrid Approach:**
- **ArXiv Search**: Access to latest scientific papers (external)
- **Knowledge Base RAG**: Fast retrieval from curated documents (internal)
- **Intelligent Fusion**: Automatic deduplication and relevance-based merging

**Decision Logic:**
- Recent/emerging topics → Prioritize ArXiv
- Established concepts → Start with knowledge base
- Comprehensive research → Use both sources intelligently

## 🛠️ Tech Stack

- **LLM Orchestration**: LangGraph
- **Agent Framework**: LangChain (ReAct pattern)
- **Specialized Analysis**: CrewAI (two-agent complementary architecture)
- **RAG & Indexing**: LlamaIndex
- **Vector Database**: ChromaDB (local, with automatic cosine similarity indexing)
- **State Management**: SQLite (for LangGraph checkpoints, local, no server required)
- **LLM Providers**: OpenAI, Hugging Face, Ollama, Groq, Google Gemini (centralized factory)
- **Embedding Providers**: HuggingFace (default, local/unlimited/free), Ollama (local), OpenAI (API)
- **API**: FastAPI, Uvicorn
- **Experiment Tracking**: Weights & Biases

## 📁 Directory Structure

```
src/
├── services/        # Reusable technical services
│   ├── llm.py       # LLM factory (OpenAI, HuggingFace, Ollama, Groq, Google Gemini)
│   ├── storage/     # ChromaDB (vector store), SQLite (checkpoints), Checkpointer
│   ├── ingestion/   # Data ingestion pipeline
│   └── evaluation/  # Evaluation services (RAG, synthesis)
├── agentic/         # Agentic system
│   ├── agents/      # Agents, prompts, and constants
│   ├── tools/       # Tools with registry pattern
│   └── workflow/    # LangGraph workflow (graph, runner, nodes, routing, state, constants)
└── application/     # User interfaces
    ├── api/         # FastAPI REST API
    └── cli/         # Command-line interface scripts
```

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.11+**
- **Poetry**: Dependency management (required)
- **Groq API Key** (default): Get your free API key from [console.groq.com](https://console.groq.com)

### Installation

1. **Clone repository**:
   ```bash
   git clone https://github.com/estephe-arnaud/makers
   cd makers
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   ```
   
          Edit `.env`:
          ```env
          # LLM Provider (default: Groq - unlimited/free tier)
          DEFAULT_LLM_MODEL_PROVIDER=groq
          GROQ_API_KEY=your_groq_api_key_here
          GROQ_MODEL_NAME=llama-3.3-70b-versatile
          
          # Embedding Provider (default: HuggingFace - local, unlimited, free)
          DEFAULT_EMBEDDING_PROVIDER=huggingface
          HUGGINGFACE_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
          
          # Alternative LLM Providers:
          # DEFAULT_LLM_MODEL_PROVIDER=google
          # GOOGLE_API_KEY=your_google_api_key_here
          # GOOGLE_GEMINI_MODEL_NAME=gemini-pro
          
          # DEFAULT_LLM_MODEL_PROVIDER=openai
          # OPENAI_API_KEY=your_key_here
          
          # DEFAULT_LLM_MODEL_PROVIDER=ollama
          # OLLAMA_BASE_URL=http://localhost:11434
          
          # Alternative Embedding Providers:
          # DEFAULT_EMBEDDING_PROVIDER=ollama
          # OLLAMA_EMBEDDING_MODEL_NAME=nomic-embed-text
          # OLLAMA_BASE_URL=http://localhost:11434
          
          # DEFAULT_EMBEDDING_PROVIDER=openai
          # OPENAI_API_KEY=your_key_here
          # OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small
          ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Set up ChromaDB (Vector Storage)**:

   ChromaDB is used for vector storage and is automatically configured. No setup required! The database is stored locally at `data/chroma_db/` by default.

   **Note**: ChromaDB automatically creates vector indexes with cosine similarity, so you get optimized vector search out of the box.

5. **Set up SQLite (for LangGraph checkpoints)**:

   SQLite is used for storing LangGraph conversation checkpoints (state management). No setup required! The database is automatically created at `data/checkpoints.sqlite` by default.

   **Note**: SQLite is a lightweight, serverless database that requires no configuration. All checkpoint data is stored locally in a single file.

6. **(Optional) Connect to Weights & Biases**:
   ```bash
   poetry run wandb login
   ```

## 🚀 Usage

### 1. Data Ingestion

By default, the pipeline loads PDFs from a local directory. To use local PDFs:

```bash
poetry run python -m src.application.cli.run_ingestion \
  --pdf_dir /path/to/my/pdfs
```

To download PDFs from ArXiv (requires `--download_from_arxiv`):

```bash
poetry run python -m src.application.cli.run_ingestion \
  --download_from_arxiv \
  --query "What are the latest advancements in face analysis" \
  --max_results 10
```

**Main options:**
- `--pdf_dir` (required by default): Path to a directory containing PDF files
- `--download_from_arxiv`: Enable downloading from ArXiv instead of using a local directory
- `--query`: Required with `--download_from_arxiv`, query for ArXiv search
- `--arxiv_keywords`: Optimized keywords for ArXiv (required with `--download_from_arxiv`)
- `--max_results`: Maximum number of papers to download (default: 10, only with `--download_from_arxiv`)
- `--sort_by`: Sort criterion (relevance, lastUpdatedDate, submittedDate, only with `--download_from_arxiv`)
- `--corpus_name`: Specific name for the corpus (optional)
- `--collection_name`: ChromaDB collection name (default: `arxiv_chunks`)

### 2. Run MAKERS Workflow

Submit a research query to the autonomous agent:

```bash
poetry run python -m src.application.cli.run_makers \
  --query "What are the latest advancements in face analysis"
```

**Options:**
- `--query` / `-q`: Research query (required)
- `--thread_id` / `-t`: Optional thread ID to resume a previous session
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### 3. Run Evaluations

Evaluate system performance:

```bash
poetry run python -m src.application.cli.run_evaluation \
  --eval_type all \
  --rag_dataset data/evaluation/rag_eval_dataset.json
```

**Options:**
- `--eval_type`: Type of evaluation (rag, synthesis, all)
- `--rag_dataset`: Path to RAG evaluation dataset
- `--synthesis_dataset`: Path to synthesis evaluation dataset
- `--wandb_project`: W&B project name (default: MAKERS-Evaluations)

### 4. API Server

Start the FastAPI server:

```bash
poetry run uvicorn src.application.api.main:app --reload --host 127.0.0.1 --port 8000
```

Access:
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Example API request:**
```bash
curl -X POST "http://localhost:8000/invoke_makers" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest advancements in face analysis"}'
```

## 🐳 Docker

Build and run with Docker:

```bash
# Build image
docker build -t makers-app .

# Run CLI
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
           makers-app \
           python -m src.application.cli.run_makers --query "What are the latest advancements in face analysis"

# Run API server
docker run -p 127.0.0.1:8000:8000 \
           -e OPENAI_API_KEY=$OPENAI_API_KEY \
           makers-app \
           uvicorn src.application.api.main:app --host 0.0.0.0 --port 8000
```

## 📄 License

MIT License - Copyright (c) 2025 Estèphe ARNAUD

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

