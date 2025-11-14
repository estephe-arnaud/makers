# MAKERS: Multi Agent Knowledge Exploration & Retrieval System

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

**MAKERS** is an advanced multi-agent research system that autonomously searches, analyzes, and synthesizes information from complex document corpora (e.g., scientific research papers). It combines the power of Large Language Models (LLMs) with strategic tool orchestration, leveraging LangGraph for workflow orchestration, LlamaIndex for advanced Retrieval Augmented Generation (RAG), and MongoDB for vector storage and persistent state management.

### Core Capabilities

*   **Intelligent Document Ingestion**: Automated pipeline for downloading, parsing, chunking, and embedding scientific papers from ArXiv
*   **Autonomous Research Agent**: A unified RAG ReAct agent that dynamically orchestrates multiple information sources
*   **Strategic Tool Selection**: Intelligent decision-making between ArXiv search (external) and knowledge base retrieval (internal)
*   **Deep Document Analysis**: Specialized CrewAI team for comprehensive PDF analysis with complementary agent architecture
*   **Stateful Workflows**: Persistent, resumable research sessions with MongoDB checkpointing
*   **Comprehensive Evaluation**: Built-in metrics for RAG performance and synthesis quality assessment

## ✨ Key Features

### 🔍 Multi-Source Information Retrieval
- **ArXiv Search**: Access to latest scientific papers and research
- **Knowledge Base RAG**: Fast, precise retrieval from curated internal documents
- **Intelligent Fusion**: Automatic deduplication and relevance-based merging of information sources

### 🤖 Autonomous Agent Architecture
- **Unified RAG ReAct Agent**: Single agent handling end-to-end research workflows
- **Strategic Tool Orchestration**: Context-aware decision-making for optimal tool selection
- **Iterative Reasoning**: ReAct pattern enabling dynamic adaptation to query complexity

### 📄 Specialized PDF Analysis
- **Two-Agent CrewAI System**: Complementary architecture with clear separation of concerns
  - **Deep Document Analyst**: Analytical precision (temperature: 0.2) for structured extraction
  - **Research Report Synthesizer**: Creative synthesis (temperature: 0.4) for comprehensive reports

### 💾 Production-Ready Infrastructure
- **MongoDB Integration**: Vector store with Atlas Vector Search + persistent checkpointing
- **Multi-LLM Support**: Centralized factory supporting OpenAI, Hugging Face, and Ollama
- **Experiment Tracking**: Weights & Biases integration for metrics and evaluation
- **FastAPI API**: RESTful interface for system integration

## 🏗️ Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER REQUEST                                      │
│                    (CLI / API / Jupyter Notebook)                           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH WORKFLOW                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Entry Point: rag_agent Node                                          │   │
│  │  • Receives user query                                                │   │
│  │  • Initializes GraphState with checkpointing                         │   │
│  │  • State persisted in MongoDB                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAG REACT AGENT                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Autonomous Decision-Making Engine                                    │   │
│  │  • Context Analysis: Evaluates query complexity                     │   │
│  │  • Strategic Planning: Determines tool usage sequence              │   │
│  │  • ReAct Pattern: Reasoning + Acting loop                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Tool 1:         │  │  Tool 2:         │  │  Tool 3:         │
    │  arxiv_search    │  │  knowledge_base  │  │  document_deep   │
    │  _tool           │  │  _retrieval_tool │  │  _dive_analysis  │
    │                  │  │                  │  │  _tool           │
    │  • Search ArXiv  │  │  • Query MongoDB │  │                  │
    │  • Find papers   │  │    vector store  │  │  • Download PDF   │
    │  • Return        │  │  • Retrieve      │  │  • Extract text   │
    │    metadata      │  │    embeddings    │  │  • Trigger CrewAI │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                      │
             └─────────┬──────────┴──────────┬───────────┘
                        │                      │
                        ▼                      ▼
        ┌──────────────────────────────────────────────┐
        │     Agent Reasoning & Decision Loop          │
        │  • Evaluates tool outputs                   │
        │  • Determines if more information needed    │
        │  • Selects next action (tool or synthesis)  │
        │  • Max iterations: 15                       │
        └──────────────────┬───────────────────────────┘
                           │
                           │ (if deep analysis needed)
                           ▼
        ┌──────────────────────────────────────────────┐
        │         CREWAI PDF ANALYSIS PIPELINE         │
        │  ┌────────────────────────────────────────┐  │
        │  │  Agent 1: Deep Document Analyst         │  │
        │  │  • Temperature: 0.2 (analytical)        │  │
        │  │  • Extract structured information      │  │
        │  │  • Perform critical analysis           │  │
        │  └──────────────┬──────────────────────────┘  │
        │                 │                             │
        │                 ▼                             │
        │  ┌────────────────────────────────────────┐  │
        │  │  Agent 2: Research Report Synthesizer  │  │
        │  │  • Temperature: 0.4 (creative)        │  │
        │  │  • Synthesize analysis                 │  │
        │  │  • Compile comprehensive report         │  │
        │  └──────────────┬──────────────────────────┘  │
        └─────────────────┼──────────────────────────────┘
                          │
                          │ (returns to RAG agent)
                          ▼
        ┌──────────────────────────────────────────────┐
        │     Agent Synthesis Phase                    │
        │  • Integrates all information sources       │
        │  • Structures coherent report               │
        │  • Executive Summary                         │
        │  • Key Developments                          │
        │  • Emerging Trends                          │
        │  • Applications & Impact                     │
        │  • Challenges & Future Outlook              │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │         STATE CHECKPOINTING                   │
        │  • Final state saved to MongoDB              │
        │  • Thread ID for resumability                │
        │  • Error handling & recovery                 │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │            FINAL OUTPUT                      │
        │   Comprehensive Research Report              │
        │   Returned to User (CLI/API/Notebook)        │
        └──────────────────────────────────────────────┘
```

### Architecture Highlights

#### 1. **Autonomous Agent Design**
The RAG ReAct agent implements a sophisticated decision-making framework that dynamically adapts its strategy based on query complexity and available information sources. This design eliminates the need for rigid, sequential workflows while maintaining high-quality outputs.

**Key Decision Logic:**
- **For recent/emerging topics**: Prioritizes ArXiv search for latest papers
- **For established concepts**: Starts with knowledge base for fast, precise access
- **For comprehensive research**: Uses both sources intelligently
- **Best Practice**: Checks knowledge base first for established work, then supplements with ArXiv for recent developments

#### 2. **Complementary Agent Architecture**
The two-agent CrewAI system is engineered with clear separation of concerns:
- **Analytical Agent** (temp: 0.2): Focuses on extraction and critical evaluation
- **Synthesis Agent** (temp: 0.4): Transforms technical analysis into accessible reports

This architecture optimizes both computational efficiency and output quality.

#### 3. **Stateful Workflow Management**
LangGraph's checkpointing mechanism enables robust handling of long-running research tasks, with full state persistence and resumability. This is critical for production deployments where reliability and fault tolerance are essential.

### Strategic Information Retrieval

The system implements an intelligent hybrid approach combining:

1. **ArXiv Search (External)**
   - Access to latest research papers
   - Real-time discovery of new publications
   - Metadata-rich results with summaries

2. **Knowledge Base RAG (Internal)**
   - Fast retrieval from curated documents
   - Semantic search with embeddings
   - Metadata filtering capabilities

3. **Intelligent Fusion**
   - Automatic deduplication by `arxiv_id`
   - Relevance-based prioritization
   - Context-aware source selection

**Decision Strategy:**
```
Query Analysis → Source Selection → Tool Execution → Result Fusion → Synthesis
```

## 📊 Architecture Analysis & Optimization

### Current Architecture Strengths

✅ **Complementary Design**
- ArXiv provides access to recent research (not yet in KB)
- RAG enables fast, precise access to processed documents
- Natural synergy between external and internal sources

✅ **Flexibility**
- Agent dynamically decides which source to use
- Can combine both sources as needed
- Adapts to query characteristics

✅ **Efficiency**
- RAG is fast for known documents
- ArXiv for new discoveries
- Deep analysis only on most relevant papers (1-3)

### Optimization Opportunities

#### 🔥 High Impact / Low Effort
1. **Intelligent Deduplication**: Check if ArXiv paper is already in KB before deep analysis
2. **ArXiv Caching**: Avoid repeated searches for same queries
3. **Enhanced Prompt Guidelines**: More precise decision-making instructions (✅ **Implemented**)

#### ⚡ High Impact / Medium Effort
4. **Web Search Tool**: Add general web search (Tavily, Serper) for broader queries
5. **Explicit Decision Logic**: Helper functions to guide agent decisions

#### 🎯 Medium Impact
6. **Hybrid Search Tool**: Combined ArXiv + KB search in single operation
7. **Quality Metrics**: Tracking to optimize decision-making

### Architecture Verdict

**Status**: ✅ **SOLID architecture with optimization potential**

The current design effectively combines external search (ArXiv) with internal RAG, providing a flexible and efficient research system. The autonomous agent architecture enables intelligent tool orchestration while maintaining simplicity and maintainability.

## 🔍 Project Audit & Code Quality

### Module Status

#### ✅ Core Modules (All Active)
- `src/data_processing/` - **ESSENTIAL**: Ingestion pipeline used by `run_ingestion.py`
- `src/agents/tool_definitions.py` - **ACTIVE**: Tools used by RAG agent
- `src/agents/crewai_teams/document_analysis_crew.py` - **ACTIVE**: Called by `document_deep_dive_analysis_tool`
- `src/rag/retrieval_engine.py` - **ACTIVE**: Used by `knowledge_base_retrieval_tool`
- `src/vector_store/mongodb_manager.py` - **ACTIVE**: Used by ingestion pipeline
- `src/llm_services/llm_factory.py` - **ACTIVE**: Centralized, used everywhere
- `src/graph/checkpointer.py` - **ACTIVE**: MongoDB checkpointing for LangGraph
- `src/graph/main_workflow.py` - **ACTIVE**: Main workflow
- `src/evaluation/` - **ACTIVE**: RAG and synthesis evaluation
- `src/api/` - **ACTIVE**: FastAPI application

#### ✅ CLI Scripts (All Useful)
- `src/scripts/run_ingestion.py` - **ESSENTIAL**: Populates knowledge base
- `src/scripts/run_makers.py` - **ESSENTIAL**: Main CLI interface
- `src/scripts/run_evaluation.py` - **USEFUL**: System evaluation

#### 📝 Legacy Agents (Preserved for Compatibility)
The following functions are kept for backward compatibility with notebooks and tests:
- `create_research_planner_agent()` 
- `create_arxiv_search_agent()`
- `create_document_analysis_agent()`
- `create_synthesis_agent()`

**Reason**: Used in demonstration notebooks and module tests. The main workflow now uses only the unified RAG ReAct agent.

### Code Quality Status

✅ **Architecture Coherent**: All modules correctly connected
✅ **No Dead Code**: All files have utility
✅ **Clean Imports**: No unused imports detected
✅ **Simplified Workflow**: Optimal architecture with single ReAct agent
✅ **Production Ready**: Consistent and well-structured

**Overall Status**: ✅ **PROJECT COHERENT AND PRODUCTION-READY**

## 🛠️ Tech Stack

### Core Technologies
- **Language**: Python 3.11+
- **LLM Orchestration**: LangGraph
- **Specialized Agents**: CrewAI (for deep document analysis)
- **RAG & Indexing**: LlamaIndex
- **LLM Framework**: LangChain
- **Vector Database**: MongoDB Atlas
- **Experiment Tracking**: Weights & Biases
- **API**: FastAPI, Uvicorn

### LLM & Embedding Providers
- **OpenAI**: GPT models + embeddings
- **Hugging Face**: Open-source models via API
- **Ollama**: Local LLM deployment

### Centralized Configuration
The `src/llm_services/llm_factory.py` module centralizes LLM instantiation for all components, ensuring consistent configuration and easy provider switching.

## 📁 Directory Structure

```
makers/
├── config/              # Configuration files (settings, logging)
├── data/                # Local data (corpus, evaluation datasets)
├── notebooks/           # Jupyter notebooks for demonstration
├── src/                 # Source code
│   ├── agents/          # Agent architectures and tool definitions
│   │   └── crewai_teams/ # CrewAI sub-task definitions
│   ├── api/             # FastAPI application
│   ├── data_processing/ # Data ingestion and preprocessing
│   ├── evaluation/      # Evaluation and W&B logging
│   ├── graph/           # LangGraph workflow definition
│   ├── llm_services/    # Centralized LLM factory
│   ├── rag/             # RAG engine (LlamaIndex)
│   ├── scripts/         # CLI scripts
│   └── vector_store/    # MongoDB management
├── .env.example         # Environment variables template
├── pyproject.toml       # Poetry dependencies
├── Dockerfile           # Docker image definition
└── README.md            # This file
```

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.11+**
- **Poetry**: Dependency management ([Installation Guide](https://python-poetry.org/docs/#installation))
- **MongoDB**: Atlas or local instance
- **Ollama** (default): Ensure Ollama server is running

### Step 1: Clone Repository

```bash
git clone https://github.com/estephe-arnaud/makers
cd makers
```

### Step 2: Configure Environment

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Configure MongoDB connection:
   ```env
   MONGODB_URI=mongodb://localhost:27017  # or your Atlas connection string
   MONGO_DATABASE_NAME=makers_db
   ```

3. (Optional) Configure LLM provider:
   ```env
   # For OpenAI
   DEFAULT_LLM_MODEL_PROVIDER=openai
   OPENAI_API_KEY=your_key_here
   
   # For Ollama (default)
   DEFAULT_LLM_MODEL_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   ```

### Step 3: Install Dependencies

```bash
poetry install
```

This creates a virtual environment and installs all dependencies. For production-only installation:
```bash
poetry install --no-dev
```

### Step 4: (Optional) Connect to Weights & Biases

```bash
poetry run wandb login
```

## 🚀 Usage

All commands should be executed from the project root directory using `poetry run` to ensure execution in the Poetry-managed virtual environment.

### 1. Data Ingestion

Populate your MongoDB knowledge base with ArXiv papers:

```bash
poetry run python -m src.scripts.run_ingestion \
  --query "explainable AI in robotics" \
  --max_results 10
```

**Options:**
- `--query`: Natural language query for corpus naming and ArXiv search
- `--arxiv_keywords`: Optimized keywords for ArXiv (e.g., "deep learning AND robotics")
- `--max_results`: Maximum papers to download (default: 10)
- `--sort_by`: Sort criterion (relevance, lastUpdatedDate, submittedDate)
- `--skip_download`: Use existing PDFs without downloading

### 2. Run MAKERS Workflow

Submit a research query to the autonomous agent system:

```bash
poetry run python -m src.scripts.run_makers \
  --query "What are the latest advancements in using large language models for robot task planning?"
```

**Options:**
- `--query` / `-q`: Research query to process (required)
- `--thread_id` / `-t`: Optional thread ID to resume a previous session
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### 3. Run Evaluations

Evaluate system performance:

```bash
poetry run python -m src.scripts.run_evaluation \
  --eval_type all \
  --rag_dataset data/evaluation/rag_eval_dataset.json
```

**Options:**
- `--eval_type`: Type of evaluation (rag, synthesis, all)
- `--rag_dataset`: Path to RAG evaluation dataset
- `--synthesis_dataset`: Path to synthesis evaluation dataset
- `--wandb_project`: W&B project name (default: MAKERS-Evaluations)

## 📓 Jupyter Notebooks

Interactive notebooks for demonstration and exploration:

1. **Activate Poetry environment**:
   ```bash
   poetry shell
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Available Notebooks**:
   - `00_setup_environment.ipynb`: Environment setup and configuration
   - `01_data_ingestion_and_embedding.ipynb`: Data pipeline demonstration
   - `02_rag_strategies_exploration.ipynb`: RAG strategies exploration
   - `03_agent_development_and_tooling.ipynb`: Agent and tool development
   - `04_langgraph_workflow_design.ipynb`: Workflow design patterns
   - `05_crewai_team_integration.ipynb`: CrewAI integration examples
   - `06_end_to_end_pipeline_test.ipynb`: Complete pipeline testing
   - `07_evaluation_and_logging.ipynb`: Evaluation and metrics logging

## 🔮 Future Work

### High Priority
- **Web Search Integration**: Add general web search tool (Tavily, Serper) for broader queries
- **Intelligent Deduplication**: Automatic detection of papers already in knowledge base
- **ArXiv Caching**: Cache search results to reduce API calls and latency

### Medium Priority
- **Advanced RAG Strategies**: Parent Document Retriever, hybrid search, query expansion
- **Citation Network Analysis**: Analyze paper citations and relationships
- **Real-time Data Sources**: Integration with live research feeds

### Enhancement Opportunities
- **Streaming Responses**: Real-time output streaming for API
- **Batch Processing**: Handle multiple queries efficiently
- **Advanced Caching**: Multi-level caching for documents and embeddings
- **Comprehensive Testing**: Test suites for agent decision-making and tool orchestration
- **LLM-based Query Optimization**: Improve search relevance through query refinement

## 📄 License

```
MIT License

Copyright (c) 2025 Estèphe ARNAUD / MAKERS: Multi Agent Knowledge Exploration & Retrieval System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for the research community**
