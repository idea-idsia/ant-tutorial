# ant-tutorial

Interactive notebooks for learning the `ant-ai` agent framework, covering basic agents, A2A servers, skills, and multi-agent systems.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- Python 3.14 (uv will install it automatically)

## Setup

1. **Clone the repo and enter the directory**

   ```bash
   git clone <repo-url>
   cd ant-tutorial
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Configure environment variables**

   Copy `.env.example` to `.env` (or create `.env`) and fill in your API key:

   ```bash
   cp .env.example .env   # if an example file exists
   ```

   Required keys:

   | Variable | Description |
   |----------|-------------|
   | `OPENAI_API_KEY` | Your OpenAI API key |

## Running the notebooks

Start the marimo server from the project root:

```bash
uv run marimo edit
```

Then you can open the notebooks in your browser by navigating to `http://localhost:2718`.

### Available notebooks

| Notebook | Topic |
|----------|-------|
| `01_agent_and_workflow.py` | Building a basic agent and conditional workflow |
| `02_a2a_server.py` | A2A (agent-to-agent) server |
| `03_skills.py` | Equipping agents with skills |
| `04_mas.py` | Multi-agent system (Colony) |
