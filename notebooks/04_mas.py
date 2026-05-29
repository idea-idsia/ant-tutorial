import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import getpass
    import json
    import marimo as mo
    import os
    import subprocess
    import threading
    import time
    import uvicorn
    from collections.abc import AsyncGenerator
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
    from ant_ai import Agent, BaseAgent, InvocationContext, State, tool
    from ant_ai.a2a import A2AClient, A2AConfig, Colony
    from ant_ai.llm.integrations import LiteLLMChat
    from ant_ai.workflow import END, START, NodeYield, Workflow
    from utils import stream_output

    return (
        A2AClient,
        A2AConfig,
        Agent,
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentSkill,
        AsyncGenerator,
        BaseAgent,
        Colony,
        END,
        InvocationContext,
        LiteLLMChat,
        NodeYield,
        START,
        State,
        Workflow,
        getpass,
        json,
        mo,
        os,
        stream_output,
        subprocess,
        threading,
        time,
        tool,
        uvicorn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 — Multi-Agent System (Colony)

    Build a **Colony** with two collaborating agents: a **Codegen** agent that writes Python code and delegates to a **Testgen** agent that writes pytest tests for it.

    ```
    User ──► Codegen :9002 ──► Testgen :9003
                 (writes code, then asks testgen for tests)
    ```

    What this notebook covers:
    1. A `run_python` tool shared by both agents to execute code
    2. Two specialised agents assembled into a `Colony`
    3. Starting both A2A servers in the background
    4. Interacting via `curl` and the Python client
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Install & API Key
    """)
    return


@app.cell
def _():
    # '%pip install ant-ai --quiet' command supported automatically in marimo
    return


@app.cell
def _(getpass, os):
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Tool — `run_python`

    Both agents can execute Python snippets to validate code or run tests.
    """)
    return


@app.cell
def _(tool):
    import sys
    import io
    import traceback

    @tool
    def run_python(code: str) -> str:
        """Execute a Python code snippet and return stdout, stderr, or any exception."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf
        try:
            exec(compile(code, "<string>", "exec"), {})  # noqa: S102
        except Exception:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            return traceback.format_exc()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        return (out + err).strip() or "(no output)"

    print("Tool:", run_python.name)
    return (run_python,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Define the Agents

    - **Codegen** (port 9002): writes Python code for a given task, then delegates to Testgen to produce tests.
    - **Testgen** (port 9003): writes pytest tests for code it receives. The Colony wires a remote `Codegen` tool into Testgen so it can ask Codegen to fix code if tests fail.
    """)
    return


@app.cell
def _(Agent, LiteLLMChat, run_python):
    codegen_agent = Agent(
        name="Codegen",
        system_prompt=(
            "You are an expert Python developer. "
            "When asked to implement something, write clean, correct Python code. "
            "Use run_python to verify your code runs without errors before replying. "
            "After verifying, delegate to the Testgen agent to write pytest tests for the code. "
            "Return both the implementation and the tests."
        ),
        llm=LiteLLMChat("gpt-4o-mini"),
        tools=[run_python],  # Testgen A2A tool is injected by Colony
    )

    testgen_agent = Agent(
        name="Testgen",
        system_prompt=(
            "You are an expert in writing pytest test suites. "
            "Given Python code, write thorough pytest tests covering the happy path and edge cases. "
            "Use run_python to execute the tests and confirm they pass before replying."
        ),
        llm=LiteLLMChat("gpt-4o-mini"),
        tools=[run_python],
    )

    print(f"codegen_agent tools: {[t.name for t in codegen_agent.tools]}")
    print(
        f"testgen_agent tools: {[t.name for t in testgen_agent.tools]} (before Colony wiring)"
    )
    return codegen_agent, testgen_agent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Workflows

    Both agents use the same minimal single-node workflow.
    """)
    return


@app.cell
def _(
    AsyncGenerator,
    BaseAgent,
    END,
    InvocationContext,
    NodeYield,
    START,
    State,
    Workflow,
):
    async def answer(
        agent: BaseAgent, state: State, ctx: InvocationContext | None
    ) -> AsyncGenerator[NodeYield, None]:
        async for event in agent.stream(state, ctx=ctx):
            yield event
        yield state

    def make_workflow() -> Workflow:
        wf = Workflow()
        wf.add_node("answer", answer)
        wf.add_edge(START, "answer")
        wf.add_edge("answer", END)
        return wf

    codegen_workflow = make_workflow()
    testgen_workflow = make_workflow()
    print("Workflows ready.")
    return codegen_workflow, testgen_workflow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · AgentCards
    """)
    return


@app.cell
def _(AgentCapabilities, AgentCard, AgentInterface, AgentSkill):
    CODEGEN_HOST, CODEGEN_PORT = "127.0.0.1", 9002
    TESTGEN_HOST, TESTGEN_PORT = "127.0.0.1", 9003
    CODEGEN_URL = f"http://{CODEGEN_HOST}:{CODEGEN_PORT}/"
    TESTGEN_URL = f"http://{TESTGEN_HOST}:{TESTGEN_PORT}/"

    codegen_card = AgentCard(
        name="Codegen",
        description="Writes and validates Python code, then delegates to Testgen for tests.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=CODEGEN_URL)
        ],
        skills=[
            AgentSkill(
                id="codegen",
                name="Code Generation",
                description="Implement Python functions or modules from a description.",
                tags=["python", "codegen", "implementation"],
            )
        ],
    )

    testgen_card = AgentCard(
        name="Testgen",
        description="Writes and runs pytest test suites for Python code.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=TESTGEN_URL)
        ],
        skills=[
            AgentSkill(
                id="testgen",
                name="Test Generation",
                description="Generate and validate pytest tests for a given Python implementation.",
                tags=["python", "pytest", "testing"],
            )
        ],
    )

    print(f"codegen_card → {CODEGEN_URL}")
    print(f"testgen_card → {TESTGEN_URL}")
    return (
        CODEGEN_HOST,
        CODEGEN_PORT,
        CODEGEN_URL,
        TESTGEN_HOST,
        TESTGEN_PORT,
        TESTGEN_URL,
        codegen_card,
        testgen_card,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Assemble the Colony

    `colony.collab("codegen", "testgen")` injects a remote `Testgen` A2A tool into the Codegen agent's registry at `asgi()` time.
    """)
    return


@app.cell
def _(
    CODEGEN_URL,
    Colony,
    TESTGEN_URL,
    codegen_agent,
    codegen_card,
    codegen_workflow,
    testgen_agent,
    testgen_card,
    testgen_workflow,
):
    colony = Colony()
    colony.agent(
        "codegen", agent=codegen_agent, workflow=codegen_workflow, card=codegen_card
    )
    colony.agent(
        "testgen", agent=testgen_agent, workflow=testgen_workflow, card=testgen_card
    )
    colony.collab("codegen", "testgen")  # codegen → testgen (unidirectional)

    codegen_app = colony.asgi(agent_name="codegen", use_fastapi=True)
    testgen_app = colony.asgi(agent_name="testgen", use_fastapi=True)

    print("Colony assembled:")
    print(f"  codegen → {CODEGEN_URL}")
    print(f"  testgen → {TESTGEN_URL}")
    print(
        f"  codegen registry after wiring: {[t.name for t in codegen_agent.registry.tools]}"
    )
    return codegen_app, testgen_app


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Start the Servers
    """)
    return


@app.cell
def _(
    CODEGEN_HOST,
    CODEGEN_PORT,
    CODEGEN_URL,
    TESTGEN_HOST,
    TESTGEN_PORT,
    TESTGEN_URL,
    codegen_app,
    testgen_app,
    threading,
    time,
    uvicorn,
):
    def _serve(app, host: str, port: int) -> None:
        uvicorn.run(app, host=host, port=port, log_level="warning")

    threading.Thread(
        target=_serve, args=(codegen_app, CODEGEN_HOST, CODEGEN_PORT), daemon=True
    ).start()
    threading.Thread(
        target=_serve, args=(testgen_app, TESTGEN_HOST, TESTGEN_PORT), daemon=True
    ).start()

    time.sleep(2)
    print(f"Codegen running at {CODEGEN_URL}")
    print(f"Testgen running at {TESTGEN_URL}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Interact from Bash

    Fetch the agent cards or send a message via `curl`:

    ```bash
    # Fetch agent cards
    curl http://127.0.0.1:9002/.well-known/agent-card.json
    curl http://127.0.0.1:9003/.well-known/agent-card.json

    # Ask Codegen (triggers delegation to Testgen)
    curl -X POST http://127.0.0.1:9002/ \
      -H 'Content-Type: application/json' \
      -H 'A2A-Version: 1.0' \
      -d '{
        "jsonrpc": "2.0",
        "id": "1",
        "method": "SendMessage",
        "params": {
          "message": {
            "role": "ROLE_USER",
            "messageId": "msg-001",
            "parts": [{"text": "Implement a function that checks if a number is prime."}]
          }
        }
      }'
    ```
    """)
    return


@app.cell
def _(
    CODEGEN_HOST,
    CODEGEN_PORT,
    TESTGEN_HOST,
    TESTGEN_PORT,
    json,
    mo,
    subprocess,
):
    _cards = []
    for _host, _port in [(CODEGEN_HOST, CODEGEN_PORT), (TESTGEN_HOST, TESTGEN_PORT)]:
        _url = f"http://{_host}:{_port}/.well-known/agent-card.json"
        _data = json.loads(
            subprocess.run(["curl", "-s", _url], capture_output=True, text=True).stdout
        )
        _cards.append(
            mo.vstack(
                [
                    mo.md(
                        f"**{_data.get('name')}** v{_data.get('version')} — `{_url}`"
                    ),
                    mo.accordion(
                        {"JSON": mo.md(f"```json\n{json.dumps(_data, indent=2)}\n```")}
                    ),
                ]
            )
        )
    mo.output.replace(mo.vstack(_cards, gap=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Python Client (streaming)

    Watch for `tool_calling` events where Codegen invokes the `Testgen` A2A tool.
    """)
    return


@app.cell
async def _(A2AClient, A2AConfig, CODEGEN_URL, stream_output):
    _prompt = "Implement dijkstra in python and a test suite."
    _client = A2AClient(config=A2AConfig(endpoint=CODEGEN_URL))
    await stream_output(_client.send_message(_prompt), _prompt)
    return


if __name__ == "__main__":
    app.run()
