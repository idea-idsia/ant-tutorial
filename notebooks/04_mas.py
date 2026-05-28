import marimo

__generated_with = "0.23.6"
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
    import uuid
    import uvicorn
    from collections.abc import AsyncGenerator
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
    from ant_ai import Agent, BaseAgent, InvocationContext, State, tool
    from ant_ai.a2a import A2AClient, A2AConfig, Colony
    from ant_ai.llm.integrations import LiteLLMChat
    from ant_ai.workflow import END, START, NodeYield, Workflow
    from utils import show_response, stream_output

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
        show_response,
        stream_output,
        subprocess,
        threading,
        time,
        tool,
        uuid,
        uvicorn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 — Multi-Agent System (Colony)

    Build a **Colony** — a group of ant-ai agents that collaborate over the A2A protocol. Each agent runs as an independent service; they delegate tasks to each other at runtime.

    In this notebook:
    - A **Calculator** agent (port 9002) owns arithmetic tools and answers computation requests directly.
    - An **Orchestrator** agent (port 9003) receives user queries and delegates to Calculator as needed.

    The `Colony` object handles tool wiring automatically — no manual HTTP plumbing.

    ```
    User ──► Orchestrator :9003 ──► Calculator :9002 ──► tools
                   ▲                      │
                   └──────── result ◄──────┘
    ```

    What this notebook covers:
    1. Creating two specialised agents
    2. Assembling them into a `Colony` with a collaboration edge
    3. Starting both A2A servers in the background
    4. Interacting with the system via `curl` from the terminal
    5. Streaming events with the Python client
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
    ## 2 · How Colony Works

    | API | What it does |
    |---|---|
    | `Colony()` | Creates a colony (in-memory task storage by default; pass `db_url=` for Postgres) |
    | `colony.agent(name, agent=…, workflow=…, card=…)` | Registers an agent |
    | `colony.collab(src, tgt)` | Gives `src` the ability to call `tgt` as an A2A tool |
    | `colony.collab(src, tgt, mutual=True)` | Bidirectional collaboration |
    | `colony.asgi(agent_name=…)` | Builds the ASGI app and wires the inter-agent tools |

    Calling `colony.asgi("orchestrator")` is where the magic happens: ant-ai creates an `A2AAgentTool` that points at the Calculator's URL and injects it into the Orchestrator's tool registry — no code change needed in either agent.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Define Tools

    The Calculator agent gets four arithmetic tools (same set as notebooks 01-03).
    """)
    return


@app.cell
def _(tool):
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    @tool
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    @tool
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert a temperature from Celsius to Fahrenheit."""
        return celsius * 9 / 5 + 32

    print("Tools:", [t.name for t in [add, multiply, divide, celsius_to_fahrenheit]])
    return add, celsius_to_fahrenheit, divide, multiply


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Create the Agents

    - **Calculator**: uses local tools to perform calculations.
    - **Orchestrator**: starts with no tools. The Colony wires a remote `Calculator` tool into it when `colony.asgi()` is called.
    """)
    return


@app.cell
def _(Agent, LiteLLMChat, add, celsius_to_fahrenheit, divide, multiply):
    calc_agent = Agent(
        name="Calculator",
        system_prompt=(
            "You are a precise calculator assistant. "
            "Use your tools to perform arithmetic and unit conversions. "
            "Always state the numeric result."
        ),
        llm=LiteLLMChat("gpt-4o-mini"),
        tools=[add, multiply, divide, celsius_to_fahrenheit],
    )

    orchestrator_agent = Agent(
        name="Orchestrator",
        system_prompt=(
            "You are a helpful assistant. "
            "For any calculation or unit conversion, delegate to the Calculator agent using its tool. "
            "Synthesise the results into a clear, user-friendly answer."
        ),
        llm=LiteLLMChat("gpt-4o-mini"),
        tools=[],  # Calculator A2A tool is injected by Colony
    )

    print(f"calc_agent        tools: {[t.name for t in calc_agent.tools]}")
    print(f"orchestrator_agent tools: {[t.name for t in orchestrator_agent.tools]} (before Colony wiring)")
    return calc_agent, orchestrator_agent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Workflows

    Both agents use the same single-node `answer` workflow from notebook 02. We create two independent `Workflow` instances so each agent has its own execution graph.
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

    calc_workflow = make_workflow()
    orchestrator_workflow = make_workflow()
    print("Workflows ready.")
    return calc_workflow, orchestrator_workflow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · AgentCards

    The `url` in each card is the address the agent will serve at. The Colony reads this to build the `A2AAgentTool` endpoint used for delegation.
    """)
    return


@app.cell
def _(AgentCapabilities, AgentCard, AgentInterface, AgentSkill):
    CALC_HOST, CALC_PORT = "127.0.0.1", 9002
    ORCH_HOST, ORCH_PORT = "127.0.0.1", 9003
    CALC_URL = f"http://{CALC_HOST}:{CALC_PORT}/"
    ORCH_URL = f"http://{ORCH_HOST}:{ORCH_PORT}/"

    calc_card = AgentCard(
        name="Calculator",
        description="Performs arithmetic and unit conversions using tools.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=CALC_URL)],
        skills=[
            AgentSkill(
                id="arithmetic",
                name="Arithmetic",
                description="Add, multiply, divide numbers and convert units.",
                tags=["math", "calculator", "conversion"],
            )
        ],
    )

    orch_card = AgentCard(
        name="Orchestrator",
        description="Routes user queries to specialist agents and synthesises their results.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=ORCH_URL)],
        skills=[
            AgentSkill(
                id="orchestrate",
                name="Orchestrate",
                description="Handle multi-part queries by delegating to the right specialist agent.",
                tags=["orchestrator", "routing", "multi-agent"],
            )
        ],
    )

    print(f"calc_card  → {CALC_URL}")
    print(f"orch_card  → {ORCH_URL}")
    return (
        CALC_HOST,
        CALC_PORT,
        CALC_URL,
        ORCH_HOST,
        ORCH_PORT,
        ORCH_URL,
        calc_card,
        orch_card,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Assemble the Colony

    1. Register each agent with `colony.agent()`.
    2. Declare the collaboration edge with `colony.collab("orchestrator", "calc")`.
    3. Call `colony.asgi()` for each agent. This builds the ASGI app **and** wires the `A2AAgentTool` into the Orchestrator's registry.
    """)
    return


@app.cell
def _(
    CALC_URL,
    Colony,
    ORCH_URL,
    calc_agent,
    calc_card,
    calc_workflow,
    orch_card,
    orchestrator_agent,
    orchestrator_workflow,
):
    colony = Colony()  # in-memory task storage
    colony.agent("calc", agent=calc_agent, workflow=calc_workflow, card=calc_card)
    colony.agent("orchestrator", agent=orchestrator_agent, workflow=orchestrator_workflow, card=orch_card)
    colony.collab("orchestrator", "calc")  # orchestrator → calc (unidirectional)

    # Build ASGI apps — wires Calculator A2AAgentTool into orchestrator_agent
    calc_app = colony.asgi(agent_name="calc", use_fastapi=True)
    orch_app = colony.asgi(agent_name="orchestrator", use_fastapi=True)

    print("Colony assembled:")
    print(f"  calc         → {CALC_URL}")
    print(f"  orchestrator → {ORCH_URL}")
    print(f"  orchestrator registry after wiring: {[t.name for t in orchestrator_agent.registry.tools]}")
    return calc_app, orch_app


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Start the Servers

    We run each ASGI app in a **daemon thread** via `uvicorn.run()`. Daemon threads are killed automatically when the kernel stops — no cleanup needed.
    """)
    return


@app.cell
def _(
    CALC_HOST,
    CALC_PORT,
    CALC_URL,
    ORCH_HOST,
    ORCH_PORT,
    ORCH_URL,
    calc_app,
    orch_app,
    threading,
    time,
    uvicorn,
):
    def _serve(app, host: str, port: int) -> None:
        uvicorn.run(app, host=host, port=port, log_level="warning")

    threading.Thread(target=_serve, args=(calc_app, CALC_HOST, CALC_PORT), daemon=True).start()
    threading.Thread(target=_serve, args=(orch_app, ORCH_HOST, ORCH_PORT), daemon=True).start()

    time.sleep(2)  # give uvicorn a moment to bind both ports
    print(f"Calculator   running at {CALC_URL}")
    print(f"Orchestrator running at {ORCH_URL}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Interact from Bash

    Both servers expose the same A2A JSON-RPC interface from notebooks 02 and 03. Copy any of these commands into a terminal while the notebook is running.

    ### Fetch agent cards

    ```bash
    curl http://127.0.0.1:9002/.well-known/agent-card.json
    curl http://127.0.0.1:9003/.well-known/agent-card.json
    ```

    ### Talk directly to the Calculator

    ```bash
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
            "parts": [{"text": "What is 144 divided by 12?"}]
          }
        }
      }'
    ```

    ### Talk to the Orchestrator (triggers delegation)

    ```bash
    curl -X POST http://127.0.0.1:9003/ \
      -H 'Content-Type: application/json' \
      -H 'A2A-Version: 1.0' \
      -d '{
        "jsonrpc": "2.0",
        "id": "1",
        "method": "SendMessage",
        "params": {
          "message": {
            "role": "ROLE_USER",
            "messageId": "msg-002",
            "parts": [{"text": "What is 15 multiplied by 7, and what is 37 °C in Fahrenheit?"}]
          }
        }
      }'
    ```

    The Orchestrator calls the Calculator agent as a tool; the Calculator calls its local tools. The result flows back transparently.
    """)
    return


@app.cell
def _(ORCH_HOST, ORCH_PORT, json, subprocess, uuid):
    def send_message(text: str, host: str = ORCH_HOST, port: int = ORCH_PORT) -> dict:
        """Send an A2A JSON-RPC SendMessage request; return the parsed response."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "SendMessage",
            "params": {
                "message": {
                    "role": "ROLE_USER",
                    "messageId": str(uuid.uuid4()),
                    "parts": [{"text": text}],
                },
            },
        }
        result = subprocess.run(
            [
                "curl", "-s", "-X", "POST", f"http://{host}:{port}/",
                "-H", "Content-Type: application/json",
                "-H", "A2A-Version: 1.0",
                "-d", json.dumps(payload),
            ],
            capture_output=True,
            text=True,
        )
        if not result.stdout:
            return {"error": f"Empty response (stderr: {result.stderr!r})"}
        return json.loads(result.stdout)

    return (send_message,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9a · Fetch the AgentCards
    """)
    return


@app.cell
def _(CALC_HOST, CALC_PORT, ORCH_HOST, ORCH_PORT, json, mo, subprocess):
    _cards = []
    for _host, _port in [(CALC_HOST, CALC_PORT), (ORCH_HOST, ORCH_PORT)]:
        _url = f"http://{_host}:{_port}/.well-known/agent-card.json"
        _data = json.loads(subprocess.run(["curl", "-s", _url], capture_output=True, text=True).stdout)
        _cards.append(mo.vstack([
            mo.md(f"**{_data.get('name')}** v{_data.get('version')} — `{_url}`"),
            mo.accordion({"JSON": mo.md(f"```json\n{json.dumps(_data, indent=2)}\n```")}),
        ]))
    mo.output.replace(mo.vstack(_cards, gap=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9b · Talk directly to the Calculator

    Bypass the Orchestrator — send a request straight to the Calculator.
    """)
    return


@app.cell
def _(CALC_HOST, CALC_PORT, send_message, show_response):
    show_response(send_message("What is 144 divided by 12?", host=CALC_HOST, port=CALC_PORT))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9c · Talk to the Orchestrator (triggers delegation)

    The Orchestrator recognises the request requires calculation, calls the `Calculator` tool over A2A, and synthesises the final answer. The delegation is invisible to the caller.
    """)
    return


@app.cell
def _(send_message, show_response):
    show_response(send_message("What is 15 multiplied by 7? Also convert 37 °C to Fahrenheit."))
    return


@app.cell
def _(send_message, show_response):
    show_response(send_message("I bought 3 items at $12.50 each. What is the total? And what is 100 °C in Fahrenheit?"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Python Client

    Use `A2AClient` to stream events from both agents. Watch for `tool_calling` events where the Orchestrator invokes the `Calculator` agent over A2A.
    """)
    return


@app.cell
async def _(A2AClient, A2AConfig, ORCH_URL, stream_output):
    _prompt = "What is 15 multiplied by 7, and what is 37 °C in Fahrenheit?"
    _client = A2AClient(config=A2AConfig(endpoint=ORCH_URL))
    await stream_output(_client.send_message(_prompt), _prompt)
    return


@app.cell
async def _(A2AClient, A2AConfig, CALC_URL, stream_output):
    _prompt = "What is 256 divided by 16?"
    _client = A2AClient(config=A2AConfig(endpoint=CALC_URL))
    await stream_output(_client.send_message(_prompt), _prompt)
    return


if __name__ == "__main__":
    app.run()
