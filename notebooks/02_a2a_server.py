import marimo

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
    from typing import Literal
    from collections.abc import AsyncGenerator
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
    from ant_ai import Agent, BaseAgent, InvocationContext, State, tool
    from ant_ai.a2a import A2AClient, A2AConfig, A2AServer
    from ant_ai.llm.integrations import LiteLLMChat
    from ant_ai.workflow import END, START, NodeYield, Workflow, build_workflow_graph
    from utils import show_raw_response, show_response, stream_output

    return (
        A2AClient,
        A2AConfig,
        A2AServer,
        Agent,
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentSkill,
        AsyncGenerator,
        BaseAgent,
        END,
        InvocationContext,
        Literal,
        LiteLLMChat,
        NodeYield,
        START,
        State,
        Workflow,
        build_workflow_graph,
        getpass,
        json,
        mo,
        os,
        show_raw_response,
        show_response,
        stream_output,
        subprocess,
        threading,
        time,
        tool,
        uuid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 — A2A Server

    Wrap the Calculator agent from notebook 01 in an **A2A (Agent-to-Agent) server** so any HTTP client — or another agent — can talk to it over JSON-RPC.

    What this notebook covers:
    1. Creating an `AgentCard` (the agent's public identity)
    2. Wrapping the agent in a minimal `Workflow`
    3. Starting an `A2AServer` in the background
    4. Sending requests via `curl` (terminal commands and Python subprocess)
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
    ## 2 · Agent + Tools

    Same Calculator agent as notebook 01.
    """)
    return


@app.cell
def _(Agent, LiteLLMChat, tool):
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    @tool
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert a temperature from Celsius to Fahrenheit."""
        return celsius * 9 / 5 + 32

    agent: Agent = Agent(
        name="Calculator",
        system_prompt=(
            "You are a helpful calculator assistant. "
            "Use the provided tools to answer maths and unit-conversion questions."
        ),
        llm=LiteLLMChat("gpt-4o-mini"),
        tools=[add, multiply, celsius_to_fahrenheit],
    )

    print(f"Agent '{agent.name}' ready.")
    return (agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Workflow

    A `Workflow` defines the graph of steps the agent executes per request. Here we use the simplest possible workflow: a single node that streams the agent's response and then returns the updated state.

    The node function signature `(agent, state, ctx)` is the convention ant-ai uses to inject dependencies.
    """)
    return


@app.cell
def _(AsyncGenerator, BaseAgent, END, InvocationContext, NodeYield, START, State, Workflow):
    async def answer(
        agent: BaseAgent, state: State, ctx: InvocationContext | None
    ) -> AsyncGenerator[NodeYield, None]:
        async for event in agent.stream(state, ctx=ctx):
            yield event
        yield state

    workflow = Workflow()
    workflow.add_node("answer", answer)
    workflow.add_edge(START, "answer")
    workflow.add_edge("answer", END)
    print("Workflow ready.")
    return (workflow,)


@app.cell
def _(build_workflow_graph, mo, workflow):
    _graph = build_workflow_graph(workflow)
    mo.Html(_graph.pipe(format="svg").decode())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3b · Conditional Edges

    A **conditional edge** routes the workflow to a different next node based on runtime state. You provide a router function that returns a `Literal` string matching one of the target node names — ant-ai reads the return annotation to discover the possible branches.

    In this example the router inspects the incoming question: conversion questions (temperature, units) go to `"convert"`, everything else goes to `"calculate"`. Both nodes stream the same agent; the branching makes the intent explicit in the graph.
    """)
    return


@app.cell
def _(AsyncGenerator, BaseAgent, END, InvocationContext, Literal, NodeYield, START, State, Workflow):
    async def calculate(
        agent: BaseAgent, state: State, ctx: InvocationContext | None
    ) -> AsyncGenerator[NodeYield, None]:
        async for event in agent.stream(state, ctx=ctx):
            yield event
        yield state

    async def convert(
        agent: BaseAgent, state: State, ctx: InvocationContext | None
    ) -> AsyncGenerator[NodeYield, None]:
        async for event in agent.stream(state, ctx=ctx):
            yield event
        yield state

    def route_question(agent, state, ctx) -> Literal["calculate", "convert"]:
        msg = state.last_message.content.lower()
        if any(kw in msg for kw in ["celsius", "fahrenheit", "°c", "°f", "convert", "temperature"]):
            return "convert"
        return "calculate"

    cond_workflow = Workflow()
    cond_workflow.add_node("calculate", calculate)
    cond_workflow.add_node("convert", convert)
    cond_workflow.add_conditional_edge(START, route_question)  # → "calculate" or "convert"
    cond_workflow.add_edge("calculate", END)
    cond_workflow.add_edge("convert", END)
    print("Conditional workflow ready.")
    return (cond_workflow,)


@app.cell
def _(build_workflow_graph, cond_workflow, mo):
    _graph = build_workflow_graph(cond_workflow)
    mo.Html(_graph.pipe(format="svg").decode())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · AgentCard

    The `AgentCard` is the agent's public identity document — think of it as an OpenAPI spec for a single agent. It is served at `/.well-known/agent.json` and tells clients:
    - What the agent can do (`skills`)
    - How to reach it (`supported_interfaces`)
    - What protocols it speaks (`capabilities`)
    """)
    return


@app.cell
def _(AgentCapabilities, AgentCard, AgentInterface, AgentSkill):
    HOST = "127.0.0.1"
    PORT = 9000
    BASE_URL = f"http://{HOST}:{PORT}/"

    card = AgentCard(
        name="Calculator",
        description="A helpful calculator that can do arithmetic and unit conversions.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=BASE_URL)],
        skills=[
            AgentSkill(
                id="arithmetic",
                name="Arithmetic",
                description="Add, multiply, and convert units.",
                tags=["math", "calculator", "conversion"],
            )
        ],
    )

    print(f"AgentCard '{card.name}' v{card.version} → {BASE_URL}")
    return BASE_URL, HOST, PORT, card


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Start the A2A Server

    `A2AServer.serve()` calls `uvicorn.run()` internally, which is blocking. We run it in a **daemon thread** so the rest of the notebook stays interactive.

    > **Kernel restart tip:** If you restart the kernel, the daemon thread is automatically killed. Re-run cells 2–6 to bring the server back up.
    """)
    return


@app.cell
def _(A2AServer, BASE_URL, HOST, PORT, agent, card, threading, time, workflow):
    server = A2AServer(
        agent=agent,
        workflow=workflow,
        agent_card=card,
        host=HOST,
        port=PORT,
    )

    server_thread = threading.Thread(
        target=server.serve,
        kwargs={"use_fastapi": True},
        daemon=True,
    )
    server_thread.start()

    time.sleep(2)  # give uvicorn a moment to bind the port
    print(f"Server running at {BASE_URL}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Send Requests

    The server exposes two endpoints:

    | Endpoint | Purpose |
    |---|---|
    | `GET /.well-known/agent-card.json` | Fetch the AgentCard |
    | `POST /` | Send a JSON-RPC message to the agent |

    We use Python's `subprocess` to run `curl` so you can see the exact commands. Copy them into any terminal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6a · Fetch the AgentCard
    """)
    return


@app.cell
def _(HOST, PORT, json, mo, subprocess):
    AGENT_CARD_URL = f"http://{HOST}:{PORT}/.well-known/agent-card.json"

    result = subprocess.run(["curl", "-s", AGENT_CARD_URL], capture_output=True, text=True)
    if not result.stdout:
        mo.output.replace(mo.callout(mo.md(f"Empty response: `{result.stderr}`"), kind="danger"))
    else:
        _data = json.loads(result.stdout)
        mo.output.replace(mo.vstack([
            mo.md(f"**{_data.get('name')}** v{_data.get('version')} — `{AGENT_CARD_URL}`"),
            mo.accordion({"JSON": mo.md(f"```json\n{json.dumps(_data, indent=2)}\n```")}),
        ]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6b · Send a Message (blocking)

    The A2A v1.0 protocol uses **JSON-RPC 2.0** with the `SendMessage` method for a blocking call that waits for the final answer before returning. Use `SendStreamingMessage` (with `Accept: text/event-stream`) for a streaming call.

    ```
    curl -X POST http://127.0.0.1:9000/ \
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
            "parts": [{"text": "What is 3 + 4?"}]
          }
        }
      }'
    ```
    """)
    return


@app.cell
def _(HOST, PORT, json, subprocess, uuid):
    RPC_URL = f"http://{HOST}:{PORT}/"

    def send_message(text: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
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
                "curl",
                "-s",
                "-X",
                "POST",
                RPC_URL,
                "-H",
                "Content-Type: application/json",
                "-H",
                "A2A-Version: 1.0",
                "-d",
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    return (send_message,)


@app.cell
def _(send_message, show_raw_response):
    show_raw_response(send_message("What is 3 + 4?"))
    return


@app.cell
def _(send_message, show_response):
    show_response(send_message("What is 12 multiplied by 8, and what is 37 °C in Fahrenheit?"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the response

    The JSON-RPC response wraps a **Task** object. The agent's reply is the last entry in `history`:

    ```
    response["result"]["task"]["history"][-1]["parts"][0]["text"]
    ```
    """)
    return


@app.cell
def _(send_message, show_response):
    show_response(send_message("What is 15 + 27?"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Using the Python Client

    Instead of raw `curl` / `subprocess`, ant-ai ships an `A2AClient` that handles JSON-RPC framing and returns typed `Event` objects. Each event has a `kind` (e.g. `final_answer`, `tool_calling`) and a `content` field with the text.

    ```python
    from ant_ai.a2a import A2AClient, A2AConfig

    client = A2AClient(config=A2AConfig(endpoint=BASE_URL))

    async for event in client.send_message("What is 3 + 4?"):
        if event.content:
            print(event.kind, event.content)
    ```
    """)
    return


@app.cell
async def _(A2AClient, A2AConfig, BASE_URL, stream_output):
    _prompt = "What is 3 + 4?"
    _client = A2AClient(config=A2AConfig(endpoint=BASE_URL))
    await stream_output(_client.send_message(_prompt), _prompt)
    return


@app.cell
async def _(A2AClient, A2AConfig, BASE_URL, stream_output):
    _prompt = "What is 12 multiplied by 8, and what is 37 °C in Fahrenheit?"
    _client = A2AClient(config=A2AConfig(endpoint=BASE_URL))
    await stream_output(_client.send_message(_prompt), _prompt)
    return


if __name__ == "__main__":
    app.run()
