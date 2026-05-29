import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import getpass
    import json
    import marimo as mo
    import os
    import threading
    import time
    import uuid
    from collections.abc import AsyncGenerator
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
    from ant_ai import Agent, BaseAgent, InvocationContext, State
    from ant_ai.a2a import A2AClient, A2AConfig, A2AServer
    from ant_ai.llm.integrations import LiteLLMChat
    from ant_ai.workflow import END, START, NodeYield, Workflow
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
        LiteLLMChat,
        NodeYield,
        START,
        State,
        Workflow,
        getpass,
        json,
        mo,
        os,
        show_raw_response,
        show_response,
        stream_output,
        threading,
        time,
        uuid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 — A2A Server

    Wrap a simple agent in an **A2A (Agent-to-Agent) server** so any HTTP client — or another agent — can talk to it over JSON-RPC.

    What this notebook covers:
    1. Creating a simple agent with a workflow
    2. Creating an `AgentCard` (the agent's public identity)
    3. Starting an `A2AServer` in the background
    4. Sending requests via `curl` and the Python client
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · API Key
    """)
    return


@app.cell
def _(getpass, os):
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Agent & Workflow

    A simple conversational agent with no tools, wired into a single-node workflow.
    """)
    return


@app.cell
def _(Agent, LiteLLMChat):
    agent: Agent = Agent(
        name="Assistant",
        system_prompt="You are a helpful assistant. Answer the user's questions clearly and concisely.",
        llm=LiteLLMChat("gpt-4o-mini"),
    )

    print(f"Agent '{agent.name}' ready.")
    return (agent,)


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

    workflow = Workflow()
    workflow.add_node("answer", answer)
    workflow.add_edge(START, "answer")
    workflow.add_edge("answer", END)

    print("Workflow ready.")
    return (workflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · AgentCard

    The `AgentCard` is the agent's public identity document, served at `/.well-known/agent-card.json`. It tells clients what the agent can do and how to reach it.
    """)
    return


@app.cell
def _(AgentCapabilities, AgentCard, AgentInterface, AgentSkill):
    HOST = "127.0.0.1"
    PORT = 9000
    BASE_URL = f"http://{HOST}:{PORT}/"

    card = AgentCard(
        name="Assistant",
        description="A helpful conversational assistant.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=BASE_URL)],
        skills=[
            AgentSkill(
                id="chat",
                name="Chat",
                description="Answer questions and have conversations.",
                tags=["chat", "assistant"],
            )
        ],
    )

    print(f"AgentCard '{card.name}' v{card.version} → {BASE_URL}")
    return BASE_URL, HOST, PORT, card


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Start the A2A Server

    `A2AServer.serve()` calls `uvicorn.run()` internally. We run it in a **daemon thread** so the notebook stays interactive.
    """)
    return


@app.cell
def _(
    A2AServer,
    BASE_URL,
    HOST,
    PORT,
    agent: "Agent",
    card,
    threading,
    time,
    workflow,
):
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

    time.sleep(2)
    print(f"Server running at {BASE_URL}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Send Requests

    | Endpoint | Purpose |
    |---|---|
    | `GET /.well-known/agent-card.json` | Fetch the AgentCard |
    | `POST /` | Send a JSON-RPC message to the agent |

    ### 5a · Fetch the AgentCard
    """)
    return


@app.cell
def _(HOST, PORT, json, mo):
    import subprocess as _subprocess

    AGENT_CARD_URL = f"http://{HOST}:{PORT}/.well-known/agent-card.json"

    _result = _subprocess.run(
        ["curl", "-s", AGENT_CARD_URL], capture_output=True, text=True
    )
    if not _result.stdout:
        mo.output.replace(
            mo.callout(mo.md(f"Empty response: `{_result.stderr}`"), kind="danger")
        )
    else:
        _data = json.loads(_result.stdout)
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        f"**{_data.get('name')}** v{_data.get('version')} — `{AGENT_CARD_URL}`"
                    ),
                    mo.accordion(
                        {"JSON": mo.md(f"```json\n{json.dumps(_data, indent=2)}\n```")}
                    ),
                ]
            )
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b · Send a Message

    The A2A protocol uses **JSON-RPC 2.0** with the `SendMessage` method.

    ```bash
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
            "parts": [{"text": "Hello! What can you help me with?"}]
          }
        }
      }'
    ```
    """)
    return


@app.cell
def _(HOST, PORT, json, uuid):
    import subprocess as _sp

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
        result = _sp.run(
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
    show_raw_response(send_message("Hello! What can you help me with?"))
    return


@app.cell
def _(send_message, show_response):
    show_response(send_message("Tell me a fun fact about space."))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Python Client

    `A2AClient` handles JSON-RPC framing and returns typed `Event` objects.
    """)
    return


@app.cell
async def _(A2AClient, A2AConfig, BASE_URL, stream_output):
    _prompt = "What is the capital of France?"
    _client = A2AClient(config=A2AConfig(endpoint=BASE_URL))
    await stream_output(_client.send_message(_prompt), _prompt)
    return


if __name__ == "__main__":
    app.run()
