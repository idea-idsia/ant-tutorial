import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import getpass
    import marimo as mo
    import os
    from collections.abc import AsyncGenerator
    from typing import Literal
    from ant_ai import Agent, BaseAgent, InvocationContext, Message, State, Tool, tool
    from ant_ai.llm.integrations import LiteLLMChat
    from ant_ai.workflow import END, START, NodeYield, Workflow, build_workflow_graph
    from pydantic import PrivateAttr
    from utils import stream_output

    return (
        Agent,
        AsyncGenerator,
        BaseAgent,
        END,
        InvocationContext,
        LiteLLMChat,
        Literal,
        Message,
        NodeYield,
        PrivateAttr,
        START,
        State,
        Tool,
        Workflow,
        build_workflow_graph,
        getpass,
        mo,
        os,
        stream_output,
        tool,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 — Agent & Workflow

    Build an **ant-ai** agent with tools, run it against several inputs, then wire it into a `Workflow` — including conditional routing.

    > **Note:** This notebook uses top-level `await`, which is supported by default in Jupyter/IPython kernels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Install
    """)
    return


@app.cell
def _():
    # '%pip install ant-ai --quiet' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · API Key

    ant-ai routes LLM calls through [LiteLLM](https://docs.litellm.ai/). Provide an OpenAI key (or any LiteLLM-compatible key).
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
    ## 3 · Simplest Agent

    Before adding tools, here's the minimum to create an agent and get a response.
    """)
    return


@app.cell
async def _(Agent, InvocationContext, LiteLLMChat, Message, State):
    simple_agent = Agent(
        name="SimpleAgent",
        system_prompt="You are a helpful assistant.",
        llm=LiteLLMChat("gpt-4o-mini"),
    )

    ctx = InvocationContext(session_id="demo")
    state = State()
    state.add_message(Message(role="user", content="What is 2 + 2?"))

    async for event in simple_agent.stream(state, ctx=ctx):
            print(event.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Define Tools

    Tools are plain Python functions decorated with `@tool`. The **docstring** is what the LLM reads to decide when and how to call each tool.
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
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert a temperature from Celsius to Fahrenheit."""
        return celsius * 9 / 5 + 32

    print("Tools ready:", add.name, multiply.name, celsius_to_fahrenheit.name)
    return add, celsius_to_fahrenheit, multiply


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Class-Based Stateful Tool

    Tools can also be defined as **subclasses of `Tool`**. Each public method becomes a separate tool named `ClassName_method`, and the instance carries state across calls within a session.

    `Tape` models an **adding-machine tape**: the agent records labeled amounts one at a time, and the running total accumulates in `_entries` — a Pydantic `PrivateAttr` that is per-instance, invisible to the LLM schema, and lives for the lifetime of the object.
    """)
    return


@app.cell
def _(PrivateAttr, Tool):
    class Tape(Tool):
        """An adding-machine tape: record labeled amounts and track a running total."""

        _entries: list[tuple[str, float]] = PrivateAttr(default_factory=list)

        def record(self, label: str, amount: float) -> str:
            """Record a labeled amount (negative for deductions) and return the updated running total."""
            self._entries.append((label, amount))
            total = sum(v for _, v in self._entries)
            return f"Recorded '{label}' {amount:+g}  →  running total = {total:g}"

        def running_total(self) -> float:
            """Return the current running total of all recorded amounts."""
            return sum(v for _, v in self._entries)

        def show_tape(self) -> str:
            """Show every recorded entry and the final total."""
            if not self._entries:
                return "Tape is empty."
            lines = [f"  {label}: {amount:+g}" for label, amount in self._entries]
            lines.append(f"  TOTAL: {sum(v for _, v in self._entries):g}")
            return "\n".join(lines)

        def clear(self) -> str:
            """Erase all entries and reset the running total to zero."""
            self._entries.clear()
            return "Tape cleared."

    tape = Tape()
    print(
        "Tape namespace tools:",
        [f"{tape.name}_{m}" for m in Tape.__namespace_methods__],
    )
    return (tape,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Create the Agent

    `Agent` is the main class. Give it:
    - `name` – used in logs and the A2A card (later notebooks)
    - `system_prompt` – the agent's persona and instructions
    - `llm` – the language model backend
    - `tools` – the callables it may invoke
    """)
    return


@app.cell
def _(Agent, LiteLLMChat, add, celsius_to_fahrenheit, multiply, tape):
    agent = Agent(
        name="Calculator",
        system_prompt=(
            "You are a helpful calculator assistant. "
            "Use the provided tools to answer maths and unit-conversion questions. "
            "Use the Tape tools to record amounts and track running totals when asked."
        ),
        llm=LiteLLMChat("gpt-4o-mini"),
        tools=[add, multiply, celsius_to_fahrenheit, tape],
    )

    print(f"Agent '{agent.name}' created with {len(agent.tools)} tool(s).")
    return (agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Run the Agent

    `agent.stream()` is an **async generator** that yields events as the agent thinks, calls tools, and produces its final answer.

    | `event.kind` | Meaning |
    |---|---|
    | `update` | Intermediate reasoning text |
    | `tool_calling` | About to invoke a tool |
    | `tool_result` | Result returned from a tool |
    | `final_answer` | The agent's complete response |
    """)
    return


@app.cell
def _(InvocationContext, Message, State, agent, stream_output):
    async def run(prompt: str, session_id: str = "demo") -> None:
        ctx = InvocationContext(session_id=session_id)
        state = State()
        state.add_message(Message(role="user", content=prompt))
        await stream_output(agent.stream(state, ctx=ctx), prompt)

    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Input 1 — simple arithmetic
    """)
    return


@app.cell
async def _(run):
    await run("What is 3 + 4?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Input 2 — chained operations
    """)
    return


@app.cell
async def _(run):
    await run(
        "What is 15 multiplied by 7? Also, what is 100 °C in Fahrenheit?",
        session_id="demo-2",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Input 3 — stateful tape across calls

    The agent uses the `Tape` to accumulate a running total across multiple tool calls. Each `record` call appends to `_entries`; the total is recalculated from the full list every time.
    """)
    return


@app.cell
async def _(run):
    await run(
        "I bought 3 items at $12.50 each. I then got a $5.00 discount. "
        "Finally there is a $8.99 shipping fee. "
        "Record each amount on the tape and tell me the final total.",
        session_id="tape-demo",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Workflow

    A `Workflow` defines the graph of steps the agent executes per request. Here we use the simplest possible workflow: a single node that streams the agent's response and then returns the updated state.

    The node function signature `(agent, state, ctx)` is the convention ant-ai uses to inject dependencies.
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
    ## 6b · Conditional Edges

    A **conditional edge** routes the workflow to a different next node based on runtime state. You provide a router function that returns a `Literal` string matching one of the target node names — ant-ai reads the return annotation to discover the possible branches.

    In this example the router inspects the incoming question: conversion questions (temperature, units) go to `"convert"`, everything else goes to `"calculate"`. Both nodes stream the same agent; the branching makes the intent explicit in the graph.
    """)
    return


@app.cell
def _(
    AsyncGenerator,
    BaseAgent,
    END,
    InvocationContext,
    Literal,
    NodeYield,
    START,
    State,
    Workflow,
):
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
    cond_workflow.add_conditional_edge(START, route_question)
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
    ## 6c · Run the Conditional Workflow

    Two inputs — one routes to `calculate`, the other to `convert` — so you can see the branching in action.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Routes to `calculate`
    """)
    return


@app.cell
async def _(
    InvocationContext,
    Message,
    State,
    agent,
    cond_workflow,
    stream_output,
):
    _prompt = "What is 6 multiplied by 9?"
    _ctx = InvocationContext(session_id="cond-calc")
    _state = State()
    _state.add_message(Message(role="user", content=_prompt))
    await stream_output(cond_workflow.stream(agent, ctx=_ctx, state=_state), _prompt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Routes to `convert`
    """)
    return


@app.cell
async def _(
    InvocationContext,
    Message,
    State,
    agent,
    cond_workflow,
    stream_output,
):
    _prompt = "What is 37 °C in Fahrenheit?"
    _ctx = InvocationContext(session_id="cond-conv")
    _state = State()
    _state.add_message(Message(role="user", content=_prompt))
    await stream_output(cond_workflow.stream(agent, ctx=_ctx, state=_state), _prompt)
    return


if __name__ == "__main__":
    app.run()
