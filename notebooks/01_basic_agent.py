import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import getpass
    import marimo as mo
    import os
    from ant_ai import Agent, InvocationContext, Message, State, Tool, tool
    from ant_ai.llm.integrations import LiteLLMChat
    from pydantic import PrivateAttr
    from utils import stream_output

    return (
        Agent,
        InvocationContext,
        LiteLLMChat,
        Message,
        PrivateAttr,
        State,
        Tool,
        getpass,
        mo,
        os,
        stream_output,
        tool,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 — Basic Agent

    Build a simple **ant-ai** agent with a few tools, then run it against two different inputs and watch the streamed events.

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
    ## 3 · Define Tools

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


if __name__ == "__main__":
    app.run()
