import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import getpass
    import marimo as mo
    import os
    import subprocess
    from pathlib import Path
    from ant_ai import Agent, InvocationContext, Message, State, tool
    from ant_ai.llm.integrations import LiteLLMChat
    from utils import stream_output

    return (
        Agent,
        InvocationContext,
        LiteLLMChat,
        Message,
        Path,
        State,
        getpass,
        mo,
        os,
        stream_output,
        subprocess,
        tool,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 — Agent with Skills

    **Skills** are Markdown documents that give an agent domain-specific instructions at runtime, without touching the agent's code. They follow the [agentskills.io](https://agentskills.io) specification.

    In this notebook we:
    1. Create a local skill directory with a `SKILL.md` file
    2. Add the skill directory to the Calculator agent from notebook 02
    3. Start a new A2A server on a different port and verify the skill is active
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Install & API Key
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
    ## 2 · How Skills Work

    A skill is a folder inside a **skills directory** (e.g. `.agents/skills/`). Each skill contains:

    ```
    .agents/skills/
    └── my-skill/          ← folder name must match the `name:` frontmatter
        ├── SKILL.md       ← instructions the agent reads
        └── scripts/       ← optional helper scripts (referenced by path)
            └── helper.py
    ```

    `SKILL.md` uses YAML frontmatter so ant-ai can index it:

    ```markdown
    ---
    name: my-skill
    description: One-line description — used by the agent to decide relevance.
    compatibility: Python 3.10+
    license: MIT
    metadata:
      author: you
      version: "1.0"
    ---

    # My Skill

    Step-by-step instructions the agent follows when this skill is relevant...
    ```

    When the agent is invoked, ant-ai scans the skills directory and activates any skills whose `description` is relevant to the user's query.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The `find-skills` Skill

    This skill was installed from the Vercel Labs registry:

    ```bash
    npx skills add https://github.com/vercel-labs/skills --skill find-skills
    ```

    It lands in `.agents/skills/find-skills/`:

    ```
    .agents/skills/find-skills/
    ├── SKILL.md
    └── scripts/
        └── search.py        ← queries the skills registry
    ```

    The skill description matches queries like "find a skill", "what skills exist for X", or "discover skills" — so ant-ai activates it without any hint in the system prompt. The agent uses the search script to query the registry and returns matching skill names and descriptions.
    """)
    return


@app.cell
def _(Path):
    SKILL_DIR = Path("../.agents/skills")
    print(f"Skills directory: {SKILL_DIR.resolve()}")
    return (SKILL_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Agent with Skills

    Pass the skills directory path to the `skills` parameter. ant-ai scans the folder at startup and injects each skill's name and description into the system prompt, plus a pointer: `Read .agents/skills/find-skills/SKILL.md for full instructions`.

    The agent needs to be able to both **read that file** and **execute the commands it prescribes** — a single `bash` tool covers both (`cat` for reading, any command for executing).
    """)
    return


@app.cell
def _(Agent, LiteLLMChat, SKILL_DIR, subprocess, tool):
    @tool
    def run_command(command: str) -> str:
        """Run a shell command and return its output."""
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout or result.stderr

    agent = Agent(
        name="Assistant",
        system_prompt="You are a helpful assistant.",
        llm=LiteLLMChat("gpt-5-nano"),
        tools=[run_command],
        skills=[".agents/skills"],
    )

    print(f"Agent '{agent.name}' created with skills from: {SKILL_DIR}")
    return (agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Quick Test (no server)

    Verify the skill is picked up by running the agent directly.
    """)
    return


@app.cell
async def _(InvocationContext, Message, State, agent, stream_output):
    async def run(prompt: str, session_id: str = "skills-demo-233") -> None:
        ctx = InvocationContext(session_id=session_id)
        state = State()
        state.add_message(Message(role="user", content=prompt))
        await stream_output(agent.stream(state, ctx=ctx), prompt)

    await run("Is there a skill for React development?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Adding More Skills

    To add another skill, create a new folder under `.agents/skills/` with its own `SKILL.md` and restart the server:

    ```
    .agents/skills/
    ├── find-skills/           ← installed via npx skills add
    │   ├── SKILL.md
    │   └── scripts/
    │       └── search.py
    └── my-new-skill/          ← your own skill
        ├── SKILL.md
        └── scripts/
            └── helper.py
    ```

    Install community skills with:

    ```bash
    npx skills add https://github.com/vercel-labs/skills --skill find-skills
    npx skills add <owner>/<skill-name>
    ```

    Skills are placed in `.agents/skills/` automatically and picked up on the next agent restart.
    """)
    return


if __name__ == "__main__":
    app.run()
