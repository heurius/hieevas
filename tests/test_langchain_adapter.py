"""The callback adapter inside a real LangGraph graph (fake chat model, real ToolNode)."""
import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from hieevas import Recorder, evaluate
from hieevas.adapters.langchain import HieevasCallbackHandler


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return "Paris is the capital of France."


def build_graph():
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="I need a source for the capital.",
                  tool_calls=[{"name": "lookup", "args": {"query": "capital of France"}, "id": "c1"}],
                  usage_metadata={"input_tokens": 50, "output_tokens": 12, "total_tokens": 62}),
        AIMessage(content="Paris", usage_metadata={"input_tokens": 80, "output_tokens": 2, "total_tokens": 82}),
    ]))

    def agent(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}

    def route(state: MessagesState):
        return "tools" if state["messages"][-1].tool_calls else END

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([lookup]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile()


def test_handler_records_llm_and_tool_calls():
    graph = build_graph()
    rec = Recorder(app="lg", architecture="A1")
    with rec.run("capital", reference="Paris") as run:
        out = graph.invoke({"messages": [HumanMessage("Capital of France?")]},
                           config={"callbacks": [HieevasCallbackHandler(run)]})
        run.answer(out["messages"][-1].content)
    r = rec.runs[0]
    llm = r.steps_of("llm")
    tools = r.steps_of("tool")
    assert len(llm) == 2 and llm[0].agent == "agent"
    assert llm[0].input_tokens == 50 and llm[1].output_tokens == 2
    assert len(tools) == 1 and tools[0].tool == "lookup" and tools[0].status == "ok"
    assert tools[0].rationale == "I need a source for the capital."
    assert r.correct is True
    report = evaluate(rec.runs, group_by=())
    assert report.value("M5").value == 62 + 82
    assert report.value("M23").value == 1.0
