# ============================================
# Simple LangGraph Example
# ============================================

from typing import TypedDict

from langgraph.graph import StateGraph, END


# ----------------------------
# Step 1: Define the State
# ----------------------------
class GraphState(TypedDict):
    message: str


# ----------------------------
# Step 2: Node 1
# ----------------------------
def greeting_node(state):
    print("Greeting Node is running...")
    print("Input :", state["message"])

    return {
        "message": "Hello from LangGraph"
    }


# ----------------------------
# Step 3: Node 2
# ----------------------------
def uppercase_node(state):
    print("Uppercase Node is running...")
    print("Input :", state["message"])

    return {
        "message": state["message"].upper()
    }


# ----------------------------
# Step 4: Build Graph
# ----------------------------
graph = StateGraph(GraphState)

graph.add_node("greeting", greeting_node)
graph.add_node("uppercase", uppercase_node)

graph.set_entry_point("greeting")

graph.add_edge("greeting", "uppercase")
graph.add_edge("uppercase", END)

app = graph.compile()


# ----------------------------
# Step 5: Run Graph
# ----------------------------
result = app.invoke({
    "message": "Hi"
})

print("\nFinal Output")
print(result["message"])
