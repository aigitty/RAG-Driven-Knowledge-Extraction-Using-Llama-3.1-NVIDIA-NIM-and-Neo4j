import os
import sys

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

# Set up environment variables
NEO4J_URI = "bolt://35.175.233.136"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "twirl-disassemblies-law"


os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

# Initialize the Neo4j graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Initialize the LLM (NVIDIA endpoint)
os.environ["NVIDIA_API_KEY"] = "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"
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# Neo4j Fulltext Index setup
graph.query("CREATE FULLTEXT INDEX drug IF NOT EXISTS FOR (d:Drug) ON EACH [d.name];")
graph.query("CREATE FULLTEXT INDEX manufacturer IF NOT EXISTS FOR (d:Manufacturer) ON EACH [d.manufacturerName];")

# Helper functions
def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    return full_text_query.strip()

candidate_query = """
CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
YIELD node
RETURN coalesce(node.manufacturerName, node.name) AS candidate, labels(node)[0] AS label
"""

def get_candidates(input: str, type: str, limit: int = 3) -> List[Dict[str, str]]:
    ft_query = generate_full_text_query(input)
    candidates = graph.query(candidate_query, {"fulltextQuery": ft_query, "index": type, "limit": limit})
    return candidates

@tool
def get_side_effects(
    drug: Optional[str] = Field(description="Disease mentioned in the question. Return None if not mentioned."),
    min_age: Optional[int] = Field(description="Minimum age of the patient. Return None if not mentioned."),
    max_age: Optional[int] = Field(description="Maximum age of the patient. Return None if not mentioned."),
    manufacturer: Optional[str] = Field(description="Manufacturer of the drug. Return None if not mentioned."),
):
    """
    Retrieve side effects associated with a drug, considering optional filters like age and manufacturer.
    """
    params = {}
    filters = []
    side_effects_base_query = """
    MATCH (c:Case)-[:HAS_REACTION]->(r:Reaction), (c)-[:IS_PRIMARY_SUSPECT]->(d:Drug)
    """
    if drug and isinstance(drug, str):
        candidate_drugs = [el["candidate"] for el in get_candidates(drug, "drug")]
        if not candidate_drugs:
            return "The mentioned drug was not found"
        filters.append("d.name IN $drugs")
        params["drugs"] = candidate_drugs

    if min_age and isinstance(min_age, int):
        filters.append("c.age > $min_age ")
        params["min_age"] = min_age
    if max_age and isinstance(max_age, int):
        filters.append("c.age < $max_age ")
        params["max_age"] = max_age
    if manufacturer and isinstance(manufacturer, str):
        candidate_manufacturers = [el["candidate"] for el in get_candidates(manufacturer, "manufacturer")]
        if not candidate_manufacturers:
            return "The mentioned manufacturer was not found"
        filters.append(
            "EXISTS {(c)<-[:REGISTERED]-(:Manufacturer {manufacturerName: $manufacturer})}"
        )
        params["manufacturer"] = candidate_manufacturers[0]

    if filters:
        side_effects_base_query += " WHERE " + " AND ".join(filters)
    side_effects_base_query += """
    RETURN d.name AS drug, r.description AS side_effect, count(*) AS count
    ORDER BY count DESC
    LIMIT 10
    """
    data = graph.query(side_effects_base_query, params=params)
    return {"output": data}  # Ensure it returns data with "output" key



# Streamlit app
st.set_page_config(page_title="Drug Reporting System", page_icon="💊", layout="wide")

# Custom CSS to enhance the app
st.markdown(
    """
    <style>
    body {
        background-color: #2c3e50; /* Dark background */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background-color: #34495e; /* Darker background for the app area */
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
        max-width: 800px;
        margin: auto;
    }
    h1 {
        color: #ecf0f1; /* Light text color for headings */
        text-align: center;
        margin-bottom: 20px;
    }
    .stTextInput, .stNumberInput {
        margin-bottom: 15px;
        background-color: #1e272e; /* Dark background for inputs */
        color: #ecf0f1; /* Light text color for inputs */
        border: 1px solid #7f8c8d; /* Border color */
        border-radius: 5px;
        padding: 10px;
    }
    .stButton {
        margin-top: 10px;
        background-color: #27ae60; /* Button color */
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .stButton:hover {
        background-color: #2ecc71; /* Button hover color */
    }
    .output-box {
        background-color: #1e272e; /* Dark background for output */
        color: #ecf0f1; /* Light text color for output */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #7f8c8d; /* Border color */
        margin-top: 20px;
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that finds information about common side effects. "
                   "If tools require follow-up questions, make sure to ask the user for clarification."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

tools = [get_side_effects]
llm_with_tools = llm.bind_tools(tools=tools)

agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"])
        if x.get("chat_history")
        else [],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


# Add typing for input
class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


class Output(BaseModel):
    output: Any


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput, output_type=Output
)



st.title("💊 Drug Reporting System")
st.write("Welcome! This application helps you find common side effects for drugs based on patient demographics.")
st.write("Please provide the following details:")

# User inputs
input_drug = st.text_input("Enter the name of the drug:", placeholder="e.g., Lyrica")
min_age = st.number_input("Enter minimum age (if applicable):", min_value=0, max_value=120, value=0)
max_age = st.number_input("Enter maximum age (if applicable):", min_value=0, max_value=120, value=35)
input_manufacturer = st.text_input("Enter the manufacturer (if applicable):", placeholder="e.g., Pfizer")

# Button to find side effects
if st.button("Find Side Effects"):
    with st.spinner("Processing..."):
        # Display the parameters being used
        st.write("Searching for side effects with the following parameters:")
        st.write(f"- Drug: {input_drug if input_drug else 'Not specified'}")
        st.write(f"- Minimum Age: {min_age}")
        st.write(f"- Maximum Age: {max_age}")
        st.write(f"- Manufacturer: {input_manufacturer if input_manufacturer else 'Not specified'}")

        # Get user input
        user_input = AgentInput(input=f"Find side effects for {input_drug}, min_age: {min_age}, max_age: {max_age}, manufacturer: {input_manufacturer}", chat_history=[])

        # Execute the agent
        output = agent_executor.invoke({"input": user_input})


        if output.get('output'):
            side_effects = output['output']  # Get the output
            
            if isinstance(side_effects, list):
                side_effects_list = [effect['side_effect'] for effect in side_effects if isinstance(effect, dict)]
                summary = f"Based on the information provided, the most common side effects of {input_drug} are:"
                st.markdown(f"<div class='output-box'>{summary}<ul>" +
                    "".join([f"<li>{side_effect}</li>" for side_effect in side_effects_list]) +
                    "</ul></div>", unsafe_allow_html=True)
            else:
                # Handle the case where side_effects is not a list (e.g., it's an error message)
                st.markdown(f"<div class='output-box'>{side_effects}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='output-box'>No side effects found.</div>", unsafe_allow_html=True)
