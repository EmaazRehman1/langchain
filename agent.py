from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import Tool

from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")  
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="mistralai/mistral-7b-instruct:free"
)

search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)

def search_wrapper(query: str) -> str:
    """Wrapper function for Tavily search"""
    return search.run(query)

search_tool = Tool(
    name="tavily_search",
    func=search_wrapper,
    description="Search the web for current or past information."
)

tools = [search_tool]

prompt = PromptTemplate.from_template("""
You are a helpful assistant called Max. Answer the following questions as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""").partial(
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools])
)

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  
)

try:
    response = agentExecutor.invoke({
        "input": "What is the temperature in Karachi today?"
    })
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
