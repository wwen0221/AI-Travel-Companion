# # ''' 
# # chatbot should be able to:
# # 	1. Travel Itinerary Planning
# #     2. Flight and Accommodation Booking
# #     3. Local Attraction Recommender
# #     4. Language Translation - Done
# #     5. Expense Tracking
# #     6. Weather Updates
# #     7. Travel Regulations Checker
# #     8. Natural Language Query Handling
# # '''


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import requests
from langchain_core.messages import ToolMessage

load_dotenv()

gemini_api_key = os.getenv("gemini-api")
tavily_api_key = os.getenv("tavily-api")
AMADEUS_API_KEY = os.getenv("amadeus-api")
AMADEUS_API_SECRET = os.getenv("amadeus-api-secret")


os.environ["GOOGLE_API_KEY"] = gemini_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
model  = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
ai_translator  = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

show_graph = False
interact_w_user = True


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
#chatbot with normal llm (w/o tools)
# def chatbot(state: State):
#     return {"messages": [model.invoke(state["messages"])]}

#chatbot with llm tools
def chatbot(state: State):
    prompt = f"""
    You are a helpful assistant. You have access to the following tools:
    1. translate_text: Translates text into a specified language.
    2. TavilySearchResults: Searches the web for information.
    3. search_hotels: Search hotels based on user's query.

    If the user asks for a translation, use the translate_text tool.
    If the user asks for hotel informations, use the search_hotels tool.
    Otherwise, respond directly or use other tools as needed.

    User's query: {state["messages"]}
    """
 
    return {"messages": [llm_with_tools.invoke(prompt)]}

def display_graph(graph):
    try:
        image_data =graph.get_graph().draw_mermaid_png()
        with open('graph.png', 'wb') as f:
            f.write(image_data)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def create_graph():

    graph_builder = StateGraph(State)
    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)
    # graph_builder.add_edge(START, "chatbot")
    # graph_builder.add_edge("chatbot", END)

    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()

    return graph

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def get_amadeus_access_token():
    """
    Retrieves an access token from the Amadeus API.
    """
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET,
    }
    
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get access token: {response.text}")


@tool
def translator(query:str, target_language: str)-> str:
    """
    Translates the given text.
    Args:
        text (str): User query.
    Returns:
        str: The translated text.
    """
    print('translator called...')
    
    try:
        translation_prompt = f"""
        Translate the following text into {target_language}:

        "{query}"
        """

        translated_response = ai_translator.invoke(translation_prompt)
        
        return translated_response.content

    except Exception as e:
        return f"Translation failed: {str(e)}"

@tool
def search_hotels(city_code: str, radius: int = 5, radius_unit: str = "KM", amenities: str = None, ratings: int = 5) -> str:
    """
    Searches for hotels in a given city using the Amadeus API.
    Args:
        city_code (str): The IATA city code (e.g., 'PAR' for Paris).
        radius (int): Search radius in kilometers (default is 5).
        radius_unit (str): Unit of measurement (default is 'KM').
        amenities (str): Comma-separated list of amenities (e.g., 'SWIMMING_POOL,WIFI').
        ratings (int): Minimum rating (default is 5).

    Returns:
        str: JSON response with hotel details.
    """
    try:
        token = get_amadeus_access_token()
        url = f"https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
        params = {
            "cityCode": city_code,
            "radius": radius,
            "radiusUnit": radius_unit,
            "hotelSource": "ALL",
            "ratings":ratings
        }
        if amenities:
            params["amenities"] = amenities

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            resp = response.json()
            for data in resp['data']:
                hotel_name = data['name'].format(data['name'])
                dist =f'{data["distance"]["value"]}{data["distance"]["unit"]}'.format(data['distance'])
                print(hotel_name, dist)
        else:
            return f"Error: {response.shtatus_code}, {response.text}"

    except Exception as e:
        return f"Hotel search failed: {str(e)}"
    
tavily_tool = TavilySearchResults(max_results=2)

tools = [tavily_tool, translator, search_hotels]

graph = create_graph()


if show_graph:
    #graph will be saved as graph.png
    display_graph(graph)

llm_with_tools = model.bind_tools(tools)

if interact_w_user:
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except Exception as e:
            print(e)
            break



            