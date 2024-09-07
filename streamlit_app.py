import warnings
import transformers
import os
import uuid
import csv
from operator import itemgetter
import streamlit as st
import re

# Suppress the warning immediately after importing warnings
warnings.filterwarnings("ignore", module="transformers") 

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

# --- Language Model ---

model = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.7,
)

# --- Conversation History Management ---

store = {}  # In-memory storage for conversation history


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a session history for a given session ID."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# --- Prompt Template ---

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly and helpful customer service assistant representing 1000Bytes Innovations. 
            Your primary goal is to assist customers with their order status inquiries. 
            You can access and provide information about order details, shipping status, delivery estimates, 
            and any potential issues related to their orders. 
            Always be polite, professional, and informative in your responses.
            
            If a customer provides an order ID, use it to look up the order information (you can pretend to do this).
            If you don't have enough information to answer a question, politely ask for clarification or more details, 
            such as the order ID or the customer's email address associated with the order.

            Answer all questions clearly and politely in {language}.""", 
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- History Trimmer ---

trimmer = trim_messages(
    max_tokens=1000,  # Adjust as needed based on your LLM's context window
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# --- Chain Construction ---

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# --- Wrap with Message History ---

with_message_history = RunnableWithMessageHistory(  # Make sure this is defined before start_chat()
    chain,
    get_session_history,
    input_messages_key="messages",
)

# --- Order Lookup Functions ---

def lookup_restaurant_order(order_id):
    """Looks up an order in the restaurants_orders.csv file."""
    with open('restaurants_orders.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['order_id'] == order_id:
                return row  # Return the row if order ID is found
    return None  # Return None if order ID is not found


def lookup_delivery_order(order_id):
    """Looks up an order in the delivery_orders.csv file."""
    with open('delivery_orders.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['order_id'] == order_id:
                return row
    return None


# --- Streamlit App ---

st.title("1000Bytes Innovations - Order Status Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())  # Generate session_id once

    # Add the initial message from the bot
    initial_message = "Hi there! Welcome to 1000Bytes Innovations. How can I help you with your order today?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

     # --- Order ID Extraction (Using Regular Expressions) ---
    if "order id" in prompt.lower() or "order number" in prompt.lower():  # Handle both "order id" and "order number"
        match = re.search(r"(\d+)", prompt)  # Search for one or more digits in the prompt
        if match:
            order_id = match.group(1)  # Extract the matched digits as the order ID

            # --- Restaurant Order Lookup ---
            restaurant_order = lookup_restaurant_order(order_id)
            print(f"restaurant_order: {restaurant_order}")  # Debugging: Print the result of the lookup

            if restaurant_order:
                if restaurant_order['status'] == 'handed over':
                    # --- Delivery Order Lookup ---
                    delivery_order = lookup_delivery_order(order_id)
                    if delivery_order:
                        response_content = f"Your order (order ID: {order_id}) is currently out for delivery with our partner {delivery_order['partner']}. You can expect it to arrive soon."  # Updated response
                    else:
                        response_content = f"Your order (order ID: {order_id}) has been handed over for delivery. I'm checking with our delivery partner for updates."
                else:
                    response_content = f"Your order (order ID: {order_id}) is currently being prepared."
            else:
                response_content = "I'm sorry, I couldn't find an order with that ID. Please double-check the order ID and try again."

    else:
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=prompt)], "language": "English"},
            config={"configurable": {"session_id": st.session_state.session_id}}  # Use the session_id from session state
        )
        response_content = response.content

    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Display bot message in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_content)