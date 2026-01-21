import streamlit as st
from langchain_core.messages import ChatMessage

st.title("Dasom")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role,message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(user_input)

    add_message("user",user_input)
    add_message("assistant",user_input)