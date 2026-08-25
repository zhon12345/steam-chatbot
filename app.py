import streamlit as st

from engine import generate_response, load_artifacts

st.set_page_config(
    page_title="SteamBot: Steam Game Discovery Assistant",
    page_icon="🎮",
    layout="centered",
)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()

st.title("🎮 SteamBot: Steam Game Discovery Assistant")
st.caption("Ask for recommendations, game info, pricing, or system support.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm SteamBot, your assistant for discovering Steam games. Try asking things like *'Recommend me Linux co-op games under $20'* or *'Is Terraria free?'*.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask SteamBot"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching steam catalog..."):
        response = generate_response(prompt, artifacts)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
