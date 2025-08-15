import streamlit as st
from auth import init_authentication
from ui_components import initialize_rag_app, render_sidebar, render_chat_interface


def main():
    # Page configuration
    st.set_page_config(
        page_title="Cogent Labs Policies Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_app" not in st.session_state:
        st.session_state.rag_app = None

    # Initialize authentication
    init_authentication()

    # Title and description
    st.title("Cogent Labs Policies Assistant")
    st.markdown("Ask questions about the Cogent Labs policies.")

    # Sidebar for document management with authentication
    with st.sidebar:
        render_sidebar()
        
    # Initialize RAG app if not already done
    if st.session_state.rag_app is None:
        with st.spinner("Loading..."):
            st.session_state.rag_app = initialize_rag_app()
            # st.session_state.rag_app.load_documents("food.docx")

    # Render the main chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
