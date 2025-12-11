import os
import streamlit as st
from openai import OpenAI

# ---------- Page config ----------
st.set_page_config(
    page_title="Shabir GPT",
    page_icon="🤖",
    layout="wide",
)

# ---------- Header ----------
st.markdown(
    """
    <div style="text-align:center; padding: 1.2rem 0; border-bottom: 1px solid #3b3b3b;">
        <h1 style="margin-bottom: 0.2rem;">Shabir GPT</h1>
        <p style="margin-top: 0; font-size: 0.95rem; color: #bbbbbb;">
            ChatGPT-style assistant powered by Perplexity Sonar · Built with Streamlit & Python
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🤖 Shabir GPT")

    st.markdown(
        "This app uses your server-side Perplexity API key. "
        "The key is never shown in the UI and should be set as an environment "
        "variable named `PERPLEXITY_API_KEY`."
    )

    # Read API key ONLY from environment (no text input)
    api_key = os.getenv("PERPLEXITY_API_KEY", "")

    model = st.selectbox(
        "Sonar model",
        options=["sonar-pro", "sonar", "sonar-reasoning", "sonar-reasoning-pro"],
        index=0,
        help="`sonar-pro` is a strong default for deep, multi-step Q&A.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Higher = more creative, lower = more focused and deterministic.",
    )

    max_tokens = st.slider(
        "Max tokens in reply",
        min_value=256,
        max_value=4096,
        value=1024,
        step=256,
    )

    system_prompt = st.text_area(
        "System prompt (optional)",
        value=(
            "You are Shabir GPT, a helpful, concise assistant powered by Perplexity's "
            "Sonar models. Answer clearly and use markdown formatting when useful."
        ),
        height=120,
    )

    clear_btn = st.button("🧹 Clear conversation")

    st.markdown("---")
    st.markdown("#### Chat history")

    if "messages" in st.session_state and st.session_state.messages:
        for msg in st.session_state.messages[-10:]:  # last 10
            role_label = "You" if msg["role"] == "user" else "Shabir GPT"
            preview = msg["content"].replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "..."
            st.markdown(f"- **{role_label}:** {preview}")
    else:
        st.caption("No messages yet.")

    st.markdown(
        """
        <small>
        Your API key stays on the server and is never displayed here.
        Configure it as PERPLEXITY_API_KEY in your environment or hosting secrets.
        </small>
        """,
        unsafe_allow_html=True,
    )

# ---------- Session state ----------
if "messages" not in st.session_state or clear_btn:
    st.session_state.messages = []
    # Seed the conversation with a system message (not displayed, only sent to API)
    st.session_state.system_message = {"role": "system", "content": system_prompt}

# If system prompt in sidebar changes mid-session, update it in state
if "system_message" in st.session_state:
    st.session_state.system_message["content"] = system_prompt

# ---------- Helper: Perplexity client ----------
def get_perplexity_client(api_key_str: str) -> OpenAI:
    if not api_key_str:
        raise ValueError("Perplexity API key is missing.")
    client = OpenAI(
        api_key=api_key_str,
        base_url="https://api.perplexity.ai",
    )
    return client

def call_perplexity(
    client: OpenAI,
    model_name: str,
    messages: list,
    temperature_val: float,
    max_tokens_val: int,
):
    # Prepend system message for the API call
    api_messages = [st.session_state.system_message] + messages
    response = client.chat.completions.create(
        model=model_name,
        messages=api_messages,
        temperature=temperature_val,
        max_tokens=max_tokens_val,
    )
    return response.choices[0].message.content

# ---------- Main chat + uploads area ----------
chat_col, upload_col = st.columns([2, 1])

with upload_col:
    st.subheader("Attachments")
    uploaded_files = st.file_uploader(
        "Upload files or images",
        type=["txt", "pdf", "docx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="These will be summarized and included in your question context.",
    )

    if "attachments" not in st.session_state:
        st.session_state.attachments = []

    # Store simple info about files in session (name + size, and text if small txt)
    if uploaded_files:
        st.session_state.attachments = []
        for f in uploaded_files:
            file_info = {
                "name": f.name,
                "type": f.type,
                "size": f.size,
            }
            # For small plain text files, read content directly
            if f.type == "text/plain" and f.size < 200_000:
                file_info["content"] = f.read().decode("utf-8", errors="ignore")
            st.session_state.attachments.append(file_info)

    if st.session_state.attachments:
        st.markdown("**Attached files in this chat:**")
        for a in st.session_state.attachments:
            st.markdown(f"- `{a['name']}` ({a['type']}, {a['size']} bytes)")

with chat_col:
    chat_container = st.container()

    # Display existing conversation
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.info(
                "👋 Welcome to Shabir GPT! Ask any question below.\n\n"
                "You can also upload files or images on the right; they will be summarized or described and used as context."
            )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input at the bottom
    prompt = st.chat_input("Type your question for Shabir GPT...")

    if prompt:
        if not api_key:
            st.error("Please enter your Perplexity API key in the sidebar first.")
        else:
            # Build an extra context string from attachments
            extra_context_parts = []
            for a in st.session_state.get("attachments", []):
                base_desc = f"File: {a['name']} (type: {a['type']}, size: {a['size']} bytes)."
                if "content" in a:
                    # Only small text files are actually inlined
                    base_desc += f" Text content (possibly truncated): {a['content'][:2000]}"
                extra_context_parts.append(base_desc)

            extra_context = "\n\n".join(extra_context_parts)
            if extra_context:
                full_user_message = (
                    prompt
                    + "\n\n[Context from uploaded files/images below]\n"
                    + extra_context
                )
            else:
                full_user_message = prompt

            # Append user message to history
            st.session_state.messages.append(
                {"role": "user", "content": full_user_message}
            )

            # Display user-visible version (without long context noise)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Call Perplexity and display assistant reply
            with st.chat_message("assistant"):
                with st.spinner("Shabir GPT is thinking..."):
                    try:
                        client = get_perplexity_client(api_key)
                        reply = call_perplexity(
                            client=client,
                            model_name=model,
                            messages=st.session_state.messages,
                            temperature_val=temperature,
                            max_tokens_val=max_tokens,
                        )
                        st.markdown(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    except Exception as e:
                        st.error(f"Error while calling Perplexity API: {e}")

# ---------- Footer ----------
st.markdown(
    """
    <hr style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <div style="text-align:center; font-size: 0.8rem; color: #777777; padding-bottom: 0.7rem;">
        Shabir GPT · Educational demo • Not affiliated with Perplexity AI.<br/>
        Built with ❤️ using Streamlit, Python, and Perplexity Sonar.
    </div>
    """,
    unsafe_allow_html=True,
)
