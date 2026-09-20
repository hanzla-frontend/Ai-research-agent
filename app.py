import os
import streamlit as st

from agent import build_crew


def get_groq_api_key() -> str:
    """Reads the Groq API key from Streamlit Cloud secrets."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("CrewAI (single agent) + Groq `openai/gpt-oss-120b` + free DuckDuckGo search")

groq_api_key = get_groq_api_key()

if not groq_api_key:
    st.error(
        "No Groq API key found in Streamlit secrets.\n\n"
        "Go to your app on Streamlit Cloud → **Settings → Secrets**, and add:\n\n"
        '```\nGROQ_API_KEY = "your_actual_key_here"\n```'
    )
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "**How it works**\n\n"
        "1. Enter a topic\n"
        "2. The agent searches DuckDuckGo for it\n"
        "3. It writes you a Markdown report"
    )
    st.markdown("---")
    st.success("Groq API key loaded from Streamlit secrets ✅")

topic = st.text_input(
    "What topic do you want researched?",
    placeholder="e.g. Impact of AI agents on customer support jobs",
)

run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)

if run_button:
    if not topic.strip():
        st.error("Please enter a topic to research.")
    else:
        with st.spinner("Researching... this can take a minute or two ⏳"):
            try:
                crew = build_crew(topic.strip())
                result = crew.kickoff()
                report = result.raw if hasattr(result, "raw") else str(result)

                st.success("Done! Here's your report:")
                st.markdown(report)

                st.download_button(
                    label="⬇️ Download report (.md)",
                    data=report,
                    file_name=f"{topic.strip().replace(' ', '_')}_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
