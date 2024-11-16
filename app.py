import streamlit as st
from src.inference_API import HuggingFaceSummarizer


def main():
    st.title("Text Summarization App")
    st.subheader("Powered by Hugging Face API")

    # Input for API token
    api_token = st.text_input("Enter your Hugging Face API token", type="password")

    # Ensure the token is provided
    if not api_token:
        st.warning("Please provide an API token to continue.")
        return

    # Input for long text
    text_input = st.text_area("Enter the text to summarize", height=200)

    if st.button("Summarize"):
        if not text_input.strip():
            st.error("Please provide a valid text input.")
        else:
            # Initialize summarizer
            summarizer = HuggingFaceSummarizer(api_token)

            # Generate summary
            with st.spinner("Generating summary..."):
                result = summarizer.generate_summary(text_input)

            # Handle and display result
            if "error" in result:
                st.error(
                    f"Error: {result['error']} (Status Code: {result.get('status_code', 'N/A')})"
                )
            else:
                summary = result[0].get("generated_text", "No summary available.")
                st.success("Summary generated!")
                st.text_area("Summary", summary, height=100)


if __name__ == "__main__":
    main()
