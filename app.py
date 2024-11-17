import streamlit as st
from src.inference_API import HuggingFaceSummarizer


def main():
    """
    Main function to run the Streamlit-based Text Summarization App.

    The app allows users to input a Hugging Face API token and a long piece of text
    to generate a summarized version using a Hugging Face model hosted on the API.
    """
    # Set the title and subheader of the Streamlit app
    st.title("Text Summarization App")
    st.subheader("Developed by BEASTBOYJAY")

    # Input field for the Hugging Face API token
    api_token = st.text_input("Enter your Hugging Face API token", type="password")

    # Ensure the user provides an API token
    if not api_token:
        st.warning("Please provide an API token to continue.")
        return

    # Input field for the text to be summarized
    text_input = st.text_area("Enter the text to summarize", height=200)

    # Button to trigger the summarization process
    if st.button("Summarize"):
        # Validate the input text
        if not text_input.strip():
            st.error("Please provide a valid text input.")
        else:
            # Initialize the Hugging Face Summarizer with the provided API token
            summarizer = HuggingFaceSummarizer(api_token)

            # Generate the summary with a loading spinner
            with st.spinner("Generating summary..."):
                result = summarizer.generate_summary(text_input)

            # Handle and display the result
            if "error" in result:
                # Display an error message if the summarization fails
                st.error(
                    f"Error: {result['error']} (Status Code: {result.get('status_code', 'N/A')})"
                )
            else:
                # Extract and display the generated summary
                summary = result[0].get("generated_text", "No summary available.")
                st.success("Summary generated!")
                st.text_area("Summary", summary, height=100)


if __name__ == "__main__":
    main()
