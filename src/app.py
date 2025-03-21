import streamlit as st
from inference import generate_response
from web_scraping import scrape_proptech_data
import json
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load keywords from a JSON file
def load_keywords():
    """
    Load keywords from a JSON file.
    """
    try:
        file_path = Path("keywords.json")
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                return data.get("keywords", [])
        else:
            return []  # Return an empty list if the file doesn't exist
    except Exception as e:
        logger.error(f"Error loading keywords: {e}")
        return []

# Update keywords in the JSON file
def update_keywords(new_keywords):
    """
    Update keywords in the JSON file.
    """
    try:
        with open("keywords.json", "w") as f:
            json.dump({"keywords": new_keywords}, f)
        logger.info("Keywords updated successfully.")
    except Exception as e:
        logger.error(f"Error updating keywords: {e}")

# Function to display a temporary inline message
def show_temp_message(message, message_type="success", duration=3):
    """
    Display a temporary inline message that disappears after a few seconds.
    """
    if message_type == "success":
        message_placeholder = st.success(message)
    elif message_type == "warning":
        message_placeholder = st.warning(message)
    elif message_type == "error":
        message_placeholder = st.error(message)
    
    # Remove the message after a delay
    time.sleep(duration)
    message_placeholder.empty()

def main():
    # Page configuration
    st.set_page_config(page_title="PropTech Insights", page_icon="🏠", layout="wide")

    # Title and description
    st.title("PropTech Smart Home Insights")

    # Sidebar for navigation
    st.sidebar.header("Menu")
    app_mode = st.sidebar.selectbox("Choose a feature", ["Q&A Assistant", "Web Research", "Manage Keywords", "About"])

    if app_mode == "Q&A Assistant":
        st.header("PropTech Q&A Assistant")
        query = st.text_input("Ask a question about PropTech and Smart Homes:")
        if query:
            with st.spinner("Generating response..."):
                try:
                    response = generate_response(query)
                    st.success("AI Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    elif app_mode == "Web Research":
        st.header("PropTech Web Research")
        research_query = st.text_input("Enter a PropTech research topic:")
        if research_query:
            with st.spinner("Searching web..."):
                try:
                    results = scrape_proptech_data(research_query)
                    st.subheader("Research Findings")
                    for result in results:
                        st.markdown(f"### {result['title']}")
                        st.write(f"**Link:** {result['link']}")
                        st.write(f"**Description:** {result['description']}")
                        st.markdown("---")
                except Exception as e:
                    st.error(f"Error in web research: {e}")

    elif app_mode == "Manage Keywords":
        st.header("Manage Keywords")
        st.write("Add or remove keywords for PropTech-specific queries.")

        # Load current keywords
        keywords = load_keywords()

        # Add new keyword
        new_keyword = st.text_input("Add a new keyword:")
        if st.button("Add Keyword") and new_keyword:
            if new_keyword.strip():  # Ensure the keyword is not empty
                if new_keyword not in keywords:
                    keywords.append(new_keyword)
                    update_keywords(keywords)
                    show_temp_message(f"Keyword '{new_keyword}' added successfully!", "success")  # Temporary inline success message
                else:
                    show_temp_message(f"Keyword '{new_keyword}' already exists.", "warning")  # Temporary inline warning message
            else:
                show_temp_message("Please enter a valid keyword.", "warning")  # Temporary inline warning message

        # Remove keyword
        if keywords:  # Ensure there are keywords to remove
            remove_keyword = st.selectbox("Select a keyword to remove:", keywords)
            if st.button("Remove Keyword"):
                keywords.remove(remove_keyword)
                update_keywords(keywords)
                show_temp_message(f"Keyword '{remove_keyword}' removed successfully!", "success")  # Temporary inline success message
        else:
            st.warning("No keywords available to remove.")  # Persistent warning message

    elif app_mode == "About":
        st.header("About PropTech Smart Home Insights")
        st.write("""
        ### 🏠 PropTech Smart Home Insights
        This application provides:
        - AI-powered Q&A about PropTech and Smart Homes
        - Web research capabilities
        - Cutting-edge insights into property technology

        #### Features:
        - Advanced language model for generating responses
        - Web scraping for latest PropTech information
        - User-friendly interface

        #### Technologies Used:
        - GPT-2 (fine-tuned for PropTech)
        - Google Custom Search API
        - Streamlit
        """)

if __name__ == "__main__":
    main()
