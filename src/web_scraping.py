import requests
import logging
from urllib.parse import quote
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Custom Search API credentials
API_KEY = st.secrets["GOOGLE_API_KEY"]
SEARCH_ENGINE_ID = st.secrets["SEARCH_ENGINE_ID"]

def scrape_proptech_data(query: str, max_results: int = 5) -> list:
    """
    Fetch real-time PropTech data using Google Custom Search API.
    """
    try:
        encoded_query = quote(query)
        url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}&num={max_results}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "items" not in data:
            logger.warning(f"No search results found for query: {query}")
            return []

        results = []
        for item in data["items"]:
            results.append({
                "title": item.get("title", "No title"),
                "link": item.get("link", "#"),
                "description": item.get("snippet", "No description available"),
            })
        return results
    except Exception as e:
        logger.error(f"Error during API request: {e}")
        return []
