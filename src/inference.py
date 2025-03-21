from transformers import GPT2Tokenizer, GPT2LMHeadModel
from web_scraping import scrape_proptech_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response(query: str) -> str:
    """
    Generate a response using a hybrid model (GPT-2 + Google Custom Search API).
    """
    try:
        # Handle empty or whitespace-only queries
        if not query or not query.strip():
            return "Please enter a valid question."

        # Step 1: Query understanding (using GPT-2)
        model_path = "models/proptech_gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        # Step 2: Fetch real-time data from Google Custom Search API
        results = scrape_proptech_data(query)

        # Step 3: Combine results with GPT-2's knowledge
        if results:
            # Format results into a response
            response = "Here's what I found:\n"
            for result in results:
                response += f"- {result['title']}: {result['description']}\n"
        else:
            # Fallback to GPT-2 if no results are found
            inputs = tokenizer(query, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {e}"
