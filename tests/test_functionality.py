
import unittest
import logging
import json
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now import the application modules
from inference import generate_response
from web_scraping import scrape_proptech_data
from app import load_keywords, update_keywords, show_temp_message, main
from data_preprocessing import preprocess_data
from fine_tuning import fine_tune_model

# Configure logging
logging.basicConfig(level=logging.CRITICAL)  # Suppress logs during testing

class TestPropTechLLM(unittest.TestCase):
    """Test suite for PropTech LLM application"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary keywords file for testing
        self.test_keywords = ["smart home", "IoT", "property technology"]
        with open("test_keywords.json", "w") as f:
            json.dump({"keywords": self.test_keywords}, f)
        
        # Mock environment variables
        os.environ["GOOGLE_API_KEY"] = "test_api_key"
        os.environ["SEARCH_ENGINE_ID"] = "test_search_engine_id"

    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary files
        if os.path.exists("test_keywords.json"):
            os.remove("test_keywords.json")

    def test_inference_valid_query(self):
        """Test inference with a valid query"""
        with patch('inference.GPT2Tokenizer.from_pretrained') as mock_tokenizer, \
             patch('inference.GPT2LMHeadModel.from_pretrained') as mock_model, \
             patch('inference.scrape_proptech_data') as mock_scrape:
            
            # Mock the model and tokenizer
            mock_tokenizer.return_value.decode.return_value = "Smart homes improve energy efficiency and comfort."
            mock_model.return_value.generate.return_value = [[1, 2, 3]]
            
            # Mock web scraping results
            mock_scrape.return_value = [
                {"title": "Smart Home Benefits", "description": "Energy efficiency and comfort", "link": "http://example.com"}
            ]
            
            query = "What are the benefits of smart home devices?"
            response = generate_response(query)
            
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 10)
            self.assertIn("Smart Home Benefits", response)

    def test_inference_empty_query(self):
        """Test inference with an empty query"""
        with patch('inference.GPT2Tokenizer.from_pretrained') as mock_tokenizer, \
             patch('inference.GPT2LMHeadModel.from_pretrained') as mock_model, \
             patch('inference.scrape_proptech_data') as mock_scrape:
            
            query = ""
            response = generate_response(query)
            
            self.assertEqual(response, "Please enter a valid question.")

    def test_web_scraping_empty_results(self):
        """Test web scraping with no results"""
        with patch('web_scraping.requests.get') as mock_get:
            # Mock the API response with no items
            mock_response = MagicMock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            query = "NonexistentPropTechTerm"
            results = scrape_proptech_data(query)
            
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 0)

    def test_web_scraping_api_error(self):
        """Test web scraping with API error"""
        with patch('web_scraping.requests.get') as mock_get:
            # Mock the API response with an error
            mock_get.side_effect = Exception("API Error")
            
            query = "PropTech smart home devices"
            results = scrape_proptech_data(query)
            
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 0)

    def test_load_keywords_success(self):
        """Test loading keywords from file"""
        with patch('app.Path') as mock_path:
            # Mock the file path
            mock_path.return_value.exists.return_value = True
            mock_instance = MagicMock()
            mock_path.return_value = mock_instance
            
            # Mock the open function
            m = unittest.mock.mock_open(read_data='{"keywords": ["smart home", "IoT"]}')
            with patch('builtins.open', m):
                keywords = load_keywords()
                
                self.assertIsInstance(keywords, list)
                self.assertEqual(len(keywords), 2)
                self.assertIn("smart home", keywords)
                self.assertIn("IoT", keywords)

    def test_load_keywords_file_not_found(self):
        """Test loading keywords when file doesn't exist"""
        with patch('app.Path') as mock_path:
            # Mock the file path to indicate file doesn't exist
            mock_path.return_value.exists.return_value = False
            
            keywords = load_keywords()
            
            self.assertIsInstance(keywords, list)
            self.assertEqual(len(keywords), 0)

    def test_load_keywords_error(self):
        """Test loading keywords with error"""
        with patch('app.Path') as mock_path:
            # Mock the file path
            mock_path.return_value.exists.return_value = True
            mock_instance = MagicMock()
            mock_path.return_value = mock_instance
            
            # Mock the open function to raise an exception
            m = unittest.mock.mock_open()
            m.side_effect = Exception("File read error")
            with patch('builtins.open', m):
                keywords = load_keywords()
                
                self.assertIsInstance(keywords, list)
                self.assertEqual(len(keywords), 0)

    def test_update_keywords_success(self):
        """Test updating keywords file"""
        # Mock the open function
        m = unittest.mock.mock_open()
        with patch('builtins.open', m):
            new_keywords = ["smart home", "IoT", "PropTech"]
            update_keywords(new_keywords)
            
            # Check if the file was written correctly
            m.assert_called_once_with("keywords.json", "w")
            handle = m()
            
            # Combine all write calls into a single string
            written_data = "".join(call[0][0] for call in handle.write.call_args_list)
            
            # Check if the correct JSON was written
            written_keywords = json.loads(written_data)["keywords"]
            self.assertEqual(written_keywords, new_keywords)

    def test_update_keywords_error(self):
        """Test updating keywords with error"""
        # Mock the open function to raise an exception
        m = unittest.mock.mock_open()
        m.side_effect = Exception("File write error")
        with patch('builtins.open', m):
            new_keywords = ["smart home", "IoT", "PropTech"]
            # This should log an error but not raise an exception
            update_keywords(new_keywords)

    def test_show_temp_message(self):
        """Test showing temporary message"""
        with patch('app.st.success') as mock_success, \
             patch('app.time.sleep') as mock_sleep:
            
            mock_placeholder = MagicMock()
            mock_success.return_value = mock_placeholder
            
            show_temp_message("Test message", "success", 2)
            
            mock_success.assert_called_once_with("Test message")
            mock_sleep.assert_called_once_with(2)
            mock_placeholder.empty.assert_called_once()

    def test_main_function_qa_mode(self):
        """Test main function in Q&A mode"""
        with patch('app.st.sidebar.selectbox') as mock_selectbox, \
             patch('app.st.text_input') as mock_text_input, \
             patch('app.st.spinner') as mock_spinner, \
             patch('app.generate_response') as mock_generate_response, \
             patch('app.st.success') as mock_success, \
             patch('app.st.write') as mock_write:
            
            # Mock the sidebar selection
            mock_selectbox.return_value = "Q&A Assistant"
            # Mock the text input
            mock_text_input.return_value = "What are smart thermostats?"
            # Mock the spinner context manager
            mock_spinner.return_value.__enter__ = MagicMock()
            mock_spinner.return_value.__exit__ = MagicMock()
            # Mock the response generation
            mock_generate_response.return_value = "Smart thermostats are devices that automate temperature control."
            
            # Call the main function
            main()
            
            # Check if the correct function was called
            mock_selectbox.assert_called_once()
            mock_text_input.assert_called_once()
            mock_generate_response.assert_called_once_with("What are smart thermostats?")
            mock_success.assert_called_once()
            mock_write.assert_called_once()

    def test_main_function_web_research_mode(self):
        """Test main function in Web Research mode"""
        with patch('app.st.sidebar.selectbox') as mock_selectbox, \
             patch('app.st.text_input') as mock_text_input, \
             patch('app.st.spinner') as mock_spinner, \
             patch('app.scrape_proptech_data') as mock_scrape, \
             patch('app.st.subheader') as mock_subheader, \
             patch('app.st.markdown') as mock_markdown, \
             patch('app.st.write') as mock_write:
            
            # Mock the sidebar selection
            mock_selectbox.return_value = "Web Research"
            # Mock the text input
            mock_text_input.return_value = "PropTech trends 2025"
            # Mock the spinner context manager
            mock_spinner.return_value.__enter__ = MagicMock()
            mock_spinner.return_value.__exit__ = MagicMock()
            # Mock the web scraping results
            mock_scrape.return_value = [
                {"title": "PropTech Trends", "link": "https://example.com", "description": "Future trends in property technology."}
            ]
            
            # Call the main function
            main()
            
            # Check if the correct function was called
            mock_selectbox.assert_called_once()
            mock_text_input.assert_called_once()
            mock_scrape.assert_called_once_with("PropTech trends 2025")
            mock_subheader.assert_called_once()
            mock_markdown.assert_called()
            mock_write.assert_called()

    def test_main_function_about_page(self):
        """Test main function in About mode"""
        with patch('app.st.sidebar.selectbox') as mock_selectbox, \
             patch('app.st.header') as mock_header, \
             patch('app.st.write') as mock_write:
            
            # Mock the sidebar selection
            mock_selectbox.return_value = "About"
            
            # Call the main function
            main()
            
            # Check if the correct functions were called
            mock_selectbox.assert_called_once()
            mock_header.assert_called_once_with("About PropTech Smart Home Insights")
            mock_write.assert_called_once()

    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        with patch('data_preprocessing.pd.read_excel') as mock_read_excel, \
             patch('data_preprocessing.Path') as mock_path, \
             patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            
            # Mock the pandas DataFrame
            mock_df = MagicMock()
            mock_df['Main Category'] = ['Smart Home', 'IoT']
            mock_df['Sub-Categories'] = ['Thermostats', 'Security']
            mock_df['Expanded Keywords'] = ['energy efficiency', 'cameras']
            mock_df['text'] = ['Smart Home Thermostats energy efficiency', 'IoT Security cameras']
            mock_read_excel.return_value = mock_df
            
            # Mock the path operations
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            
            # Call the function
            preprocess_data("test_file.xlsx")
            
            # Check if the correct functions were called
            mock_read_excel.assert_called_once_with("test_file.xlsx")
            mock_path_instance.parent.mkdir.assert_called_once_with(exist_ok=True)
            mock_df['text'].to_csv.assert_called_once()

    def test_fine_tuning(self):
        with patch('fine_tuning.torch.device') as mock_device, \
            patch('fine_tuning.torch.cuda.is_available') as mock_cuda, \
            patch('fine_tuning.GPT2Tokenizer.from_pretrained') as mock_tokenizer, \
            patch('fine_tuning.GPT2LMHeadModel.from_pretrained') as mock_model, \
            patch('fine_tuning.load_dataset') as mock_load_dataset, \
            patch('fine_tuning.DataCollatorForLanguageModeling') as mock_collator, \
            patch('fine_tuning.TrainingArguments') as mock_args, \
            patch('fine_tuning.Trainer') as mock_trainer:
            
            # Mock CUDA availability
            mock_cuda.return_value = False
            mock_device.return_value = "cpu"
            
            # Mock tokenizer and model
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.return_value = mock_tokenizer_instance
            mock_model_instance = MagicMock()
            mock_model.return_value.to.return_value = mock_model_instance
            
            # Mock dataset
            mock_dataset_instance = MagicMock()
            mock_dataset_instance['train'].train_test_split.return_value = {
                'train': MagicMock(),
                'test': MagicMock()
            }
            mock_load_dataset.return_value = mock_dataset_instance
            
            # Mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance
            
            # Call the function
            fine_tune_model()
            
            # Check if the correct functions were called
            mock_cuda.assert_called()  # Check if called at least once
            mock_tokenizer.assert_called_once_with("gpt2")
            mock_model.assert_called_once_with("gpt2")
            mock_load_dataset.assert_called_once()
            mock_collator.assert_called_once()
            mock_args.assert_called_once()
            mock_trainer.assert_called_once()
            mock_trainer_instance.train.assert_called_once()
            mock_trainer_instance.save_model.assert_called_once_with("models/proptech_gpt2")
            mock_tokenizer_instance.save_pretrained.assert_called_once_with("models/proptech_gpt2")

if __name__ == "__main__":
    unittest.main()
