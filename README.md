# 📌 PropTech LLM  
A GPT-2-based AI assistant for answering PropTech-related questions and web research.  

## 🚀 Features  
✅ Fine-tuned **GPT-2 model** for PropTech queries  
✅ Dynamic keyword updates for better responses  
✅ Web scraping for real-time property insights  
✅ Streamlit-based UI for easy interaction  

---

## ⚙️ Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/GamageThanuja/proptech-llm.git
cd PropTech_LLM


Setup
1️⃣ Clone the Repository
git clone https://github.com/GamageThanuja/proptech-llm.git
cd PropTech_LLM

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Preprocess the Data
python src/data_preprocessing.py

5️⃣ Fine-tune the Model
python src/fine_tuning.py

6️⃣ Run Inference
python src/inference.py

7️⃣ Start the Web App
streamlit run src/app.py

🛠 Testing
python -m unittest tests/test_functionality.py


📜 Project Structure
PropTech_LLM/
│── src/
│   ├── data_preprocessing.py   # Data processing script
│   ├── fine_tuning.py          # GPT-2 fine-tuning script
│   ├── inference.py            # Model inference script
│   ├── app.py                  # Streamlit UI script
│── models/                      # Saved model files
│── data/                        # Dataset storage
│── tests/                       # Unit tests
│── requirements.txt             # Dependencies
│── README.md                    # Project Documentation
│── .gitignore                    # Git Ignore File


📜 License
This project is licensed under the MIT License.
