import pandas as pd
from pathlib import Path

def preprocess_data(file_path):
    """
    Preprocesses the PropTech dataset for fine-tuning.
    """
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)

        # Combine columns into a single text column
        df['text'] = df['Main Category'].astype(str) + " " + df['Sub-Categories'].astype(str) + " " + df['Expanded Keywords'].astype(str)

        # Save the processed data as a text file
        output_path = Path("data/proptech_training_data.txt")
        output_path.parent.mkdir(exist_ok=True)
        df['text'].to_csv(output_path, index=False, header=False)

        print(f"Data preprocessing complete. Training data saved to '{output_path}'.")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")

if __name__ == "__main__":
    # Path to the dataset
    file_path = Path("data/PropTech LLM Keywords.xlsx")
    preprocess_data(file_path)
