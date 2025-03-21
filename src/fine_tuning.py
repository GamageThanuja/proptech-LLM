import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from pathlib import Path

def fine_tune_model():
    """
    Fine-tunes GPT-2 on the PropTech dataset.
    """
    try:
        # Set device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained GPT-2 model and tokenizer
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

        # Ensure the tokenizer adds a pad token
        tokenizer.pad_token = tokenizer.eos_token

        # Load the preprocessed dataset
        dataset = load_dataset("text", data_files="data/proptech_training_data.txt")

        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, max_length=256, padding="max_length")

        # Tokenize the dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Split dataset into training and evaluation sets (90% train, 10% eval)
        train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="models/proptech_gpt2",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_dir="logs",
            logging_steps=100,
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            eval_steps=500,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # Pass the evaluation dataset
        )

        # Fine-tune the model
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model("models/proptech_gpt2")
        tokenizer.save_pretrained("models/proptech_gpt2")
        print("Fine-tuning complete. Model saved to 'models/proptech_gpt2'.")
    except Exception as e:
        print(f"Error during fine-tuning: {e}")

if __name__ == "__main__":
    fine_tune_model()
