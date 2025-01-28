import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "gpt2"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT2 maximum sequence length
MODEL_MAX_LENGTH = model.config.max_position_embeddings  # Usually 1024 for GPT2

# Function to calculate perplexity for a chunk
def calculate_chunk_perplexity(chunk, tokenizer, model):
    try:
        tokens = tokenizer.encode(chunk, return_tensors="pt", truncation=True, max_length=MODEL_MAX_LENGTH)
        if tokens.size(1) == 0:
            print(f"Empty token sequence for chunk: {chunk[:50]}...")
            return None
        with torch.no_grad():
            outputs = model(tokens, labels=tokens)
            loss = outputs.loss
            if loss is None or torch.isnan(loss):
                print(f"Loss is NaN for chunk: {chunk[:50]}...")
                return None
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Error processing chunk: {chunk[:50]}... | Error: {e}")
        return None

# Function to handle long texts by splitting into chunks
def calculate_text_perplexity(text, tokenizer, model):
    tokens = tokenizer.encode(text)
    if len(tokens) <= MODEL_MAX_LENGTH:
        # If the sequence is within the model's maximum length, process directly
        return calculate_chunk_perplexity(text, tokenizer, model)
    else:
        # Split into chunks of up to MODEL_MAX_LENGTH tokens
        chunks = [tokens[i:i + MODEL_MAX_LENGTH] for i in range(0, len(tokens), MODEL_MAX_LENGTH)]
        chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        perplexities = []
        for chunk in chunk_texts:
            chunk_perplexity = calculate_chunk_perplexity(chunk, tokenizer, model)
            if chunk_perplexity is not None:
                perplexities.append(chunk_perplexity)
        if perplexities:
            return sum(perplexities) / len(perplexities)  # Average perplexity over chunks
        return None

# Function to process dataset in batches for NaN perplexity rows
def recalculate_na_perplexity_in_batches(dataset, batch_size=100, save_path="Dataset_with_new_features.csv"):
    rows_with_na = dataset[dataset['perplexity'].isna()]  # Filter rows with NaN perplexity
    print(f"Total rows with NaN perplexity: {len(rows_with_na)}")
    
    total_rows = len(rows_with_na)
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_indices = rows_with_na.index[start:end]  # Indices of the current batch
        print(f"Processing batch {start // batch_size + 1} ({start}-{end})...")
        
        for idx in batch_indices:
            row = dataset.loc[idx]
            perplexity = calculate_text_perplexity(row["text"], tokenizer, model)
            dataset.at[idx, "perplexity"] = perplexity  # Update the dataset
        
        # Save progress after each batch
        dataset.to_csv(save_path, index=False)
        print(f"Completed batch {start // batch_size + 1} and saved to {save_path}")

# Load dataset
dataset = pd.read_csv("Dataset_with_new_features.csv")

# Ensure 'perplexity' column exists
if "perplexity" not in dataset.columns:
    dataset["perplexity"] = None

# Clean text column to avoid issues
dataset['text'] = dataset['text'].fillna('').str.strip()

# Recalculate perplexity for NaN rows in batches
recalculate_na_perplexity_in_batches(dataset, batch_size=100, save_path="Dataset_with_new_features.csv")
