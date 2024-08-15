import torch
import argparse
from tqdm import tqdm
import os

# Import your GPT model and config
from model import GPT, GPTConfig

def load_model(checkpoint_path):
    config = GPTConfig(
        block_size=35,
        vocab_size=50304,  # Adjust if your vocab size is different
        n_layer=2,
        n_head=2,
        n_embd=256,
        dropout=0.0,
        bias=True
    )
    model = GPT(config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model

def extract_features(model, texts, tokenizer, args):
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    
    all_features = []
    
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i:i+args.batch_size]
        
        # Tokenize
        tokens = tokenizer(batch_texts, padding=True, truncation=True, max_length=model.config.block_size, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Extract features based on the pooling method
            if args.pool == 'last':
                features = outputs[0][:, -1, :]  # Last hidden state
            elif args.pool == 'avg':
                features = outputs[0].mean(dim=1)  # Average of all hidden states
            
            all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save extracted features")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--pool", type=str, default="last", choices=["last", "avg"], help="Pooling method for features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")
    args = parser.parse_args()

    # Load the model
    model = load_model(args.checkpoint)
    
    # Load the tokenizer (you might need to adjust this based on your tokenizer)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load input texts
    with open(args.input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    texts = [text.strip() for text in texts]
    
    # Extract features
    features = extract_features(model, texts, tokenizer, args)
    
    # Save features
    torch.save(features, args.output_file)
    print(f"Features saved to {args.output_file}")

if __name__ == "__main__":
    main()