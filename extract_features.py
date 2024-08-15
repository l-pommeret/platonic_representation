import torch
import argparse
from tqdm import tqdm
import os
import pickle

# Import your GPT model and config
from model import GPT, GPTConfig

class GPTWithHiddenStates(GPT):
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        hidden_states = [x]
        for block in self.transformer.h:
            x = block(x)
            hidden_states.append(x)
        
        x = self.transformer.ln_f(x)
        hidden_states.append(x)

        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss, hidden_states

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    
    model = GPTWithHiddenStates(config)
    
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    print(f"Model configuration: n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}")
    return model

def tokenize_and_pad(texts, tokenizer, block_size):
    tokenized = [torch.tensor([tokenizer(c) for c in text]) for text in texts]
    max_len = min(max(len(t) for t in tokenized), block_size)
    padded = torch.full((len(texts), max_len), tokenizer(';'))  # Assuming ';' is the padding token
    for i, t in enumerate(tokenized):
        padded[i, :len(t)] = t[:max_len]
    return padded

def extract_features(model, texts, tokenizer, args):
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    
    all_hidden_states = []
    
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i:i+args.batch_size]
        
        input_ids = tokenize_and_pad(batch_texts, tokenizer, model.config.block_size)
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            _, _, hidden_states = model(input_ids)
            
            # Convert hidden states to CPU and store
            hidden_states = [h.cpu() for h in hidden_states]
            all_hidden_states.append(hidden_states)
    
    # Combine hidden states from all batches
    combined_hidden_states = [torch.cat([batch[i] for batch in all_hidden_states], dim=0) for i in range(len(all_hidden_states[0]))]
    
    return combined_hidden_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save extracted features")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    
    with open('data/txt/meta.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    
    tokenizer = lambda x: vocab_info['stoi'].get(x, vocab_info['stoi'][';'])  # Use ';' for unknown tokens
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    texts = [text.strip() for text in texts]
    
    hidden_states = extract_features(model, texts, tokenizer, args)
    
    torch.save(hidden_states, args.output_file)
    print(f"Hidden states saved to {args.output_file}")
    print(f"Number of hidden state tensors: {len(hidden_states)}")
    print(f"Shape of first hidden state tensor: {hidden_states[0].shape}")

if __name__ == "__main__":
    main()