import torch

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

def tokenize_and_pad(texts, tokenizer, max_length):
    """Tokenize and pad a list of texts."""
    tokenized = [[tokenizer(c) for c in text] for text in texts]
    padded = torch.full((len(texts), max_length), tokenizer(';'))
    for i, seq in enumerate(tokenized):
        length = min(len(seq), max_length)
        padded[i, :length] = torch.tensor(seq[:length])
    return padded