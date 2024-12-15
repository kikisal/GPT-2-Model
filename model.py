
import torch
import torch.nn as nn
from torch.nn import functional as F


epochs        = 5000
block_size    = 256
learning_rate = 3e-4
batch_size    = 64

device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd        = 384
n_layer       = 6
n_head        = 6
dropout       = .2

print(f'device: {device}')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda arr: ''.join([itos[e] for e in arr])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))

train_data = data[:n]
val_data   = data[n:]


x = train_data[:block_size]
y = train_data[1: block_size + 1]


#for t in range(block_size):
#    context = x[:t+1]
#    target  = y[t]
#    print(f'when input is {context} target is {target}')
    

torch.manual_seed(1337)



def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size, ))
    x    = torch.stack([data[i: i + block_size] for i in ix])
    y    = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    return x, y


Xb, Yb = get_batch('train')

def print_batch():
    for b in range(batch_size):
        print(f'batch n {b + 1}')
        for t in range(block_size):
            context = Xb[b, :t + 1]
            target  = Yb[b, t]
            print(f'when input is [{decode(context.tolist())}] target is {itos[int(target)]}')
        print('---------------------')

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()

        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value =  nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # compute attention scores "affinities"

        wei = q @ k.transpose(-2, -1) * C ** -.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class GPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
    
        self.l_n = nn.LayerNorm(n_embd)
            

        self.sa_head                  = MultiHeadAttention(4, n_embd//4)
        self.lm_head                  = nn.Linear(n_embd, vocab_size)
        self.ffw                      = FeedForward(n_embd)


    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens):
            # (B, T)
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond) # (B, T, vocab_size)

            logits = logits[:, -1, :] # (B, C)

            probs = F.softmax(logits, dim=-1) # (B, vocab_size)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

    def forward(self, idx, targets=None):
        [B, T] = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)

        x       = tok_emb + pos_emb # (B, T, C)
        x       = self.blocks(x)

        x       = self.l_n(x)

        logits  = self.lm_head(x) # (B, T, vocab_size)
        
        loss   = None
        if targets is not None:            
            [B, T, C] = logits.shape

            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)
    
        return logits, loss
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size   = n_embd // n_head
        self.sa     = MultiHeadAttention(n_head, head_size)
        self.ffwd   = FeedForward(n_embd)
        self.ln1    = nn.LayerNorm(n_embd)
        self.ln2    = nn.LayerNorm(n_embd)
        

    def forward(self, x):
    
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

m = GPT2Model()

[logits, loss] = m(Xb, Yb)

#out = m.generate(torch.tensor([[4]]), 100)

print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for steps in range(epochs):
    Xb, Yb = get_batch('train')

    logits, loss = m(Xb, Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


gen = f'LOSS: {loss.item()}\n\n'

for g in range(300):
    gen += f'------------------------ GEN {g + 1} ------------------------\n\n'
    gen += decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())
    gen += '\n\n------------------------------------------------------------------------------------\n'


with open("out.txt", "w") as outf:
    outf.write(gen)
