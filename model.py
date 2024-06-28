
import torch
import torch.nn as nn
from torch.nn import functional as F


block_size = 8
batch_size = 32

with open('input2.txt', 'r', encoding='utf-8') as f:
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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # (B, T, C)

            logits = logits[:, -1, :] # (B, C)

            probs = F.softmax(logits, dim=-1) # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        loss   = None
        if targets is not None:            
            [B, T, C] = logits.shape

            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)
    
        return logits, loss

m = BigramLanguageModel(vocab_size)

[logits, loss] = m(Xb, Yb)

#out = m.generate(torch.tensor([[4]]), 100)

print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

'''
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32

for steps in range(10000):
    Xb, Yb = get_batch('train')

    logits, loss = m(Xb, Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(loss.item())

for g in range(300):
    print(f'------------------------ GEN {g + 1} ------------------------')
    print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
    print('------------------------------------------------------------------------------------')
'''

torch.manual_seed(1337)
B, T, C = 4, 8, 2

x = torch.rand(B, T, C)

xbow = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):
        xprev = x[b,:t + 1] # (t+1, C)
        xbow[b, t] = torch.mean(xprev, 0)

weight = torch.tril(torch.ones(T, T))
weight = weight/weight.sum(1, keepdim=True)

xbow2 = weight @ x

print(xbow[0])
print(xbow2[0])