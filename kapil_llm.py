"""
╔════════════════════════════════════════════════════════════════╗
║  KAPIL'S TINY LLM — Scratch se banaya hua mini Transformer     ║
║  Data: Hinglish sentences about dost, chai, coding             ║
║  Architecture: Character-level Transformer (like GPT, chota)   ║
╚════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════
# STEP 1: TRAINING DATA (Hinglish corpus)
# ═══════════════════════════════════════════════════════════════
text = """rahul ek engineer hai. rahul python code likhta hai.
rahul ko chai bahut pasand hai. rahul roz chai peeta hai.
kapil bhi engineer hai. kapil bhi python code likhta hai.
kapil ko coffee pasand hai. kapil roz coffee peeta hai.
priya ek doctor hai. priya hospital jaati hai roz.
priya ko juice pasand hai. priya roz juice peeti hai.
rahul aur kapil dost hai. dono saath code likhte hai.
kapil aur priya bhi dost hai. dono kanpur mein rehte hai.
python ek programming language hai. python bahut popular hai.
engineers python use karte hai. engineers code likhte hai roz.
chai garam hoti hai aur meethi hoti hai. coffee bhi garam hoti hai.
juice thanda hota hai aur meetha hota hai. sab drinks achi hoti hai.
kapil kanpur mein rehta hai. rahul delhi mein rehta hai.
priya mumbai mein rehti hai. teeno alag alag sheher mein hai.
kapil ka startup hai. startup ka naam khetikaro hai.
khetikaro ek farming startup hai. kisano ko help karta hai.
rahul kapil ke startup mein invest karna chahta hai.
priya bhi startup mein interested hai. sab mil ke kaam karenge.
coding maza aata hai. python easy language hai. sab log seekh sakte hai.
"""

# Corpus ko bada karo taaki model achhe se seekhe
text = text * 20

# ═══════════════════════════════════════════════════════════════
# STEP 2: TOKENIZATION (character-level — sabse simple)
# ═══════════════════════════════════════════════════════════════
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)

print(f"📚 Vocab size: {vocab_size} unique characters")
print(f"📖 Total training tokens: {len(data):,}")
print(f"🔤 Characters: {''.join(chars)}")

# ═══════════════════════════════════════════════════════════════
# STEP 3: HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════
BLOCK_SIZE = 32       # context window — kitne chars piche dekhega
BATCH_SIZE = 32       # ek baar mein kitne examples
EMBED_DIM = 64        # har char ka vector size
N_HEADS = 4           # multi-head attention
N_LAYERS = 3          # kitne transformer blocks stack karenge
DROPOUT = 0.1
LR = 3e-3
EPOCHS = 3000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════
# STEP 4: BATCH BANANE KA FUNCTION
# Ek batch = (input, target) jodi — target = input shifted by 1
# ═══════════════════════════════════════════════════════════════
def get_batch():
    ix = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ═══════════════════════════════════════════════════════════════
# STEP 5: TRANSFORMER — Ye hai asli magic
# ═══════════════════════════════════════════════════════════════

class Head(nn.Module):
    """Ek single attention head — 'kis word pe dhyaan du?' calculate karta hai"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        # Causal mask — future words nahi dekh sakta (GPT-style)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # "main kya offer karta hu"
        q = self.query(x)    # "main kya dhundh raha hu"
        # Attention scores — Q aur K ka dot product
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)    # "meri actual info"
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple heads parallel mein — har head alag pattern seekhta hai"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Simple MLP — thinking karne ki jagah"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    """Ek Transformer block = Attention + FeedForward + LayerNorm"""
    def __init__(self):
        super().__init__()
        head_size = EMBED_DIM // N_HEADS
        self.sa = MultiHeadAttention(N_HEADS, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        # Residual connections + pre-normalization
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class KapilLLM(nn.Module):
    """Full mini-GPT — sab kuch jodta hai"""
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb           # meaning + position
        x = self.blocks(x)              # transformer magic
        x = self.ln_f(x)
        logits = self.head(x)           # har char ki probability
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """Ek-ek char generate karo, sampling se"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ═══════════════════════════════════════════════════════════════
# STEP 6: MODEL BANAO AUR TRAIN KARO
# ═══════════════════════════════════════════════════════════════
model = KapilLLM().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 Model parameters: {total_params:,} (~{total_params/1000:.1f}K)")
print(f"💻 Device: {DEVICE}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"\n🚀 Training started ({EPOCHS} epochs)...")
print("-" * 60)
for epoch in range(EPOCHS):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if epoch % 300 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:4d}/{EPOCHS} → loss = {loss.item():.4f}")

print("-" * 60)
print("✅ Training complete!\n")

# ═══════════════════════════════════════════════════════════════
# STEP 7: GENERATE KARO — apna LLM chalao!
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("🎯 GENERATED TEXT (tumhara LLM baat kar raha hai)")
print("=" * 60)

prompts = ["kapil ", "rahul ", "python ", "chai ", "khetikaro "]
for prompt in prompts:
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=150, temperature=0.8)[0].tolist()
    output = decode(generated)
    print(f"\n💬 Prompt: '{prompt}'")
    print(f"   Output: {output}")
    print("-" * 60)
