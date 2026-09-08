"""
╔══════════════════════════════════════════════════════════╗
║  MonoceptGPT — ChatGPT-style UI for your tiny LLM          ║
║  Run: python3 app.py  →  http://localhost:5000          ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template_string, request, Response, stream_with_context, jsonify
import jira_client
import teams_store

torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════
# TRAINING DATA & TOKENIZER (same as before)
# ═══════════════════════════════════════════════════════════
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
coding maza aata hai. python easy language hai. sab log seekh sakte hai.
kapil ko coding pasand hai. rahul bhi coding karta hai roz.
kanpur ek shehar hai. delhi bhi ek shehar hai. mumbai bhi shehar hai.
sab dost mil ke coding seekhte hai. sab log ache dost hai.
""" * 20

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
def encode(s): return [stoi.get(c, 0) for c in s.lower()]
def decode(l): return ''.join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)

# ═══════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════
BLOCK_SIZE = 32
BATCH_SIZE = 32
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1
DEVICE = 'cpu'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
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
    def __init__(self):
        super().__init__()
        head_size = EMBED_DIM // N_HEADS
        self.sa = MultiHeadAttention(N_HEADS, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class KapilLLM(nn.Module):
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
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None: return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# ═══════════════════════════════════════════════════════════
# TRAIN or LOAD
# ═══════════════════════════════════════════════════════════
model = KapilLLM().to(DEVICE)

def get_batch():
    ix = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"📂 Loading saved model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
else:
    print("🚀 Training model (first time only, ~2 min)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for epoch in range(3000):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    print(f"✅ Model saved to {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════
# GENERATION (streaming char by char)
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def generate_stream(prompt, max_tokens=200, temperature=0.8):
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    for _ in range(max_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        yield itos[idx_next.item()]

# ═══════════════════════════════════════════════════════════
# FLASK APP + UI
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>MonoceptGPT</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Söhne', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #212121;
    color: #ececec;
    height: 100vh;
    display: flex;
    overflow: hidden;
  }

  /* Sidebar */
  .sidebar {
    width: 260px;
    background: #171717;
    padding: 12px;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #2a2a2a;
    overflow: hidden;
  }
  #chat-history {
    max-height: 40vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .new-chat-btn {
    background: transparent;
    color: #ececec;
    border: 1px solid #3a3a3a;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.15s;
  }
  .new-chat-btn:hover { background: #2a2a2a; }
  .sidebar-title {
    color: #8e8ea0;
    font-size: 12px;
    margin: 20px 8px 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .chat-search-wrap {
    position: relative;
    margin: 4px 0 8px;
  }
  .chat-search-icon {
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    color: #666;
    font-size: 12px;
    pointer-events: none;
  }
  .chat-search-input {
    width: 100%;
    background: #0f0f0f;
    border: 1px solid #2a2a2a;
    color: #ececec;
    padding: 7px 30px 7px 30px;
    border-radius: 8px;
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s, background 0.15s;
  }
  .chat-search-input:focus {
    border-color: #FF6B35;
    background: #1a1a1a;
  }
  .chat-search-input::placeholder { color: #666; }
  .chat-search-clear {
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    background: transparent;
    border: none;
    color: #666;
    cursor: pointer;
    font-size: 16px;
    padding: 2px 6px;
    border-radius: 4px;
    display: none;
    line-height: 1;
  }
  .chat-search-clear.visible { display: block; }
  .chat-search-clear:hover { color: #ececec; background: #2a2a2a; }

  .history-item {
    padding: 8px 10px 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    color: #ececec;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background 0.15s;
    margin-bottom: 2px;
  }
  .history-item:hover { background: #2a2a2a; }
  .history-item.active { background: #2a2a2a; border-left: 2px solid #FF6B35; padding-left: 10px; }
  .history-item .hist-title {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .history-item .hist-del {
    background: transparent;
    border: none;
    color: #666;
    cursor: pointer;
    font-size: 16px;
    padding: 2px 6px;
    border-radius: 4px;
    visibility: hidden;
    flex-shrink: 0;
    line-height: 1;
  }
  .history-item:hover .hist-del { visibility: visible; }
  .history-item .hist-del:hover { color: #ff9999; background: #3a1a1a; }
  .sidebar-footer {
    margin-top: auto;
    padding: 10px 8px 6px;
    border-top: 1px solid #2a2a2a;
    font-size: 12px;
    color: #8e8ea0;
  }
  .footer-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 4px;
    border-radius: 8px;
  }
  .footer-row .avatar {
    width: 34px;
    height: 34px;
    flex-shrink: 0;
  }
  .footer-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .footer-name {
    color: #ececec;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .footer-status {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #8e8ea0;
    line-height: 1;
  }
  .footer-status .s-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #666;
    box-shadow: none;
    display: inline-block;
  }
  .footer-status.on .s-dot {
    background: #22c55e;
    box-shadow: 0 0 6px rgba(34,197,94,0.6);
  }
  .footer-status.on { color: #b8e5ca; }
  .footer-gear {
    background: transparent;
    color: #8e8ea0;
    border: none;
    padding: 6px 8px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 16px;
    transition: color 0.15s, background 0.15s;
    flex-shrink: 0;
  }
  .footer-gear:hover { color: #FF6B35; background: #2a2a2a; }
  .avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #FF6B35, #C7431A);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    color: white;
    font-size: 13px;
  }

  /* Main area */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #212121;
  }
  .header {
    padding: 14px 20px;
    border-bottom: 1px solid #2a2a2a;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: 600;
  }
  .header .badge {
    background: #FF6B35;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }

  .chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px 0;
  }
  .chat-inner {
    max-width: 780px;
    margin: 0 auto;
    padding: 0 20px;
  }

  /* Welcome screen */
  .welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    padding: 20px;
  }
  .welcome h1 {
    font-size: 30px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #FF6B35, #FF9F6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .welcome p { color: #8e8ea0; margin-bottom: 30px; font-size: 14px; }
  .suggestions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 600px;
    width: 100%;
  }
  .suggestion {
    background: #2a2a2a;
    padding: 14px 16px;
    border-radius: 12px;
    cursor: pointer;
    text-align: left;
    font-size: 14px;
    transition: background 0.15s;
    border: 1px solid transparent;
  }
  .suggestion:hover { background: #333; border-color: #444; }
  .suggestion .title { font-weight: 500; margin-bottom: 4px; }
  .suggestion .subtitle { color: #8e8ea0; font-size: 12px; }

  /* Messages */
  .message {
    display: flex;
    gap: 16px;
    padding: 20px 0;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 13px;
  }
  .user-avatar {
    background: linear-gradient(135deg, #555, #333);
    color: white;
  }
  .bot-avatar {
    background: linear-gradient(135deg, #FF6B35, #C7431A);
    color: white;
  }
  .msg-body { flex: 1; padding-top: 4px; min-width: 0; overflow: hidden; }
  .msg-role {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 4px;
  }
  .msg-content {
    font-size: 15px;
    line-height: 1.6;
    color: #ececec;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .cursor {
    display: inline-block;
    width: 8px;
    height: 16px;
    background: #FF6B35;
    animation: blink 1s infinite;
    vertical-align: middle;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* Input area */
  .input-area {
    padding: 20px;
    max-width: 780px;
    margin: 0 auto;
    width: 100%;
  }
  .input-wrapper {
    position: relative;
    background: #2a2a2a;
    border-radius: 24px;
    display: flex;
    align-items: flex-end;
    padding: 8px;
    border: 1px solid #3a3a3a;
    transition: border 0.15s;
  }
  .input-wrapper:focus-within { border-color: #FF6B35; }
  #prompt {
    flex: 1;
    background: transparent;
    border: none;
    color: #ececec;
    font-size: 15px;
    font-family: inherit;
    padding: 10px 14px;
    resize: none;
    outline: none;
    max-height: 200px;
    line-height: 1.5;
  }
  #send-btn {
    background: #FF6B35;
    border: none;
    color: white;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s;
    flex-shrink: 0;
  }
  #send-btn:hover { background: #E5551F; }
  #send-btn:disabled { background: #3a3a3a; cursor: not-allowed; }
  .disclaimer {
    text-align: center;
    color: #8e8ea0;
    font-size: 12px;
    margin-top: 10px;
  }

  /* Settings gear */
  .settings-btn {
    background: transparent;
    color: #8e8ea0;
    border: none;
    padding: 6px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 18px;
    transition: background 0.15s, color 0.15s;
  }
  .settings-btn:hover { background: #2a2a2a; color: #ececec; }

  /* Modal */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 100;
    animation: fadeIn 0.2s;
  }
  .modal-backdrop.open { display: flex; }
  .modal {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    width: 500px;
    max-width: 90vw;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }
  .modal-header {
    padding: 18px 20px;
    border-bottom: 1px solid #3a3a3a;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .modal-header h2 { font-size: 18px; font-weight: 600; }
  .modal-close {
    background: transparent;
    border: none;
    color: #8e8ea0;
    font-size: 24px;
    cursor: pointer;
    line-height: 1;
    padding: 0 4px;
  }
  .modal-close:hover { color: #ececec; }
  .modal-body { padding: 20px; }
  .form-group { margin-bottom: 16px; }
  .form-label {
    display: block;
    font-size: 13px;
    color: #ececec;
    margin-bottom: 6px;
    font-weight: 500;
  }
  .form-hint {
    font-size: 11px;
    color: #8e8ea0;
    margin-top: 4px;
  }
  .form-input {
    width: 100%;
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    color: #ececec;
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 14px;
    font-family: inherit;
    outline: none;
    transition: border 0.15s;
  }
  .form-input:focus { border-color: #FF6B35; }
  .form-input.mono { font-family: 'SF Mono', Menlo, monospace; font-size: 13px; }
  .modal-footer {
    padding: 14px 20px;
    border-top: 1px solid #3a3a3a;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }
  .btn {
    padding: 8px 16px;
    border-radius: 8px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
    font-family: inherit;
  }
  .btn-primary { background: #FF6B35; color: white; }
  .btn-primary:hover { background: #E5551F; }
  .btn-primary:disabled { background: #3a3a3a; cursor: not-allowed; }
  .btn-secondary { background: transparent; color: #ececec; border: 1px solid #3a3a3a; }
  .btn-secondary:hover { background: #3a3a3a; }
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
  }
  .status-badge.connected { background: #FF6B35; color: white; }
  .status-badge.disconnected { background: #3a3a3a; color: #8e8ea0; }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: white; }
  .save-status {
    font-size: 13px;
    padding: 8px 12px;
    border-radius: 6px;
    margin-top: 10px;
    display: none;
  }
  .save-status.success { background: #1a3a2a; color: #7ade9f; display: block; }
  .save-status.error { background: #3a1a1a; color: #ff9999; display: block; }
  .info-box {
    background: #1a2a3a;
    border-left: 3px solid #FF9F6B;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 12px;
    color: #b8d4f0;
    margin-bottom: 16px;
    line-height: 1.5;
  }
  .info-box a { color: #FFB88C; text-decoration: none; }
  .info-box a:hover { text-decoration: underline; }

  /* JIRA cards */
  .jira-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    color: #8e8ea0;
    font-size: 12px;
  }
  .jira-header .jql-pill {
    background: #1a3a5a;
    color: #FFB88C;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'SF Mono', Menlo, monospace;
    font-size: 11px;
  }
  .jira-card {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border 0.15s, background 0.15s;
  }
  .jira-card:hover { border-color: #FF6B35; background: #303030; }
  .jira-card-top {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
    flex-wrap: wrap;
  }
  .jira-key {
    font-family: 'SF Mono', Menlo, monospace;
    color: #FFB88C;
    font-weight: 600;
    text-decoration: none;
    font-size: 13px;
  }
  .jira-key:hover { text-decoration: underline; }
  .jira-status {
    background: #FF6B35;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }
  .jira-status.open, .jira-status.todo { background: #FF9F6B; }
  .jira-status.progress { background: #f5a623; }
  .jira-status.done, .jira-status.closed { background: #666; }
  .jira-type {
    font-size: 11px;
    color: #8e8ea0;
    padding: 2px 6px;
    background: #1a1a1a;
    border-radius: 4px;
  }
  .jira-priority {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 4px;
    background: #2a1a1a;
    color: #ff9999;
  }
  .jira-summary {
    font-size: 14px;
    color: #ececec;
    margin-bottom: 6px;
    line-height: 1.4;
  }
  .jira-meta {
    font-size: 12px;
    color: #8e8ea0;
    display: flex;
    gap: 12px;
  }
  /* Quick access buttons */
  .quick-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    background: linear-gradient(135deg, rgba(220,38,38,0.12), rgba(220,38,38,0.04));
    border: 1px solid rgba(220,38,38,0.35);
    color: #ececec;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    text-align: left;
    transition: background 0.15s, border-color 0.15s;
  }
  .quick-btn:hover {
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(220,38,38,0.08));
    border-color: rgba(220,38,38,0.6);
  }
  .quick-btn .q-icon { font-size: 15px; }
  .quick-btn .q-label { flex: 1; }
  .quick-btn .q-count {
    background: #dc2626;
    color: white;
    font-size: 11px;
    padding: 1px 8px;
    border-radius: 10px;
    font-weight: 600;
    min-width: 24px;
    text-align: center;
    display: none;
  }
  .quick-btn .q-count.visible { display: inline-block; }

  /* Production issues view */
  .prod-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(220,38,38,0.05));
    border: 1px solid rgba(220,38,38,0.35);
    border-radius: 12px;
    margin-bottom: 14px;
  }
  .prod-header .p-title { font-size: 17px; font-weight: 700; color: #ececec; display: flex; align-items: center; gap: 8px; }
  .prod-header .p-sub { font-size: 12px; color: #8e8ea0; margin-top: 4px; }
  .prod-header .p-jql {
    font-family: 'SF Mono', Menlo, monospace;
    background: rgba(0,0,0,0.3);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    color: #b8d4f0;
    margin-top: 6px;
    display: inline-block;
    word-break: break-all;
  }
  .prod-header .p-count {
    margin-left: auto;
    text-align: center;
    flex-shrink: 0;
  }
  .prod-header .p-count .n {
    font-size: 28px;
    font-weight: 700;
    color: #dc2626;
    line-height: 1;
  }
  .prod-header .p-count .l {
    font-size: 11px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .prod-open-link {
    color: #FFB88C;
    font-size: 12px;
    text-decoration: none;
    margin-top: 4px;
    display: inline-block;
  }
  .prod-open-link:hover { color: #FF6B35; }
  .prod-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 14px;
  }
  .prod-stat-block {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 10px 12px;
  }
  .prod-stat-block h4 {
    font-size: 11px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    font-weight: 600;
  }
  .prod-stat-block .row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 2px 0;
    color: #ececec;
  }
  .prod-stat-block .row .n { color: #FFB88C; font-family: 'SF Mono', Menlo, monospace; }
  .prod-issue-wrap {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    margin-bottom: 4px;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .prod-issue-wrap:hover { border-color: #dc2626; }
  .prod-issue-wrap.expanded { border-color: #dc2626; }
  .prod-issue-row {
    display: grid;
    grid-template-columns: 20px 100px 90px 1fr auto auto;
    gap: 10px;
    align-items: center;
    padding: 10px 14px;
    font-size: 12px;
    cursor: pointer;
  }
  .prod-issue-row .p-arrow {
    color: #666;
    font-size: 10px;
    transition: transform 0.15s, color 0.15s;
  }
  .prod-issue-wrap.expanded .p-arrow {
    transform: rotate(90deg);
    color: #dc2626;
  }
  .prod-issue-details {
    display: none;
    padding: 0 14px 14px;
    border-top: 1px solid #3a3a3a;
    background: #1a1a1a;
  }
  .prod-issue-wrap.expanded .prod-issue-details { display: block; }
  .prod-detail-section { margin-top: 12px; }
  .prod-detail-section h5 {
    font-size: 11px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    font-weight: 600;
  }
  .prod-desc {
    background: #2a2a2a;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 12px;
    line-height: 1.6;
    color: #ececec;
    white-space: pre-wrap;
    max-height: 260px;
    overflow-y: auto;
  }
  .prod-meta {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px;
    font-size: 11px;
    background: #2a2a2a;
    padding: 10px 12px;
    border-radius: 6px;
  }
  .prod-meta .k { color: #8e8ea0; }
  .prod-meta .v { color: #ececec; word-break: break-word; }
  .prod-actions {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
  }
  .prod-actions .btn {
    padding: 6px 12px;
    font-size: 12px;
  }
  .prod-issue-row .p-date { font-family: 'SF Mono', Menlo, monospace; color: #8e8ea0; font-size: 11px; }
  .prod-issue-row .p-key {
    font-family: 'SF Mono', Menlo, monospace;
    color: #FFB88C;
    text-decoration: none;
    font-weight: 600;
  }
  .prod-issue-row .p-key:hover { color: #FF6B35; }
  .prod-issue-row .p-sum {
    color: #ececec;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .prod-issue-row .p-assignee {
    color: #8e8ea0;
    font-size: 11px;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Teams panel */
  .team-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    color: #ececec;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.15s;
    margin-bottom: 2px;
  }
  .team-item:hover { background: #2a2a2a; }
  .team-item .team-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .team-item .team-count {
    background: #FF6B35;
    color: white;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    font-weight: 600;
  }
  .team-item .team-idle {
    background: #f5a623;
    color: white;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    font-weight: 600;
  }
  .add-team-btn {
    background: transparent;
    color: #8e8ea0;
    border: 1px dashed #3a3a3a;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 12px;
    width: 100%;
    text-align: left;
    transition: color 0.15s, border-color 0.15s;
  }
  .add-team-btn:hover { color: #FF6B35; border-color: #FF6B35; }

  /* Team view in chat */
  .team-view-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(255,107,53,0.05));
    border: 1px solid rgba(255,107,53,0.3);
    border-radius: 12px;
    margin-bottom: 14px;
  }
  .team-view-title { font-size: 17px; font-weight: 700; color: #ececec; }
  .team-view-subtitle { font-size: 12px; color: #8e8ea0; margin-top: 4px; }
  .team-view-stats { display: flex; gap: 16px; margin-left: auto; }
  .team-stat { text-align: center; }
  .team-stat .n { font-size: 22px; font-weight: 700; }
  .team-stat .l { font-size: 10px; color: #8e8ea0; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }
  .team-stat .n.busy { color: #FF6B35; }
  .team-stat .n.idle { color: #f5a623; }
  .team-stat .n.total { color: #ececec; }

  .member-row {
    display: grid;
    grid-template-columns: 40px 1fr 100px 60px auto;
    gap: 12px;
    align-items: center;
    padding: 12px 14px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    margin-bottom: 6px;
    transition: border-color 0.15s;
  }
  .member-row:hover { border-color: #FF6B35; }
  .member-row.idle {
    background: linear-gradient(90deg, rgba(245,166,35,0.12), transparent);
    border-color: rgba(245,166,35,0.4);
  }
  .member-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #FF6B35;
    background-size: cover;
    background-position: center;
  }
  .member-info .name { font-size: 14px; color: #ececec; font-weight: 500; }
  .member-info .latest { font-size: 11px; color: #8e8ea0; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .member-workload { text-align: center; }
  .member-workload .count {
    font-size: 20px;
    font-weight: 700;
    color: #FF6B35;
  }
  .member-workload .count.idle { color: #f5a623; }
  .member-workload .label {
    font-size: 10px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .member-flags {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 10px;
  }
  .flag-hi { background: #5a2a2a; color: #ff9999; padding: 2px 6px; border-radius: 4px; }
  .flag-od { background: #5a3a1a; color: #ffcc66; padding: 2px 6px; border-radius: 4px; }
  .idle-badge {
    background: #f5a623;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .remove-btn {
    background: transparent;
    border: none;
    color: #8e8ea0;
    cursor: pointer;
    font-size: 16px;
    padding: 4px 8px;
    border-radius: 4px;
  }
  .remove-btn:hover { color: #ff6b6b; background: #3a1a1a; }

  /* Search UI (member search + chat search) */
  .search-wrap {
    position: relative;
  }
  .search-wrap .search-icon {
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: #8e8ea0;
    font-size: 16px;
    pointer-events: none;
  }
  .search-wrap .search-clear {
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    background: transparent;
    border: none;
    color: #8e8ea0;
    cursor: pointer;
    font-size: 18px;
    padding: 2px 8px;
    border-radius: 4px;
    display: none;
  }
  .search-wrap .search-clear:hover {
    color: #ececec;
    background: #3a3a3a;
  }
  .search-wrap .search-clear.visible { display: block; }
  .search-wrap .search-spinner {
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    width: 14px;
    height: 14px;
    border: 2px solid #3a3a3a;
    border-top-color: #FF6B35;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    display: none;
  }
  .search-wrap .search-spinner.visible { display: block; }
  @keyframes spin {
    to { transform: translateY(-50%) rotate(360deg); }
  }
  .search-wrap .form-input {
    padding-left: 36px;
    padding-right: 40px;
  }

  .search-hint {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 14px;
    background: #1a2a3a;
    border-left: 3px solid #4a90e2;
    border-radius: 6px;
    font-size: 12px;
    color: #b8d4f0;
    margin-top: 8px;
  }
  .search-hint.warn {
    background: #3a2a1a;
    border-left-color: #f5a623;
    color: #ffcc66;
  }
  .search-hint.empty-results {
    background: #2a2a2a;
    border-left-color: #666;
    color: #8e8ea0;
    justify-content: center;
    padding: 20px 14px;
    text-align: center;
    flex-direction: column;
  }
  .search-hint.empty-results .big { font-size: 24px; margin-bottom: 4px; }

  .search-results {
    margin-top: 8px;
    max-height: 260px;
    overflow-y: auto;
  }
  .search-results-header {
    font-size: 11px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 6px 4px 4px;
    display: flex;
    justify-content: space-between;
  }

  /* Team management modal — member row with workload */
  .tm-member {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    margin-bottom: 6px;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .tm-member:hover { border-color: #FF6B35; }
  .tm-member.idle { border-color: rgba(245,166,35,0.5); }
  .tm-member.expanded { border-color: #FF6B35; }
  .tm-member-row {
    display: grid;
    grid-template-columns: 32px 1fr auto auto auto;
    gap: 10px;
    align-items: center;
    padding: 10px 12px;
    cursor: pointer;
  }
  .tm-member-row .avatar-sm {
    width: 32px; height: 32px; border-radius: 50%;
    background: #FF6B35; background-size: cover;
  }
  .tm-member-row .m-name { font-size: 13px; color: #ececec; font-weight: 500; }
  .tm-member-row .m-email { font-size: 11px; color: #8e8ea0; margin-top: 1px; }
  .tm-workload-pill {
    background: #FF6B35;
    color: white;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 10px;
    font-weight: 600;
    min-width: 40px;
    text-align: center;
  }
  .tm-workload-pill.idle {
    background: #f5a623;
  }
  .tm-workload-pill.busy-high {
    background: #d94a4a;
  }
  .tm-today-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10a37f;
    box-shadow: 0 0 8px rgba(16,163,127,0.6);
  }
  .tm-today-dot.no-work {
    background: #666;
    box-shadow: none;
  }
  .tm-expand-arrow {
    color: #8e8ea0;
    transition: transform 0.15s;
    font-size: 12px;
  }
  .tm-member.expanded .tm-expand-arrow {
    transform: rotate(90deg);
    color: #FF6B35;
  }
  .tm-member-details {
    padding: 0 12px 14px;
    border-top: 1px solid #2a2a2a;
    display: none;
  }
  .tm-member.expanded .tm-member-details { display: block; }
  .tm-details-section { margin-top: 12px; }
  .tm-details-section h5 {
    font-size: 11px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tm-details-section h5 .count-badge {
    background: #2a2a2a;
    color: #ececec;
    padding: 1px 6px;
    border-radius: 8px;
    font-size: 10px;
  }
  .tm-task-row {
    display: grid;
    grid-template-columns: 80px 90px 1fr auto auto;
    gap: 10px;
    align-items: center;
    padding: 6px 8px;
    background: #2a2a2a;
    border-radius: 6px;
    margin-bottom: 3px;
    font-size: 12px;
    transition: background 0.15s;
  }
  .tm-log-btn {
    background: transparent;
    border: 1px solid #3a3a3a;
    color: #FF6B35;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  .tm-log-btn:hover {
    background: rgba(255,107,53,0.15);
    border-color: #FF6B35;
  }
  .tm-task-row:hover { background: #333; }
  .tm-task-row .t-date { font-family: 'SF Mono', Menlo, monospace; color: #8e8ea0; font-size: 11px; }
  .tm-task-row .t-key {
    font-family: 'SF Mono', Menlo, monospace;
    color: #FFB88C;
    text-decoration: none;
    font-weight: 600;
    font-size: 11px;
  }
  .tm-task-row .t-key:hover { color: #FF6B35; }
  .tm-task-row .t-sum {
    color: #ececec;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tm-task-row .t-status {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 8px;
    background: #444;
    color: #ececec;
    white-space: nowrap;
  }
  .tm-no-today {
    background: rgba(245,166,35,0.1);
    border: 1px solid rgba(245,166,35,0.3);
    color: #ffcc66;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    text-align: center;
  }
  .tm-remove-btn {
    background: transparent;
    border: none;
    color: #666;
    cursor: pointer;
    font-size: 18px;
    padding: 4px 8px;
    border-radius: 4px;
    transition: color 0.15s, background 0.15s;
  }
  .tm-remove-btn:hover { color: #ff9999; background: #3a1a1a; }

  .user-search-result {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 4px;
    background: #1a1a1a;
    border: 1px solid transparent;
    transition: background 0.15s, border-color 0.15s;
  }
  .user-search-result:hover {
    background: #2a2a2a;
    border-color: #FF6B35;
  }
  .user-search-result.already-added {
    cursor: default;
    opacity: 0.55;
  }
  .user-search-result.already-added:hover {
    background: #1a1a1a;
    border-color: transparent;
  }
  .user-search-result .avatar-sm {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #FF6B35;
    background-size: cover;
    background-position: center;
    flex-shrink: 0;
  }
  .user-search-result .name-block { flex: 1; min-width: 0; }
  .user-search-result .name-block .n {
    font-size: 13px;
    color: #ececec;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .user-search-result .name-block .e {
    font-size: 11px;
    color: #8e8ea0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-top: 1px;
  }
  .user-search-result .action-cue {
    font-size: 11px;
    color: #FF6B35;
    display: flex;
    align-items: center;
    gap: 4px;
    font-weight: 500;
    padding: 4px 10px;
    background: rgba(255,107,53,0.12);
    border-radius: 12px;
  }
  .user-search-result:hover .action-cue {
    background: #FF6B35;
    color: white;
  }
  .user-search-result.already-added .action-cue {
    background: rgba(16,163,127,0.15);
    color: #7ade9f;
  }
  .current-members-list {
    max-height: 200px;
    overflow-y: auto;
    margin-top: 10px;
  }

  /* Morning briefing on welcome */
  .morning-briefing {
    background: linear-gradient(135deg, #5a3a1a, #3a2a1a);
    border: 1px solid rgba(245,166,35,0.4);
    border-radius: 12px;
    padding: 16px 18px;
    margin: 20px 0;
    max-width: 600px;
    width: 100%;
  }
  .morning-briefing h4 {
    color: #f5a623;
    font-size: 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .morning-briefing .briefing-body {
    font-size: 13px;
    color: #ececec;
    line-height: 1.6;
  }
  .briefing-idle-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
  }
  .briefing-idle-list .pill {
    background: rgba(245,166,35,0.2);
    border: 1px solid rgba(245,166,35,0.4);
    color: #ffcc66;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
  }

  /* Universal search results (users + issues combined) */
  .search-section-title {
    font-size: 12px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
  }
  .search-section-title .count-badge {
    background: #FF6B35;
    color: white;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 10px;
  }
  .uni-users-grid {
    display: grid;
    gap: 6px;
    margin-bottom: 8px;
  }
  .uni-user-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .uni-user-card:hover {
    border-color: #FF6B35;
    background: #303030;
  }
  .uni-user-card .avatar-sm {
    width: 34px; height: 34px; border-radius: 50%;
    background: #FF6B35; background-size: cover; background-position: center;
    flex-shrink: 0;
  }
  .uni-user-name {
    font-size: 13px;
    color: #ececec;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .uni-user-email {
    font-size: 11px;
    color: #8e8ea0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-top: 2px;
  }
  .uni-issues-list {
    display: grid;
    gap: 6px;
  }
  .uni-issue-card {
    padding: 10px 14px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .uni-issue-card:hover {
    border-color: #FF6B35;
    background: #303030;
  }
  .uni-issue-top {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    flex-wrap: wrap;
  }
  .uni-issue-top .jira-key {
    color: #FFB88C;
    font-size: 12px;
  }
  .uni-issue-summary {
    font-size: 13px;
    color: #ececec;
    line-height: 1.4;
    margin-bottom: 4px;
  }
  .uni-issue-meta {
    font-size: 11px;
    color: #8e8ea0;
  }

  /* User search results */
  .user-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(255,107,53,0.05));
    border: 1px solid rgba(255,107,53,0.3);
    border-radius: 12px;
    margin-bottom: 14px;
  }
  .user-avatar-lg {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background-color: #FF6B35;
    background-size: cover;
    background-position: center;
    flex-shrink: 0;
    border: 2px solid rgba(255,107,53,0.5);
  }
  .user-name {
    font-size: 15px;
    font-weight: 600;
    color: #ececec;
  }
  .user-email {
    font-size: 12px;
    color: #8e8ea0;
    margin-top: 2px;
  }
  .user-count {
    margin-left: auto;
    text-align: right;
    font-size: 28px;
    font-weight: 700;
    color: #FF6B35;
    line-height: 1;
  }
  .user-count span {
    display: block;
    font-size: 11px;
    color: #8e8ea0;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
  }
  .task-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .task-row {
    display: grid;
    grid-template-columns: 90px 80px 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 10px 12px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    transition: background 0.15s, border 0.15s;
  }
  .task-row:hover {
    background: #303030;
    border-color: #FF6B35;
  }
  .task-date {
    font-family: 'SF Mono', Menlo, monospace;
    font-size: 12px;
    color: #8e8ea0;
  }
  .task-key {
    font-family: 'SF Mono', Menlo, monospace;
    font-size: 12px;
    font-weight: 600;
    color: #FFB88C;
    text-decoration: none;
  }
  .task-key:hover { color: #FF6B35; text-decoration: underline; }
  .task-summary {
    font-size: 13px;
    color: #ececec;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Dashboard */
  .dash-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }
  .dash-stat {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    padding: 14px;
  }
  .dash-stat .num {
    font-size: 28px;
    font-weight: 700;
    color: #FF6B35;
    margin-bottom: 2px;
  }
  .dash-stat .lbl {
    font-size: 12px;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .dash-stat.open .num { color: #FF9F6B; }
  .dash-stat.done .num { color: #FF6B35; }
  .dash-breakdown {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
  .dash-breakdown h3 {
    font-size: 13px;
    color: #ececec;
    margin-bottom: 10px;
    font-weight: 600;
  }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 12px;
  }
  .bar-label { min-width: 90px; color: #ececec; }
  .bar-track {
    flex: 1;
    background: #1a1a1a;
    height: 18px;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  .bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #FF6B35, #FF9F6B);
    border-radius: 4px;
    transition: width 0.4s;
  }
  .bar-count {
    min-width: 30px;
    text-align: right;
    color: #8e8ea0;
    font-family: 'SF Mono', Menlo, monospace;
    font-size: 11px;
  }

  /* Issue detail */
  .issue-detail {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 18px 20px;
    overflow: hidden;
    max-width: 100%;
  }
  .issue-detail-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    min-width: 0;
  }
  .issue-detail-header .log-work-btn {
    margin-left: auto;
    padding: 6px 14px;
    font-size: 12px;
    white-space: nowrap;
  }
  .issue-detail h3 {
    font-size: 17px;
    color: #ececec;
    margin: 10px 0;
    line-height: 1.4;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }
  .issue-meta-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin: 14px 0;
    padding: 12px;
    background: #1a1a1a;
    border-radius: 8px;
    font-size: 12px;
  }
  .issue-meta-grid > div {
    min-width: 0;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }
  .issue-meta-grid .k { color: #8e8ea0; }
  .issue-meta-grid .v { color: #ececec; }
  .issue-section {
    margin-top: 16px;
  }
  .issue-section-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8e8ea0;
    margin-bottom: 8px;
    font-weight: 600;
  }
  .issue-desc {
    font-size: 13px;
    color: #ececec;
    line-height: 1.6;
    white-space: pre-wrap;
    background: #1a1a1a;
    padding: 12px;
    border-radius: 8px;
    max-height: 200px;
    overflow-y: auto;
  }
  .comment {
    background: #1a1a1a;
    padding: 10px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .comment-head {
    color: #FFB88C;
    margin-bottom: 4px;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .comment-body {
    color: #ececec;
    line-height: 1.5;
    white-space: pre-wrap;
  }
  .reply-btn {
    background: transparent;
    border: 1px solid #3a3a3a;
    color: #FFB88C;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .reply-btn:hover {
    background: rgba(255,107,53,0.12);
    border-color: #FF6B35;
    color: #FF6B35;
  }
  .reply-box {
    margin-top: 8px;
    padding: 8px;
    background: #2a2a2a;
    border-radius: 6px;
    border-left: 2px solid #FF6B35;
    display: none;
  }
  .reply-box.open { display: block; }
  .reply-box textarea {
    width: 100%;
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    color: #ececec;
    font-family: inherit;
    font-size: 12px;
    padding: 6px 8px;
    border-radius: 4px;
    resize: vertical;
    min-height: 50px;
    outline: none;
  }
  .reply-box textarea:focus { border-color: #FF6B35; }
  .reply-box .actions {
    display: flex;
    gap: 6px;
    margin-top: 6px;
    justify-content: flex-end;
  }
  .reply-box .btn {
    padding: 4px 10px;
    font-size: 11px;
  }
  .reply-box .status {
    color: #ff9999;
    font-size: 11px;
    margin-top: 4px;
  }
  .reply-box .status.ok { color: #7ade9f; }
  .add-comment-btn {
    background: transparent;
    border: 1px dashed #3a3a3a;
    color: #FFB88C;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    width: 100%;
    transition: border-color 0.15s, color 0.15s;
    margin-top: 4px;
  }
  .add-comment-btn:hover {
    border-color: #FF6B35;
    color: #FF6B35;
  }
  .subtask {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    background: #1a1a1a;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 12px;
  }
  .subtask a { color: #FFB88C; text-decoration: none; font-family: 'SF Mono', Menlo, monospace; }
  .subtask a:hover { text-decoration: underline; }
  .label-pill {
    display: inline-block;
    background: #3a3a3a;
    color: #ececec;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    margin-right: 4px;
  }

  .jira-error {
    background: #3a1a1a;
    border: 1px solid #5a2a2a;
    color: #ff9999;
    padding: 14px;
    border-radius: 8px;
    font-size: 13px;
    white-space: pre-wrap;
    font-family: 'SF Mono', Menlo, monospace;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #4a4a4a; }
</style>
</head>
<body>
  <div class="sidebar">
    <button class="new-chat-btn" onclick="newChat()">
      <span>+</span> New chat
    </button>
    <div class="sidebar-title">Chats</div>
    <div class="chat-search-wrap">
      <span class="chat-search-icon">🔍</span>
      <input type="text" id="chat-search" class="chat-search-input" placeholder="Filter chats..." oninput="filterChats(this.value)"/>
      <button class="chat-search-clear" id="chat-search-clear" onclick="clearChatFilter()">×</button>
    </div>
    <div id="chat-history"></div>

    <div class="sidebar-title" style="margin-top: 16px;">Quick Access</div>
    <button class="quick-btn prod-issues-btn" onclick="showProductionIssues()">
      <span class="q-icon">🚨</span>
      <span class="q-label">Production Issues</span>
      <span class="q-count" id="prod-count"></span>
    </button>

    <div class="sidebar-title" style="margin-top: 16px;">Teams</div>
    <div id="teams-list"></div>
    <button class="add-team-btn" onclick="openNewTeamModal()">+ New team</button>

    <div class="sidebar-footer">
      <div class="footer-row">
        <div class="avatar">K</div>
        <div class="footer-info">
          <div class="footer-name">Kapil</div>
          <div class="footer-status" id="jira-status-badge">
            <span class="s-dot"></span><span>JIRA off</span>
          </div>
        </div>
        <button class="footer-gear" onclick="openSettings()" title="Settings">⚙</button>
      </div>
    </div>
  </div>

  <div class="main">
    <div class="header">
      <span>MonoceptGPT</span>
      <span class="badge">Tiny · Local</span>
    </div>

    <div class="chat-container" id="chat"></div>

    <!-- Welcome template — re-used every time we return to empty state -->
    <template id="welcome-template">
      <div class="welcome" id="welcome">
        <h1>MonoceptGPT</h1>
        <p>Quick JIRA task access. Type <code>today</code>, <code>mine</code>, or an issue ID.</p>
        <div class="suggestions">
          <div class="suggestion" onclick="sendPrompt('today')">
            <div class="title">📅 today</div>
            <div class="subtitle">Today's tasks</div>
          </div>
          <div class="suggestion" onclick="sendPrompt('week')">
            <div class="title">📆 week</div>
            <div class="subtitle">This week's tasks</div>
          </div>
          <div class="suggestion" onclick="sendPrompt('dashboard')">
            <div class="title">📊 dashboard</div>
            <div class="subtitle">Full analysis + charts</div>
          </div>
          <div class="suggestion" onclick="sendPrompt('mine')">
            <div class="title">👤 mine</div>
            <div class="subtitle">All assigned tasks</div>
          </div>
        </div>
      </div>
    </template>

    <div class="input-area">
      <div class="input-wrapper">
        <textarea
          id="prompt"
          placeholder="Search anything — name, issue key, or keyword (e.g. Kapil, A20M-2939, VYMO, login bug)"
          rows="1"
          onkeydown="handleKey(event)"
          oninput="autoResize(this)"></textarea>
        <button id="send-btn" onclick="sendMessage()">↑</button>
      </div>
      <div class="disclaimer">
        Search anything: <code>Kapil</code>, <code>A20M-2939</code>, <code>VYMO</code>, <code>login</code> · Commands: <code>today</code>, <code>mine</code>, <code>dashboard</code>
      </div>
    </div>
  </div>

  <!-- Worklog Modal -->
  <div class="modal-backdrop" id="worklog-modal" onclick="if(event.target===this) closeWorklogModal()">
    <div class="modal" style="width: 480px;">
      <div class="modal-header">
        <div>
          <h2>⏱ Log Work</h2>
          <div id="worklog-issue-title" style="font-size: 12px; color: #8e8ea0; margin-top: 2px;"></div>
        </div>
        <button class="modal-close" onclick="closeWorklogModal()">×</button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label class="form-label">Time Spent</label>
          <input type="text" id="worklog-time" class="form-input mono" placeholder="e.g. 30m, 1h, 2h 30m, 1d"/>
          <div class="form-hint">Format: <code>Nw Nd Nh Nm</code> (weeks/days/hours/minutes)</div>
          <div style="display:flex; gap:6px; margin-top:8px; flex-wrap: wrap;">
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="setWorklogTime('15m')">15m</button>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="setWorklogTime('30m')">30m</button>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="setWorklogTime('1h')">1h</button>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="setWorklogTime('2h')">2h</button>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="setWorklogTime('4h')">4h</button>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="setWorklogTime('8h')">8h (full day)</button>
          </div>
        </div>
        <div class="form-group" style="margin-top: 16px;">
          <label class="form-label">Comment (optional)</label>
          <textarea id="worklog-comment" class="form-input" rows="3" placeholder="What did you work on?" style="resize: vertical; min-height: 60px;"></textarea>
        </div>
        <div class="save-status" id="worklog-status"></div>
      </div>
      <div class="modal-footer">
        <div></div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-secondary" onclick="closeWorklogModal()">Cancel</button>
          <button class="btn btn-primary" id="worklog-submit-btn" onclick="submitWorklog()">Log Work</button>
        </div>
      </div>
    </div>
  </div>

  <!-- New Team Modal -->
  <div class="modal-backdrop" id="new-team-modal" onclick="if(event.target===this) closeNewTeamModal()">
    <div class="modal">
      <div class="modal-header">
        <h2>+ New Team</h2>
        <button class="modal-close" onclick="closeNewTeamModal()">×</button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label class="form-label">Team Name</label>
          <input type="text" id="new-team-name" class="form-input" placeholder="e.g. MREC Team, BAU, Nudge Squad"/>
          <div class="form-hint">Give your team a clear name</div>
        </div>
      </div>
      <div class="modal-footer">
        <div></div>
        <div style="display:flex; gap: 8px;">
          <button class="btn btn-secondary" onclick="closeNewTeamModal()">Cancel</button>
          <button class="btn btn-primary" onclick="submitNewTeam()">Create Team</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Team View + Manage Modal -->
  <div class="modal-backdrop" id="manage-team-modal" onclick="if(event.target===this) closeManageTeamModal()">
    <div class="modal" style="width: 640px; max-height: 92vh;">
      <div class="modal-header">
        <div>
          <h2 id="manage-team-title">Team</h2>
          <div id="team-stats-line" style="font-size: 12px; color: #8e8ea0; margin-top: 2px;"></div>
        </div>
        <button class="modal-close" onclick="closeManageTeamModal()">×</button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label class="form-label">Add Member</label>
          <div class="search-wrap">
            <span class="search-icon">🔍</span>
            <input type="text" id="member-search" class="form-input" placeholder="Search by name or email — e.g. Kapil, Pratibha" oninput="onMemberSearch(this.value)"/>
            <div class="search-spinner" id="member-search-spinner"></div>
            <button class="search-clear" id="member-search-clear" onclick="clearMemberSearch()">×</button>
          </div>
          <div id="member-search-results"></div>
        </div>

        <div style="margin-top: 20px;">
          <label class="form-label">Members (<span id="current-count">0</span>) — click a name to see their tasks</label>
          <div id="current-members" style="margin-top: 8px;"></div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" onclick="deleteCurrentTeam()" style="color: #ff9999; border-color: #5a2a2a;">Delete Team</button>
        <button class="btn btn-primary" onclick="closeManageTeamModal(); loadTeams();">Done</button>
      </div>
    </div>
  </div>

  <!-- Settings Modal -->
  <div class="modal-backdrop" id="settings-modal" onclick="if(event.target===this) closeSettings()">
    <div class="modal">
      <div class="modal-header">
        <h2>⚙ Settings</h2>
        <button class="modal-close" onclick="closeSettings()">×</button>
      </div>
      <div class="modal-body">
        <div class="info-box">
          <strong>JIRA API Token required</strong><br/>
          Create token: <a href="https://id.atlassian.com/manage-profile/security/api-tokens" target="_blank">
          id.atlassian.com/manage-profile/security/api-tokens</a><br/>
          → "Create API token" → name it "MonoceptGPT" → copy the value
        </div>

        <div class="form-group">
          <label class="form-label">JIRA Domain</label>
          <input type="text" id="s-domain" class="form-input" placeholder="your-domain.atlassian.net"/>
          <div class="form-hint">Domain only — do not include https://</div>
        </div>

        <div class="form-group">
          <label class="form-label">Email</label>
          <input type="email" id="s-email" class="form-input" placeholder="you@company.com"/>
          <div class="form-hint">Your JIRA account email</div>
        </div>

        <div class="form-group">
          <label class="form-label">API Token</label>
          <input type="password" id="s-token" class="form-input mono" placeholder="ATATT3xFf..."/>
          <div class="form-hint" id="token-hint">Saved to local .env file (chmod 600)</div>
        </div>

        <div class="save-status" id="save-status"></div>
      </div>
      <div class="modal-footer">
        <div id="current-status" style="font-size: 12px; color: #8e8ea0;"></div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-secondary" onclick="closeSettings()">Cancel</button>
          <button class="btn btn-primary" id="save-btn" onclick="saveSettings()">Save & Test</button>
        </div>
      </div>
    </div>
  </div>

<script>
  // ══════ CHAT HISTORY (persisted in localStorage) ══════
  let allChats = [];
  let activeChatId = null;
  const CHATS_KEY = 'monogpt_chats';
  const ACTIVE_KEY = 'monogpt_active_chat';

  function loadChatsFromStorage() {
    try {
      const raw = localStorage.getItem(CHATS_KEY);
      allChats = raw ? JSON.parse(raw) : [];
    } catch (e) { allChats = []; }
    const savedActive = localStorage.getItem(ACTIVE_KEY);
    activeChatId = savedActive && allChats.some(c => c.id === savedActive) ? savedActive : null;
  }

  function saveChatsToStorage() {
    try {
      localStorage.setItem(CHATS_KEY, JSON.stringify(allChats));
      if (activeChatId) localStorage.setItem(ACTIVE_KEY, activeChatId);
      else localStorage.removeItem(ACTIVE_KEY);
    } catch (e) {
      console.warn('localStorage quota reached — pruning oldest chats');
      // Keep only 30 most recent
      allChats = allChats.slice(0, 30);
      try { localStorage.setItem(CHATS_KEY, JSON.stringify(allChats)); } catch (e2) {}
    }
  }

  function activeChat() {
    return allChats.find(c => c.id === activeChatId);
  }

  let chatFilterQuery = '';

  function chatMatchesFilter(chat, q) {
    if (!q) return true;
    const query = q.toLowerCase();
    if (chat.title.toLowerCase().includes(query)) return true;
    // Deep search: look inside message text/html too
    for (const m of chat.messages || []) {
      if (m.role === 'user' && (m.text || '').toLowerCase().includes(query)) return true;
      if (m.role === 'bot' && (m.html || '').toLowerCase().includes(query)) return true;
    }
    return false;
  }

  function filterChats(q) {
    chatFilterQuery = (q || '').trim();
    document.getElementById('chat-search-clear').classList.toggle('visible', chatFilterQuery.length > 0);
    renderChatHistory();
  }

  function clearChatFilter() {
    document.getElementById('chat-search').value = '';
    filterChats('');
    document.getElementById('chat-search').focus();
  }

  function renderChatHistory() {
    const el = document.getElementById('chat-history');
    if (!allChats.length) {
      el.innerHTML = '<div style="color:#8e8ea0; font-size:12px; padding: 6px 12px;">No chats yet</div>';
      return;
    }
    const filtered = allChats.filter(c => chatMatchesFilter(c, chatFilterQuery));
    if (!filtered.length) {
      el.innerHTML = `<div style="color:#8e8ea0; font-size:12px; padding: 10px 12px; text-align:center;">
        No chats match "<b>${escapeHtml(chatFilterQuery)}</b>"
      </div>`;
      return;
    }
    // Render oldest → newest (newest at bottom, WhatsApp-style)
    const ordered = [...filtered].reverse();
    el.innerHTML = ordered.map(c => `
      <div class="history-item ${c.id === activeChatId ? 'active' : ''}" onclick="switchChat('${c.id}')">
        <span class="hist-title">${highlightMatch(c.title, chatFilterQuery)}</span>
        <button class="hist-del" onclick="event.stopPropagation(); deleteChat('${c.id}')" title="Delete">×</button>
      </div>`).join('');
    // Auto-scroll to bottom to show newest
    el.scrollTop = el.scrollHeight;
  }

  function highlightMatch(text, q) {
    if (!q) return escapeHtml(text);
    const idx = text.toLowerCase().indexOf(q.toLowerCase());
    if (idx < 0) return escapeHtml(text);
    return escapeHtml(text.slice(0, idx)) +
           '<mark style="background: rgba(255,107,53,0.35); color: #fff; padding: 0 2px; border-radius: 2px;">' +
           escapeHtml(text.slice(idx, idx + q.length)) +
           '</mark>' +
           escapeHtml(text.slice(idx + q.length));
  }

  function switchChat(id) {
    activeChatId = id;
    saveChatsToStorage();
    renderChatHistory();
    renderActiveChatMessages();
  }

  function deleteChat(id) {
    if (!confirm('Delete this chat?')) return;
    allChats = allChats.filter(c => c.id !== id);
    if (activeChatId === id) {
      activeChatId = allChats.length ? allChats[0].id : null;
    }
    saveChatsToStorage();
    renderChatHistory();
    renderActiveChatMessages();
  }

  function newChat() {
    activeChatId = null;
    saveChatsToStorage();
    renderChatHistory();
    renderActiveChatMessages();
    document.getElementById('prompt').focus();
  }

  function ensureActiveChat(firstPrompt) {
    if (activeChatId && activeChat()) return;
    const id = Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
    const title = (firstPrompt || 'New chat').substring(0, 40);
    allChats.unshift({
      id, title,
      timestamp: Date.now(),
      messages: []
    });
    activeChatId = id;
  }

  function pushMessage(role, payload) {
    // payload = { text } for user, { html } for bot
    const c = activeChat();
    if (!c) return -1;
    c.messages.push(Object.assign({ role }, payload));
    c.timestamp = Date.now();
    saveChatsToStorage();
    return c.messages.length - 1;
  }

  function updateBotMessage(idx, html) {
    const c = activeChat();
    if (!c || idx < 0 || idx >= c.messages.length) return;
    c.messages[idx].html = html;
    saveChatsToStorage();
  }

  function renderWelcome() {
    document.getElementById('chat').innerHTML = document.getElementById('welcome-template').innerHTML;
    // Re-run morning briefing after re-rendering
    if (typeof showMorningBriefing === 'function') {
      showMorningBriefing();
    }
  }

  function renderActiveChatMessages() {
    const chat = document.getElementById('chat');
    const c = activeChat();
    if (!c || !c.messages.length) {
      renderWelcome();
      return;
    }
    let html = '<div class="chat-inner">';
    for (const m of c.messages) {
      html += `
        <div class="message">
          <div class="msg-avatar ${m.role === 'user' ? 'user-avatar' : 'bot-avatar'}">${m.role === 'user' ? 'U' : 'M'}</div>
          <div class="msg-body">
            <div class="msg-role">${m.role === 'user' ? 'You' : 'MonoceptGPT'}</div>
            <div class="msg-content">${m.role === 'user' ? escapeHtml(m.text) : (m.html || '')}</div>
          </div>
        </div>`;
    }
    html += '</div>';
    chat.innerHTML = html;
    chat.scrollTop = chat.scrollHeight;
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function sendPrompt(text) {
    document.getElementById('prompt').value = text;
    sendMessage();
  }

  function addMessage(role, content) {
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();
    const chat = document.getElementById('chat');
    if (!chat.querySelector('.chat-inner')) {
      chat.innerHTML = '<div class="chat-inner"></div>';
    }
    const inner = chat.querySelector('.chat-inner');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';
    msgDiv.innerHTML = `
      <div class="msg-avatar ${role === 'user' ? 'user-avatar' : 'bot-avatar'}">
        ${role === 'user' ? 'U' : 'M'}
      </div>
      <div class="msg-body">
        <div class="msg-role">${role === 'user' ? 'You' : 'MonoceptGPT'}</div>
        <div class="msg-content"></div>
      </div>`;
    inner.appendChild(msgDiv);
    msgDiv.querySelector('.msg-content').textContent = content;
    chat.scrollTop = chat.scrollHeight;
    return msgDiv.querySelector('.msg-content');
  }

  const JIRA_KEY_RE = /\b([A-Z][A-Z0-9]+-\d+)\b/;

  function detectIntent(prompt) {
    const lower = prompt.toLowerCase().trim();
    const upper = prompt.toUpperCase();
    const keyMatch = upper.match(JIRA_KEY_RE);

    if (['/jira dashboard', '/jira analysis', 'dashboard', 'analysis', 'summary', 'overview'].includes(lower)) return 'dashboard';
    if (['today', 'today work', "today's work", 'today task', "today's task", 'todays task', 'todays work'].includes(lower)) return 'today';
    if (['week', 'this week', 'weekly'].includes(lower)) return 'week';
    if (['/jira mine', 'mine', 'my tasks', 'my task', 'assigned to me'].includes(lower)) return 'mine';
    if (lower.startsWith('/jira ') && keyMatch) return 'issue_key';
    if (lower.startsWith('/jira')) return 'jql_command';
    if (keyMatch) return 'issue_key';
    return 'universal_search';
  }

  // Fire-and-forget log call
  function logQuery(query, intent, matchedKey, source) {
    fetch('/log/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query, intent,
        matched_key: matchedKey || null,
        source: source || 'chat',
        chat_id: activeChatId,
      })
    }).catch(() => {});  // silent
  }

  async function sendMessage() {
    const input = document.getElementById('prompt');
    const prompt = input.value.trim();
    if (!prompt) return;
    input.value = '';
    autoResize(input);
    document.getElementById('send-btn').disabled = true;

    // Persist: ensure active chat + save user message
    ensureActiveChat(prompt);
    pushMessage('user', { text: prompt });
    const botIdx = pushMessage('bot', { html: '' });
    renderChatHistory();

    // Log for training data
    const intent = detectIntent(prompt);
    const upper = prompt.toUpperCase();
    const matchedKey = (upper.match(JIRA_KEY_RE) || [])[1] || null;
    logQuery(prompt, intent, matchedKey, 'chat');

    addMessage('user', prompt);
    const botContent = addMessage('bot', '');
    botContent.innerHTML = '<span class="cursor"></span>';

    try {
      const lower = prompt.toLowerCase().trim();
      const keyMatch = upper.match(JIRA_KEY_RE);

      const todayPhrases = ['today', 'today work', "today's work", 'today task', "today's task",
                            'todays task', 'todays work'];
      const dashboardPhrases = ['/jira dashboard', '/jira analysis', 'dashboard', 'analysis',
                                 'summary', 'overview'];
      const minePhrases = ['/jira mine', 'mine', 'my tasks', 'my task', 'assigned to me'];
      const weekPhrases = ['week', 'this week', 'weekly'];

      if (dashboardPhrases.includes(lower)) {
        await handleDashboard(botContent);
      } else if (lower.startsWith('/jira ') && keyMatch) {
        await handleIssueDetail(keyMatch[1], botContent);
      } else if (todayPhrases.includes(lower)) {
        await handleJira('/jira today', botContent);
      } else if (weekPhrases.includes(lower)) {
        await handleJira('/jira week', botContent);
      } else if (minePhrases.includes(lower)) {
        await handleJira('/jira mine', botContent);
      } else if (lower.startsWith('/jira')) {
        await handleJira(prompt, botContent);
      } else if (keyMatch) {
        await handleIssueDetail(keyMatch[1], botContent);
      } else {
        await handleUniversalSearch(prompt, botContent);
      }
    } catch (err) {
      botContent.textContent = 'Error: ' + err.message;
    } finally {
      // Save final bot HTML
      updateBotMessage(botIdx, botContent.innerHTML);
      document.getElementById('send-btn').disabled = false;
      input.focus();
    }
  }

  async function handleDashboard(botContent) {
    botContent.innerHTML = '<span style="color:#8e8ea0">📊 Dashboard loading...</span>';
    const res = await fetch('/jira/dashboard');
    const d = await res.json();
    if (d.error) {
      botContent.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
      return;
    }
    const maxStatus = Math.max(...d.by_status.map(x => x[1]));
    const maxPrio = Math.max(...d.by_priority.map(x => x[1]));
    const maxProj = Math.max(...d.by_project.map(x => x[1]));

    let html = `
      <div class="jira-header">
        <span>📊 Your JIRA Analysis · ${d.total} tasks total</span>
      </div>
      <div class="dash-grid">
        <div class="dash-stat"><div class="num">${d.total}</div><div class="lbl">Total assigned</div></div>
        <div class="dash-stat open"><div class="num">${d.open}</div><div class="lbl">Open</div></div>
        <div class="dash-stat done"><div class="num">${d.done}</div><div class="lbl">Done</div></div>
        <div class="dash-stat"><div class="num">${d.by_project.length}</div><div class="lbl">Projects</div></div>
      </div>

      <div class="dash-breakdown">
        <h3>📁 By Project</h3>
        ${d.by_project.map(([k,v]) => `
          <div class="bar-row">
            <div class="bar-label">${escapeHtml(k)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${(v/maxProj*100).toFixed(0)}%"></div></div>
            <div class="bar-count">${v}</div>
          </div>`).join('')}
      </div>

      <div class="dash-breakdown">
        <h3>🚦 By Status</h3>
        ${d.by_status.map(([k,v]) => `
          <div class="bar-row">
            <div class="bar-label">${escapeHtml(k)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${(v/maxStatus*100).toFixed(0)}%"></div></div>
            <div class="bar-count">${v}</div>
          </div>`).join('')}
      </div>

      <div class="dash-breakdown">
        <h3>⚡ By Priority</h3>
        ${d.by_priority.map(([k,v]) => `
          <div class="bar-row">
            <div class="bar-label">${escapeHtml(k)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${(v/maxPrio*100).toFixed(0)}%"></div></div>
            <div class="bar-count">${v}</div>
          </div>`).join('')}
      </div>

      <div class="dash-breakdown">
        <h3>🕐 Recent Activity (Top 5)</h3>
        ${d.recent.map(r => `
          <div class="jira-card" style="margin-bottom:6px;">
            <div class="jira-card-top">
              <a href="${r.url}" target="_blank" class="jira-key">${escapeHtml(r.key)}</a>
              <span class="jira-status ${r.status.toLowerCase().replace(/\\s+/g,'')}">${escapeHtml(r.status)}</span>
              <span class="jira-priority">${escapeHtml(r.priority)}</span>
            </div>
            <div class="jira-summary" style="font-size:13px;">${escapeHtml(r.summary)}</div>
            <div class="jira-meta">📅 ${escapeHtml(r.updated)}</div>
          </div>`).join('')}
      </div>
    `;
    botContent.innerHTML = html;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
  }

  async function handleIssueDetail(key, botContent) {
    botContent.innerHTML = `<span style="color:#8e8ea0">🔍 Fetching ${key}...</span>`;
    const res = await fetch(`/jira/issue/${encodeURIComponent(key)}`);
    const d = await res.json();
    if (d.error) {
      botContent.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
      return;
    }
    const statusClass = d.status.toLowerCase().replace(/\s+/g, '');
    let html = `
      <div class="issue-detail">
        <div class="issue-detail-header">
          <a href="${d.url}" target="_blank" class="jira-key" style="font-size:15px;">${escapeHtml(d.key)}</a>
          <span class="jira-status ${statusClass}">${escapeHtml(d.status)}</span>
          <span class="jira-type">${escapeHtml(d.type)}</span>
          <span class="jira-priority">${escapeHtml(d.priority)}</span>
          <button class="btn btn-primary log-work-btn"
                  onclick="openWorklogModal('${escapeAttr(d.key)}', '${escapeAttr((d.summary || '').replace(/"/g, '&quot;'))}')">
            ⏱ Log Work
          </button>
        </div>
        <h3>${escapeHtml(d.summary)}</h3>

        <div class="issue-meta-grid">
          <div><span class="k">Assignee:</span> <span class="v">${escapeHtml(d.assignee)}</span></div>
          <div><span class="k">Reporter:</span> <span class="v">${escapeHtml(d.reporter)}</span></div>
          <div><span class="k">Created:</span> <span class="v">${escapeHtml(d.created)}</span></div>
          <div><span class="k">Updated:</span> <span class="v">${escapeHtml(d.updated)}</span></div>
          <div><span class="k">Due:</span> <span class="v">${escapeHtml(d.duedate)}</span></div>
          <div><span class="k">Resolution:</span> <span class="v">${escapeHtml(d.resolution)}</span></div>
        </div>
    `;

    if (d.labels && d.labels.length) {
      html += `<div class="issue-section">
        <div class="issue-section-title">Labels</div>
        ${d.labels.map(l => `<span class="label-pill">${escapeHtml(l)}</span>`).join('')}
      </div>`;
    }

    html += `<div class="issue-section">
      <div class="issue-section-title">Description</div>
      <div class="issue-desc">${escapeHtml(d.description)}</div>
    </div>`;

    if (d.subtasks && d.subtasks.length) {
      html += `<div class="issue-section">
        <div class="issue-section-title">Subtasks (${d.subtasks.length})</div>
        ${d.subtasks.map(s => `
          <div class="subtask">
            <a href="${s.url}" target="_blank">${escapeHtml(s.key)}</a>
            <span class="jira-status ${s.status.toLowerCase().replace(/\\s+/g,'')}">${escapeHtml(s.status)}</span>
            <span>${escapeHtml(s.summary)}</span>
          </div>`).join('')}
      </div>`;
    }

    html += `<div class="issue-section">
        <div class="issue-section-title">Comments${d.comments && d.comments.length ? ' (' + d.comments.length + ')' : ''}</div>
        ${(d.comments || []).map(c => renderCommentWithReply(c, d.key)).join('')}
        ${renderAddCommentButton(d.key)}
      </div>`;

    html += `</div>`;
    botContent.innerHTML = html;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
  }

  function looksLikeName(s) {
    // letters, spaces, dots, apostrophes only. 1-4 words. Not starting with /
    return /^[A-Za-z][A-Za-z .'\-]{1,40}$/.test(s.trim()) && s.trim().split(/\s+/).length <= 4;
  }

  async function handleUserSearch(name, botContent) {
    botContent.innerHTML = `<span style="color:#8e8ea0">🔍 Fetching tasks for <b>${escapeHtml(name)}</b> from JIRA...</span>`;
    const res = await fetch(`/jira/user/${encodeURIComponent(name.trim())}`);
    const d = await res.json();
    if (d.error) {
      botContent.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
      return;
    }
    if (!d.issues || d.issues.length === 0) {
      botContent.innerHTML = `
        <div class="jira-header">
          <span>👤 ${escapeHtml(d.user.displayName)} · ${escapeHtml(d.user.email || '')}</span>
        </div>
        <div style="color:#8e8ea0; padding: 10px 0;">No tasks assigned.</div>`;
      return;
    }

    const otherMatchesHtml = (d.other_matches || []).length ? `
      <div style="color: #8e8ea0; font-size: 11px; margin-top: 4px;">
        Other matches: ${d.other_matches.map(u => escapeHtml(u.displayName)).join(', ')}
      </div>` : '';

    let html = `
      <div class="user-header">
        <div class="user-avatar-lg" style="background-image:url('${d.user.avatar}')"></div>
        <div>
          <div class="user-name">${escapeHtml(d.user.displayName)}</div>
          <div class="user-email">${escapeHtml(d.user.email || '')}</div>
          ${otherMatchesHtml}
        </div>
        <div class="user-count">${d.total} <span>tasks</span></div>
      </div>
      <div class="task-list">
    `;

    for (const t of d.issues) {
      const statusClass = t.status.toLowerCase().replace(/\s+/g, '');
      html += `
        <div class="task-row">
          <div class="task-date">${escapeHtml(t.updated)}</div>
          <a href="${t.url}" target="_blank" class="task-key">${escapeHtml(t.key)}</a>
          <div class="task-summary">${escapeHtml(t.summary)}</div>
          <span class="jira-status ${statusClass}">${escapeHtml(t.status)}</span>
        </div>`;
    }
    html += `</div>`;
    botContent.innerHTML = html;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
  }

  async function handleUniversalSearch(query, botContent) {
    botContent.innerHTML = `<span style="color:#8e8ea0">🔍 Searching JIRA for <b>${escapeHtml(query)}</b>...</span>`;
    try {
      const res = await fetch(`/jira/search?q=${encodeURIComponent(query)}`);
      const d = await res.json();
      if (d.error) {
        botContent.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
        return;
      }
      renderUniversalResults(botContent, d, query);
    } catch (err) {
      botContent.innerHTML = `<div class="jira-error">${escapeHtml(err.message)}</div>`;
    }
  }

  function renderUniversalResults(botContent, d, query) {
    const users = d.users || [];
    const issues = d.issues || [];

    if (!users.length && !issues.length) {
      botContent.innerHTML = `
        <div class="search-hint empty-results" style="margin: 0;">
          <span class="big">🔎</span>
          <div>No JIRA results for <b>"${escapeHtml(query)}"</b></div>
          <div style="font-size: 11px; margin-top: 4px;">
            Try: full name, issue key (e.g. A20M-2939), keyword from summary, or type <code>help</code> for commands
          </div>
        </div>`;
      return;
    }

    let html = `
      <div class="jira-header">
        <span>🔍 Results for <b>${escapeHtml(query)}</b></span>
        <span style="color:#8e8ea0; font-size: 11px;">
          ${users.length} user${users.length !== 1 ? 's' : ''} · ${issues.length} issue${issues.length !== 1 ? 's' : ''}
        </span>
      </div>`;

    if (users.length) {
      html += `
        <div class="search-section-title">👤 Matching people <span class="count-badge">${users.length}</span></div>
        <div class="uni-users-grid">`;
      for (const u of users) {
        html += `
          <div class="uni-user-card" onclick="sendPrompt('${escapeAttr(u.displayName)}')">
            <div class="avatar-sm" style="background-image:url('${u.avatar || ''}')"></div>
            <div style="flex:1; min-width:0;">
              <div class="uni-user-name">${escapeHtml(u.displayName)}</div>
              <div class="uni-user-email">${escapeHtml(u.email || 'No email')}</div>
            </div>
            <span class="action-cue">View tasks →</span>
          </div>`;
      }
      html += `</div>`;
    }

    if (issues.length) {
      html += `
        <div class="search-section-title" style="margin-top: 18px;">
          🎫 Matching issues <span class="count-badge">${issues.length}</span>
        </div>
        <div class="uni-issues-list">`;
      for (const i of issues) {
        const statusClass = i.status.toLowerCase().replace(/\s+/g, '');
        html += `
          <div class="uni-issue-card" onclick="sendPrompt('${escapeAttr(i.key)}')">
            <div class="uni-issue-top">
              <span class="jira-key" style="font-family:'SF Mono',Menlo,monospace; font-weight:600;">${escapeHtml(i.key)}</span>
              <span class="jira-status ${statusClass}">${escapeHtml(i.status)}</span>
              <span class="jira-type">${escapeHtml(i.type)}</span>
              <span style="margin-left:auto; color:#8e8ea0; font-size:11px;">${escapeHtml(i.updated || '')}</span>
            </div>
            <div class="uni-issue-summary">${escapeHtml(i.summary)}</div>
            <div class="uni-issue-meta">
              👤 ${escapeHtml(i.assignee)} · ⚡ ${escapeHtml(i.priority)}
            </div>
          </div>`;
      }
      html += `</div>`;
    }

    html += `
      <div style="font-size: 11px; color: #8e8ea0; margin-top: 10px; text-align: center;">
        💡 Click a person to see their tasks, or a ticket to see full details
      </div>`;

    botContent.innerHTML = html;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
  }

  function handleHelp(botContent) {
    botContent.innerHTML = `
      <div style="line-height: 1.7;">
        <div style="margin-bottom: 10px;">I only handle JIRA queries. Try:</div>
        <div style="display: grid; gap: 6px;">
          <div>📊 <code>dashboard</code> — full analysis</div>
          <div>📅 <code>today</code> — today's tasks</div>
          <div>📆 <code>week</code> — this week's tasks</div>
          <div>👤 <code>mine</code> — all your assigned tasks</div>
          <div>🎯 <code>A20M-2939</code> — full issue details</div>
          <div>🧑 <code>Kapil</code> / <code>Pratibha</code> — anyone's tasks (type a name)</div>
          <div>🔍 <code>/jira MREC open</code> — project filter</div>
        </div>
      </div>`;
  }

  async function handleJira(cmd, botContent) {
    botContent.innerHTML = '<span style="color:#8e8ea0">🔍 Fetching tasks from JIRA...</span>';
    const res = await fetch('/jira', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cmd })
    });
    const data = await res.json();

    if (data.error) {
      botContent.innerHTML = `<div class="jira-error">${escapeHtml(data.error)}</div>`;
      return;
    }

    if (!data.issues || data.issues.length === 0) {
      botContent.innerHTML = `
        <div class="jira-header">
          <span>JQL:</span>
          <span class="jql-pill">${escapeHtml(data.jql || '')}</span>
        </div>
        <div style="color:#8e8ea0">No tasks found.</div>`;
      return;
    }

    let html = `
      <div class="jira-header">
        <span>${data.total} tasks · JQL:</span>
        <span class="jql-pill">${escapeHtml(data.jql)}</span>
      </div>`;

    for (const iss of data.issues) {
      const statusClass = iss.status.toLowerCase().replace(/\s+/g, '');
      html += `
        <div class="jira-card">
          <div class="jira-card-top">
            <a href="${iss.url}" target="_blank" class="jira-key">${escapeHtml(iss.key)}</a>
            <span class="jira-status ${statusClass}">${escapeHtml(iss.status)}</span>
            <span class="jira-type">${escapeHtml(iss.type)}</span>
            <span class="jira-priority">${escapeHtml(iss.priority)}</span>
          </div>
          <div class="jira-summary">${escapeHtml(iss.summary)}</div>
          <div class="jira-meta">
            <span>👤 ${escapeHtml(iss.assignee)}</span>
            <span>📅 ${escapeHtml(iss.updated)}</span>
          </div>
        </div>`;
    }
    botContent.innerHTML = html;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ══════ SETTINGS MODAL ══════
  let tokenAlreadySaved = false;

  async function openSettings() {
    const res = await fetch('/settings');
    const cfg = await res.json();
    document.getElementById('s-domain').value = cfg.domain || '';
    document.getElementById('s-email').value = cfg.email || '';
    document.getElementById('s-token').value = '';
    tokenAlreadySaved = !!cfg.token_set;
    document.getElementById('s-token').placeholder = cfg.token_hint
      ? cfg.token_hint + ' (leave blank to keep)'
      : 'ATATT3xFf... (paste your API token)';
    const tokenHint = document.getElementById('token-hint');
    if (tokenHint) {
      tokenHint.textContent = cfg.token_set
        ? '✓ Token already saved. Leave blank to keep, or paste new one to update.'
        : 'Saved to local .env file (chmod 600)';
    }
    const status = document.getElementById('current-status');
    if (cfg.configured) {
      status.innerHTML = '<span class="status-badge connected"><span class="status-dot"></span>Configured</span>';
    } else {
      status.innerHTML = '<span class="status-badge disconnected"><span class="status-dot"></span>Not configured</span>';
    }
    document.getElementById('save-status').style.display = 'none';
    document.getElementById('settings-modal').classList.add('open');
  }

  function closeSettings() {
    document.getElementById('settings-modal').classList.remove('open');
  }

  async function saveSettings() {
    const domain = document.getElementById('s-domain').value.trim();
    const email = document.getElementById('s-email').value.trim();
    const token = document.getElementById('s-token').value.trim();
    const statusEl = document.getElementById('save-status');
    const btn = document.getElementById('save-btn');

    if (!domain || !email) {
      statusEl.className = 'save-status error';
      statusEl.textContent = 'Domain and email are required';
      statusEl.style.display = 'block';
      return;
    }
    if (!token && !tokenAlreadySaved) {
      statusEl.className = 'save-status error';
      statusEl.textContent = 'Token required (first-time setup)';
      statusEl.style.display = 'block';
      return;
    }

    btn.disabled = true;
    btn.textContent = 'Testing...';
    statusEl.className = 'save-status';
    statusEl.textContent = '';

    try {
      const res = await fetch('/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain, email, token })
      });
      const data = await res.json();
      if (data.ok) {
        statusEl.className = 'save-status success';
        statusEl.textContent = '✓ ' + data.message;
        updateJiraBadge(true);
        setTimeout(() => closeSettings(), 1200);
      } else {
        statusEl.className = 'save-status error';
        statusEl.textContent = '✗ ' + (data.error || 'Save failed');
      }
    } catch (err) {
      statusEl.className = 'save-status error';
      statusEl.textContent = '✗ ' + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Save & Test';
    }
  }

  function updateJiraBadge(connected) {
    const el = document.getElementById('jira-status-badge');
    el.classList.toggle('on', !!connected);
    el.innerHTML = `<span class="s-dot"></span><span>JIRA ${connected ? 'on' : 'off'}</span>`;
  }

  // Check JIRA status on load — badge reflects ACTUAL reachability, not just token presence
  fetch('/settings').then(r => r.json()).then(cfg => {
    const on = cfg.configured && cfg.reachable;
    updateJiraBadge(on);
    if (cfg.configured && !cfg.reachable) {
      console.warn('JIRA token invalid:', cfg.reachable_error);
    }
  });

  // ══════ TEAMS ══════
  let currentTeams = [];
  let managingTeam = null;

  async function loadTeams() {
    const res = await fetch('/teams');
    const data = await res.json();
    currentTeams = data.teams || [];
    renderTeamsList();
    // Fetch quick status for badges
    for (const t of currentTeams) {
      fetchTeamBadge(t.name);
    }
    return currentTeams;
  }

  function renderTeamsList() {
    const el = document.getElementById('teams-list');
    if (!currentTeams.length) {
      el.innerHTML = '<div style="color:#8e8ea0; font-size:12px; padding: 6px 12px;">No teams yet</div>';
      return;
    }
    el.innerHTML = currentTeams.map(t => `
      <div class="team-item" onclick="openManageTeamModal('${escapeAttr(t.name)}')">
        <span>👥</span>
        <span class="team-name">${escapeHtml(t.name)}</span>
        <span id="team-badge-${escapeAttr(t.name)}"></span>
      </div>`).join('');
  }

  async function fetchTeamBadge(teamName) {
    try {
      const res = await fetch(`/teams/${encodeURIComponent(teamName)}/status`);
      if (!res.ok) return;
      const d = await res.json();
      const el = document.getElementById('team-badge-' + escapeAttr(teamName));
      if (!el) return;
      const parts = [];
      if (d.idle_count > 0) parts.push(`<span class="team-idle">${d.idle_count} idle</span>`);
      parts.push(`<span class="team-count">${d.workload.length}</span>`);
      el.innerHTML = parts.join(' ');
    } catch (e) {}
  }

  function escapeAttr(s) {
    return s.replace(/'/g, "\\'").replace(/"/g, '&quot;');
  }

  function openNewTeamModal() {
    document.getElementById('new-team-name').value = '';
    document.getElementById('new-team-modal').classList.add('open');
    setTimeout(() => document.getElementById('new-team-name').focus(), 100);
  }
  function closeNewTeamModal() {
    document.getElementById('new-team-modal').classList.remove('open');
  }

  async function submitNewTeam() {
    const name = document.getElementById('new-team-name').value.trim();
    if (!name) return;
    const res = await fetch('/teams', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    closeNewTeamModal();
    await loadTeams();
    openManageTeamModal(name);
  }

  async function openManageTeamModal(teamName) {
    managingTeam = teamName;
    document.getElementById('manage-team-title').textContent = '👥 ' + teamName;
    document.getElementById('team-stats-line').textContent = 'Loading workload...';
    document.getElementById('member-search').value = '';
    onMemberSearch('');  // show initial hint
    document.getElementById('current-members').innerHTML = '<div style="color:#8e8ea0; font-size:12px; padding: 8px;">Loading members...</div>';
    document.getElementById('manage-team-modal').classList.add('open');
    await refreshTeamView();
  }
  function closeManageTeamModal() {
    document.getElementById('manage-team-modal').classList.remove('open');
    managingTeam = null;
  }

  async function refreshTeamView() {
    // Fetch team + workload in one shot
    const res = await fetch(`/teams/${encodeURIComponent(managingTeam)}/status`);
    const listEl = document.getElementById('current-members');
    const countEl = document.getElementById('current-count');
    const statsLine = document.getElementById('team-stats-line');

    if (!res.ok) {
      const err = await res.json();
      listEl.innerHTML = `<div class="jira-error">${escapeHtml(err.error || 'Failed to load')}</div>`;
      return;
    }
    const d = await res.json();
    const members = d.team.members || [];
    countEl.textContent = members.length;

    if (!members.length) {
      statsLine.textContent = 'No members yet — search and add above';
      listEl.innerHTML = '<div style="color:#8e8ea0; font-size:12px; padding: 12px; text-align:center;">No members yet. Add someone above.</div>';
      return;
    }

    statsLine.innerHTML = `<span style="color:#FF6B35">${d.busy_count} busy</span> · <span style="color:#f5a623">${d.idle_count} idle</span> · ${members.length} total`;

    // Build workload map by accountId
    const workloadMap = {};
    for (const w of d.workload) {
      workloadMap[w.member.accountId] = w;
    }

    listEl.innerHTML = members.map(m => {
      const w = workloadMap[m.accountId] || { open_count: 0, high_priority_count: 0, overdue_count: 0 };
      const idle = w.open_count === 0;
      const highLoad = w.open_count > 20;
      const pillClass = idle ? 'idle' : (highLoad ? 'busy-high' : '');
      const flags = [];
      if (w.high_priority_count > 0) flags.push(`🔥${w.high_priority_count}`);
      if (w.overdue_count > 0) flags.push(`⏰${w.overdue_count}`);
      return `
        <div class="tm-member ${idle ? 'idle' : ''}" id="tm-mem-${escapeAttr(m.accountId)}">
          <div class="tm-member-row" onclick="toggleMemberDetails('${escapeAttr(m.accountId)}')">
            <div class="avatar-sm" style="background-image:url('${m.avatar || ''}')"></div>
            <div>
              <div class="m-name">${escapeHtml(m.displayName)}</div>
              <div class="m-email">${escapeHtml(m.email || '')}${flags.length ? ' · ' + flags.join(' ') : ''}</div>
            </div>
            <div class="tm-workload-pill ${pillClass}">${w.open_count} open</div>
            <span class="tm-expand-arrow">▶</span>
            <button class="tm-remove-btn" onclick="event.stopPropagation(); removeMember('${escapeAttr(m.accountId)}')" title="Remove from team">×</button>
          </div>
          <div class="tm-member-details" id="tm-det-${escapeAttr(m.accountId)}"></div>
        </div>`;
    }).join('');
  }

  async function toggleMemberDetails(accountId) {
    const memEl = document.getElementById('tm-mem-' + accountId);
    const detEl = document.getElementById('tm-det-' + accountId);
    if (!memEl) return;

    // Close others
    document.querySelectorAll('.tm-member.expanded').forEach(el => {
      if (el.id !== 'tm-mem-' + accountId) el.classList.remove('expanded');
    });

    if (memEl.classList.contains('expanded')) {
      memEl.classList.remove('expanded');
      return;
    }
    memEl.classList.add('expanded');
    detEl.innerHTML = '<div style="color:#8e8ea0; padding: 12px; text-align:center;">Loading tasks...</div>';

    try {
      const res = await fetch(`/jira/member/${encodeURIComponent(accountId)}`);
      const d = await res.json();
      if (d.error) {
        detEl.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
        return;
      }
      renderMemberDetails(detEl, d, accountId);
    } catch (err) {
      detEl.innerHTML = `<div class="jira-error">${escapeHtml(err.message)}</div>`;
    }
  }

  function renderMemberDetails(el, d, accountId) {
    const todaySection = d.has_today_work
      ? `
        <div class="tm-details-section">
          <h5>📅 Today's work <span class="count-badge">${d.today_tasks.length}</span></h5>
          ${d.today_tasks.map(t => renderTinyTaskRow(t, accountId)).join('')}
        </div>`
      : `
        <div class="tm-details-section">
          <h5>📅 Today's work</h5>
          <div class="tm-no-today">⚠️ No work updated today</div>
        </div>`;

    const openSection = d.open_tasks.length
      ? `
        <div class="tm-details-section">
          <h5>📋 All open tasks <span class="count-badge">${d.open_count}</span></h5>
          ${d.open_tasks.slice(0, 10).map(t => renderTinyTaskRow(t, accountId)).join('')}
          ${d.open_count > 10 ? `<div style="color:#8e8ea0; font-size:11px; text-align:center; margin-top:6px;">+ ${d.open_count - 10} more</div>` : ''}
        </div>`
      : `
        <div class="tm-details-section">
          <h5>📋 All open tasks</h5>
          <div class="tm-no-today" style="background: rgba(16,163,127,0.1); border-color: rgba(16,163,127,0.3); color: #7ade9f;">
            ✓ No open tasks. Fully caught up!
          </div>
        </div>`;

    el.innerHTML = todaySection + openSection;
  }

  function renderTinyTaskRow(t, accountId) {
    const summary = (t.summary || '').replace(/"/g, '&quot;');
    const aid = accountId || '';
    return `
      <div class="tm-task-row">
        <div class="t-date">${escapeHtml(t.updated || '')}</div>
        <a href="${t.url}" target="_blank" class="t-key">${escapeHtml(t.key)}</a>
        <div class="t-sum">${escapeHtml(t.summary || '')}</div>
        <div class="t-status">${escapeHtml(t.status || '')}</div>
        <button class="tm-log-btn" onclick="openWorklogModal('${escapeAttr(t.key)}', '${escapeAttr(summary)}', '${escapeAttr(aid)}')" title="Log work">⏱ Log</button>
      </div>`;
  }

  let searchDebounce = null;

  function clearMemberSearch() {
    document.getElementById('member-search').value = '';
    onMemberSearch('');
    document.getElementById('member-search').focus();
  }

  function setSearchSpinner(visible) {
    document.getElementById('member-search-spinner').classList.toggle('visible', visible);
  }
  function setSearchClear(visible) {
    document.getElementById('member-search-clear').classList.toggle('visible', visible);
  }

  function renderSearchHint(html, cssClass) {
    document.getElementById('member-search-results').innerHTML =
      `<div class="search-hint ${cssClass || ''}">${html}</div>`;
  }

  function onMemberSearch(q) {
    clearTimeout(searchDebounce);
    const trimmed = q.trim();
    const resultsEl = document.getElementById('member-search-results');
    setSearchClear(trimmed.length > 0);

    // Empty input → helpful onboarding hint
    if (trimmed.length === 0) {
      resultsEl.innerHTML = `
        <div class="search-hint">
          <span>💡</span>
          <span>Start typing a name or email to search JIRA users. Click a result to add them to <b>${escapeHtml(managingTeam || 'the team')}</b>.</span>
        </div>`;
      setSearchSpinner(false);
      return;
    }

    // Too short → tell user why nothing is happening
    if (trimmed.length < 2) {
      resultsEl.innerHTML = `
        <div class="search-hint warn">
          <span>⌨️</span>
          <span>Type at least 2 characters to search.</span>
        </div>`;
      setSearchSpinner(false);
      return;
    }

    setSearchSpinner(true);
    searchDebounce = setTimeout(async () => {
      try {
        const res = await fetch(`/teams/${encodeURIComponent(managingTeam)}/search-users?q=${encodeURIComponent(trimmed)}`);
        const data = await res.json();
        setSearchSpinner(false);

        if (!data.users || !data.users.length) {
          resultsEl.innerHTML = `
            <div class="search-hint empty-results">
              <span class="big">🔎</span>
              <div>No users found for <b>"${escapeHtml(trimmed)}"</b></div>
              <div style="font-size: 11px; margin-top: 4px;">Try a full name, partial name, or email.</div>
            </div>`;
          return;
        }

        // Get currently added accountIds to mark them
        const teamRes = await fetch('/teams');
        const teamsData = await teamRes.json();
        const currentTeam = (teamsData.teams || []).find(t => t.name === managingTeam);
        const existingIds = new Set((currentTeam?.members || []).map(m => m.accountId));

        const addable = data.users.filter(u => !existingIds.has(u.accountId));
        const already = data.users.filter(u => existingIds.has(u.accountId));

        let html = `<div class="search-results-header">
          <span>${data.users.length} match${data.users.length > 1 ? 'es' : ''} found</span>
          <span>Click to add</span>
        </div><div class="search-results">`;

        for (const u of addable) {
          html += `
            <div class="user-search-result" onclick='addMember(${JSON.stringify(u).replace(/"/g, "&quot;")})'>
              <div class="avatar-sm" style="background-image:url('${u.avatar || ''}')"></div>
              <div class="name-block">
                <div class="n">${escapeHtml(u.displayName)}</div>
                <div class="e">${escapeHtml(u.email || 'No email')}</div>
              </div>
              <span class="action-cue">+ Add</span>
            </div>`;
        }
        for (const u of already) {
          html += `
            <div class="user-search-result already-added" title="Already in this team">
              <div class="avatar-sm" style="background-image:url('${u.avatar || ''}')"></div>
              <div class="name-block">
                <div class="n">${escapeHtml(u.displayName)}</div>
                <div class="e">${escapeHtml(u.email || 'No email')}</div>
              </div>
              <span class="action-cue">✓ In team</span>
            </div>`;
        }
        html += '</div>';

        resultsEl.innerHTML = html;
      } catch (err) {
        setSearchSpinner(false);
        resultsEl.innerHTML = `
          <div class="search-hint warn">
            <span>⚠️</span>
            <span>Search failed: ${escapeHtml(err.message)}</span>
          </div>`;
      }
    }, 300);
  }

  async function addMember(memberJsonStr) {
    let member = memberJsonStr;
    if (typeof memberJsonStr === 'string') {
      try { member = JSON.parse(memberJsonStr); } catch (e) { return; }
    }
    const res = await fetch(`/teams/${encodeURIComponent(managingTeam)}/members`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(member)
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    document.getElementById('member-search').value = '';
    onMemberSearch('');
    await refreshTeamView();
  }

  async function removeMember(accountId) {
    if (!confirm('Remove this member from team?')) return;
    await fetch(`/teams/${encodeURIComponent(managingTeam)}/members/${encodeURIComponent(accountId)}`, {
      method: 'DELETE'
    });
    await refreshTeamView();
  }

  async function deleteCurrentTeam() {
    if (!confirm(`Delete team "${managingTeam}"?`)) return;
    await fetch(`/teams/${encodeURIComponent(managingTeam)}`, { method: 'DELETE' });
    closeManageTeamModal();
    await loadTeams();
  }

  async function viewTeam(teamName) {
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();
    const chat = document.getElementById('chat');
    if (!chat.querySelector('.chat-inner')) {
      chat.innerHTML = '<div class="chat-inner"></div>';
    }
    const inner = chat.querySelector('.chat-inner');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';
    msgDiv.innerHTML = `
      <div class="msg-avatar bot-avatar">M</div>
      <div class="msg-body">
        <div class="msg-role">MonoceptGPT</div>
        <div class="msg-content"><span style="color:#8e8ea0">👥 Loading ${escapeHtml(teamName)} status...</span></div>
      </div>`;
    inner.appendChild(msgDiv);
    const botContent = msgDiv.querySelector('.msg-content');
    chat.scrollTop = chat.scrollHeight;

    try {
      const res = await fetch(`/teams/${encodeURIComponent(teamName)}/status`);
      const d = await res.json();
      if (d.error) {
        botContent.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
        return;
      }
      renderTeamView(botContent, d);
    } catch (err) {
      botContent.textContent = 'Error: ' + err.message;
    }
  }

  function renderTeamView(botContent, d) {
    const idleMembers = d.workload.filter(w => w.open_count === 0);
    let html = `
      <div class="team-view-header">
        <div>
          <div class="team-view-title">👥 ${escapeHtml(d.team.name)}</div>
          <div class="team-view-subtitle">${d.workload.length} members · Open task workload</div>
        </div>
        <div class="team-view-stats">
          <div class="team-stat"><div class="n total">${d.workload.length}</div><div class="l">Total</div></div>
          <div class="team-stat"><div class="n busy">${d.busy_count}</div><div class="l">Busy</div></div>
          <div class="team-stat"><div class="n idle">${d.idle_count}</div><div class="l">Idle</div></div>
        </div>
      </div>
    `;

    if (idleMembers.length > 0) {
      html += `
        <div class="morning-briefing">
          <h4>⚠️ ${idleMembers.length} member${idleMembers.length > 1 ? 's' : ''} idle — no open tasks</h4>
          <div class="briefing-body">Assign work or check status:</div>
          <div class="briefing-idle-list">
            ${idleMembers.map(m => `<span class="pill">${escapeHtml(m.member.displayName)}</span>`).join('')}
          </div>
        </div>`;
    }

    for (const w of d.workload) {
      const isIdle = w.open_count === 0;
      const latest = w.latest_task;
      html += `
        <div class="member-row ${isIdle ? 'idle' : ''}">
          <div class="member-avatar" style="background-image:url('${w.member.avatar || ''}')"></div>
          <div class="member-info">
            <div class="name">${escapeHtml(w.member.displayName)}</div>
            <div class="latest">${latest
              ? '📌 ' + escapeHtml(latest.key) + ' · ' + escapeHtml(latest.summary)
              : (isIdle ? 'No open tasks' : '')}</div>
          </div>
          <div class="member-workload">
            <div class="count ${isIdle ? 'idle' : ''}">${w.open_count}</div>
            <div class="label">open</div>
          </div>
          <div class="member-flags">
            ${w.high_priority_count > 0 ? `<span class="flag-hi">🔥 ${w.high_priority_count} hi</span>` : ''}
            ${w.overdue_count > 0 ? `<span class="flag-od">⏰ ${w.overdue_count} od</span>` : ''}
          </div>
          ${isIdle ? '<span class="idle-badge">Idle</span>' : '<span></span>'}
        </div>`;
    }
    botContent.innerHTML = html;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
  }

  // ══════ COMMENT REPLY / ADD ══════
  let replyCounter = 0;  // unique IDs for reply boxes

  function renderCommentWithReply(c, issueKey) {
    const boxId = 'reply-' + (++replyCounter);
    const prefill = c.author ? `@${c.author}, ` : '';
    return `
      <div class="comment">
        <div class="comment-head">
          <span>${escapeHtml(c.author || 'Unknown')} · ${escapeHtml(c.date || '')}</span>
          <button class="reply-btn" onclick="toggleReplyBox('${boxId}', '${escapeAttr(issueKey)}', ${JSON.stringify(prefill).replace(/"/g, '&quot;')})">↩ Reply</button>
        </div>
        <div class="comment-body">${escapeHtml(c.body || '(empty)')}</div>
        <div class="reply-box" id="${boxId}">
          <textarea placeholder="Write your reply..."></textarea>
          <div class="status"></div>
          <div class="actions">
            <button class="btn btn-secondary" onclick="toggleReplyBox('${boxId}')">Cancel</button>
            <button class="btn btn-primary" onclick="submitReply('${boxId}', '${escapeAttr(issueKey)}')">Post Reply</button>
          </div>
        </div>
      </div>`;
  }

  function renderAddCommentButton(issueKey) {
    const boxId = 'reply-' + (++replyCounter);
    return `
      <button class="add-comment-btn" onclick="toggleReplyBox('${boxId}', '${escapeAttr(issueKey)}', '')">+ Add Comment</button>
      <div class="reply-box" id="${boxId}" style="margin-top: 4px;">
        <textarea placeholder="Write a new comment..."></textarea>
        <div class="status"></div>
        <div class="actions">
          <button class="btn btn-secondary" onclick="toggleReplyBox('${boxId}')">Cancel</button>
          <button class="btn btn-primary" onclick="submitReply('${boxId}', '${escapeAttr(issueKey)}')">Post Comment</button>
        </div>
      </div>`;
  }

  function toggleReplyBox(boxId, issueKey, prefill) {
    const box = document.getElementById(boxId);
    if (!box) return;
    const isOpen = box.classList.contains('open');
    if (isOpen) {
      box.classList.remove('open');
      return;
    }
    box.classList.add('open');
    const ta = box.querySelector('textarea');
    if (prefill && !ta.value) ta.value = prefill;
    setTimeout(() => {
      ta.focus();
      const len = ta.value.length;
      ta.setSelectionRange(len, len);
    }, 50);
  }

  async function submitReply(boxId, issueKey) {
    const box = document.getElementById(boxId);
    if (!box) return;
    const ta = box.querySelector('textarea');
    const status = box.querySelector('.status');
    const btn = box.querySelector('.btn-primary');
    const body = (ta.value || '').trim();
    if (!body) {
      status.textContent = 'Please write something';
      status.classList.remove('ok');
      return;
    }
    btn.disabled = true;
    btn.textContent = 'Posting...';
    status.textContent = '';
    try {
      const res = await fetch(`/jira/comment/${encodeURIComponent(issueKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ body })
      });
      const d = await res.json();
      if (d.ok) {
        status.classList.add('ok');
        status.textContent = '✓ Comment posted';
        ta.value = '';
        setTimeout(() => {
          box.classList.remove('open');
          status.textContent = '';
          status.classList.remove('ok');
        }, 1200);
      } else {
        status.classList.remove('ok');
        status.textContent = '✗ ' + (d.error || 'Post failed');
      }
    } catch (err) {
      status.classList.remove('ok');
      status.textContent = '✗ ' + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = boxId.startsWith('reply-') ? 'Post Reply' : 'Post Comment';
    }
  }

  // ══════ PRODUCTION ISSUES ══════
  async function showProductionIssues() {
    // Open as a new chat message
    ensureActiveChat('Production Issues');
    pushMessage('user', { text: '🚨 Production Issues' });
    const botIdx = pushMessage('bot', { html: '' });
    renderChatHistory();

    addMessage('user', '🚨 Production Issues');
    const botContent = addMessage('bot', '');
    botContent.innerHTML = '<span style="color:#8e8ea0">🚨 Loading production issues from JIRA dashboard...</span>';

    try {
      const res = await fetch('/jira/production');
      const d = await res.json();
      if (d.error) {
        botContent.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
        return;
      }
      renderProductionIssues(botContent, d);
      updateProdCount(d.total);
    } catch (err) {
      botContent.innerHTML = `<div class="jira-error">${escapeHtml(err.message)}</div>`;
    } finally {
      updateBotMessage(botIdx, botContent.innerHTML);
    }
  }

  function updateProdCount(n) {
    const el = document.getElementById('prod-count');
    if (!el) return;
    if (n > 0) {
      el.textContent = n;
      el.classList.add('visible');
    } else {
      el.classList.remove('visible');
    }
  }

  function renderProductionIssues(botContent, d) {
    let html = `
      <div class="prod-header">
        <div style="flex: 1; min-width: 0;">
          <div class="p-title">🚨 ${escapeHtml(d.filter_name)}</div>
          <div class="p-sub">Live from dashboard 17442 · gadget 75070</div>
          <div class="p-jql">${escapeHtml(d.jql)}</div>
          <a href="${d.dashboard_url}" target="_blank" class="prod-open-link">🔗 Open full dashboard in JIRA →</a>
        </div>
        <div class="p-count">
          <div class="n">${d.total}</div>
          <div class="l">Issues</div>
        </div>
      </div>`;

    if (d.by_status && d.by_status.length) {
      html += `
        <div class="prod-stats">
          <div class="prod-stat-block">
            <h4>🚦 By Status</h4>
            ${d.by_status.map(([k,v]) => `<div class="row"><span>${escapeHtml(k)}</span><span class="n">${v}</span></div>`).join('')}
          </div>
          <div class="prod-stat-block">
            <h4>⚡ By Severity</h4>
            ${d.by_severity.map(([k,v]) => `<div class="row"><span>${escapeHtml(k)}</span><span class="n">${v}</span></div>`).join('')}
          </div>
        </div>`;
    }

    if (!d.issues || !d.issues.length) {
      html += `<div style="color:#8e8ea0; padding: 10px; text-align:center;">No production issues found ✨</div>`;
    } else {
      html += `<div style="margin-bottom: 6px; color:#8e8ea0; font-size: 12px;">Latest ${d.issues.length} issues (click any row for full description)</div>`;
      for (const i of d.issues) {
        const statusClass = i.status.toLowerCase().replace(/\s+/g, '');
        html += `
          <div class="prod-issue-wrap" id="prod-wrap-${escapeAttr(i.key)}">
            <div class="prod-issue-row" onclick="toggleProdIssue('${escapeAttr(i.key)}')">
              <span class="p-arrow">▶</span>
              <div class="p-date">${escapeHtml(i.updated || '')}</div>
              <a href="${i.url}" target="_blank" class="p-key" onclick="event.stopPropagation();">${escapeHtml(i.key)}</a>
              <div class="p-sum">${escapeHtml(i.summary || '')}</div>
              <div class="p-assignee">👤 ${escapeHtml(i.assignee)}</div>
              <span class="jira-status ${statusClass}">${escapeHtml(i.status)}</span>
            </div>
            <div class="prod-issue-details" id="prod-det-${escapeAttr(i.key)}"></div>
          </div>`;
      }
    }

    botContent.innerHTML = html;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
  }

  async function toggleProdIssue(key) {
    const wrap = document.getElementById('prod-wrap-' + key);
    const det = document.getElementById('prod-det-' + key);
    if (!wrap || !det) return;

    if (wrap.classList.contains('expanded')) {
      wrap.classList.remove('expanded');
      return;
    }
    wrap.classList.add('expanded');

    if (det.dataset.loaded === 'true') return;

    det.innerHTML = '<div style="color:#8e8ea0; padding: 12px; text-align:center;">Loading full details...</div>';
    try {
      const res = await fetch(`/jira/issue/${encodeURIComponent(key)}`);
      const d = await res.json();
      if (d.error) {
        det.innerHTML = `<div class="jira-error">${escapeHtml(d.error)}</div>`;
        return;
      }

      let html = `
        <div class="prod-detail-section">
          <h5>📝 Description</h5>
          <div class="prod-desc">${escapeHtml(d.description || '(no description)')}</div>
        </div>
        <div class="prod-detail-section">
          <h5>ℹ Details</h5>
          <div class="prod-meta">
            <div><span class="k">Assignee:</span> <span class="v">${escapeHtml(d.assignee)}</span></div>
            <div><span class="k">Reporter:</span> <span class="v">${escapeHtml(d.reporter)}</span></div>
            <div><span class="k">Created:</span> <span class="v">${escapeHtml(d.created)}</span></div>
            <div><span class="k">Updated:</span> <span class="v">${escapeHtml(d.updated)}</span></div>
            <div><span class="k">Priority:</span> <span class="v">${escapeHtml(d.priority)}</span></div>
            <div><span class="k">Due:</span> <span class="v">${escapeHtml(d.duedate)}</span></div>
          </div>
        </div>`;

      if (d.labels && d.labels.length) {
        html += `<div class="prod-detail-section">
          <h5>🏷 Labels</h5>
          <div>${d.labels.map(l => `<span class="label-pill">${escapeHtml(l)}</span>`).join('')}</div>
        </div>`;
      }

      if (d.subtasks && d.subtasks.length) {
        html += `<div class="prod-detail-section">
          <h5>📋 Subtasks (${d.subtasks.length})</h5>
          ${d.subtasks.map(s => `
            <div class="subtask">
              <a href="${s.url}" target="_blank">${escapeHtml(s.key)}</a>
              <span class="jira-status ${s.status.toLowerCase().replace(/\s+/g,'')}">${escapeHtml(s.status)}</span>
              <span>${escapeHtml(s.summary)}</span>
            </div>`).join('')}
        </div>`;
      }

      html += `<div class="prod-detail-section">
          <h5>💬 Comments${d.comments && d.comments.length ? ' (' + d.comments.length + ')' : ''}</h5>
          ${(d.comments || []).map(c => renderCommentWithReply(c, d.key)).join('')}
          ${renderAddCommentButton(d.key)}
        </div>`;

      const summaryEsc = (d.summary || '').replace(/"/g, '&quot;');
      html += `
        <div class="prod-actions">
          <a href="${d.url}" target="_blank" class="btn btn-secondary">🔗 Open in JIRA</a>
          <button class="btn btn-primary" onclick="openWorklogModal('${escapeAttr(d.key)}', '${escapeAttr(summaryEsc)}')">⏱ Log Work</button>
        </div>`;

      det.innerHTML = html;
      det.dataset.loaded = 'true';
    } catch (err) {
      det.innerHTML = `<div class="jira-error">${escapeHtml(err.message)}</div>`;
    }
  }

  // Refresh production issue count in sidebar on boot
  async function refreshProdCount() {
    try {
      const res = await fetch('/jira/production');
      if (!res.ok) return;
      const d = await res.json();
      updateProdCount(d.total || 0);
    } catch (e) {}
  }

  // ══════ WORKLOG ══════
  let worklogIssueKey = null;
  let worklogAccountId = null;

  function openWorklogModal(issueKey, summary, accountId) {
    worklogIssueKey = issueKey;
    worklogAccountId = accountId || null;
    document.getElementById('worklog-issue-title').textContent = issueKey + ' — ' + (summary || '');
    document.getElementById('worklog-time').value = '';
    document.getElementById('worklog-comment').value = '';
    document.getElementById('worklog-status').style.display = 'none';
    document.getElementById('worklog-submit-btn').disabled = false;
    document.getElementById('worklog-submit-btn').textContent = 'Log Work';
    document.getElementById('worklog-modal').classList.add('open');
    setTimeout(() => document.getElementById('worklog-time').focus(), 100);
  }

  function closeWorklogModal() {
    document.getElementById('worklog-modal').classList.remove('open');
    worklogIssueKey = null;
  }

  function setWorklogTime(t) {
    document.getElementById('worklog-time').value = t;
    document.getElementById('worklog-comment').focus();
  }

  async function submitWorklog() {
    const time = document.getElementById('worklog-time').value.trim();
    const comment = document.getElementById('worklog-comment').value.trim();
    const statusEl = document.getElementById('worklog-status');
    const btn = document.getElementById('worklog-submit-btn');

    if (!time) {
      statusEl.className = 'save-status error';
      statusEl.textContent = 'Time is required (e.g. 30m, 1h, 2h 30m)';
      statusEl.style.display = 'block';
      return;
    }

    btn.disabled = true;
    btn.textContent = 'Logging...';
    statusEl.style.display = 'none';

    try {
      const res = await fetch(`/jira/worklog/${encodeURIComponent(worklogIssueKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ time, comment })
      });
      const data = await res.json();
      if (data.ok) {
        statusEl.className = 'save-status success';
        statusEl.textContent = `✓ Logged ${data.timeSpent} on ${worklogIssueKey}`;
        statusEl.style.display = 'block';
        setTimeout(() => {
          closeWorklogModal();
          // Refresh member details if open
          if (worklogAccountId) {
            const memEl = document.getElementById('tm-mem-' + worklogAccountId);
            if (memEl && memEl.classList.contains('expanded')) {
              memEl.classList.remove('expanded');
              toggleMemberDetails(worklogAccountId);
            }
          }
        }, 900);
      } else {
        statusEl.className = 'save-status error';
        statusEl.textContent = '✗ ' + (data.error || 'Log failed');
        statusEl.style.display = 'block';
      }
    } catch (err) {
      statusEl.className = 'save-status error';
      statusEl.textContent = '✗ ' + err.message;
      statusEl.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Log Work';
    }
  }

  // Morning briefing on welcome page — show idle counts across teams
  async function showMorningBriefing() {
    if (!currentTeams.length) return;
    const welcome = document.getElementById('welcome');
    if (!welcome) return;

    const briefings = [];
    for (const t of currentTeams) {
      try {
        const res = await fetch(`/teams/${encodeURIComponent(t.name)}/status`);
        if (!res.ok) continue;
        const d = await res.json();
        if (d.idle_count > 0) {
          briefings.push({
            team: t.name,
            idle: d.workload.filter(w => w.open_count === 0),
          });
        }
      } catch (e) {}
    }
    if (!briefings.length) return;

    const container = document.createElement('div');
    container.className = 'morning-briefing';
    container.innerHTML = `
      <h4>☀️ Morning Briefing — idle team members</h4>
      <div class="briefing-body">
        ${briefings.map(b => `
          <div style="margin-top: 8px;">
            <strong style="color: #ffcc66;">${escapeHtml(b.team)}</strong>
            <span style="color:#8e8ea0"> · ${b.idle.length} idle</span>
            <div class="briefing-idle-list">
              ${b.idle.map(w => `<span class="pill" onclick="viewTeam('${escapeAttr(b.team)}')">${escapeHtml(w.member.displayName)}</span>`).join('')}
            </div>
          </div>`).join('')}
      </div>`;
    welcome.appendChild(container);
  }

  // Boot: load chats first (so welcome or restored chat renders correctly)
  loadChatsFromStorage();
  renderChatHistory();
  renderActiveChatMessages();

  // Boot teams
  loadTeams().then(() => {
    // Show morning briefing only if we're on the welcome screen (no active chat)
    if (!activeChat()) showMorningBriefing();
  });

  // Fetch production issue count for sidebar badge
  refreshProdCount();

  document.getElementById('prompt').focus();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

# ═══════════════════════════════════════════════════════
# SEARCH LOGGING — for training data
# ═══════════════════════════════════════════════════════
SEARCH_LOG_PATH = os.path.expanduser('~/Desktop/monogpt_searches.jsonl')


@app.route('/log/query', methods=['POST'])
def log_query():
    data = request.json or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({"ok": False, "error": "empty query"}), 400

    entry = {
        "ts": datetime.datetime.now().isoformat(timespec='seconds'),
        "query": query,
        "intent": data.get('intent', 'unknown'),
        "matched_key": data.get('matched_key'),
        "source": data.get('source', 'chat'),
        "user": os.environ.get('JIRA_EMAIL', ''),
        "chat_id": data.get('chat_id'),
    }
    try:
        with open(SEARCH_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        return jsonify({"ok": True, "path": SEARCH_LOG_PATH})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/log/query', methods=['GET'])
def log_stats():
    """Return simple stats about logged queries"""
    if not os.path.exists(SEARCH_LOG_PATH):
        return jsonify({"total": 0, "path": SEARCH_LOG_PATH})
    total = 0
    intents = {}
    with open(SEARCH_LOG_PATH) as f:
        for line in f:
            try:
                obj = json.loads(line)
                total += 1
                i = obj.get('intent', 'unknown')
                intents[i] = intents.get(i, 0) + 1
            except json.JSONDecodeError:
                continue
    return jsonify({
        "total": total,
        "path": SEARCH_LOG_PATH,
        "by_intent": sorted(intents.items(), key=lambda x: -x[1]),
    })


@app.route('/settings', methods=['GET'])
def get_settings():
    cfg = jira_client.get_config_status()
    # Actually verify token works, not just that it's set
    if cfg.get("configured"):
        test = jira_client.test_connection()
        cfg["reachable"] = test.get("ok", False)
        if not test.get("ok"):
            cfg["reachable_error"] = test.get("error", "")
    else:
        cfg["reachable"] = False
    return jsonify(cfg)


@app.route('/settings', methods=['POST'])
def save_settings():
    data = request.json or {}
    domain = (data.get('domain') or '').strip()
    email = (data.get('email') or '').strip()
    token = (data.get('token') or '').strip()

    if not domain or not email:
        return jsonify({"ok": False, "error": "Domain and email are required"}), 400

    # If token is blank, keep the existing one
    if not token:
        token = os.environ.get('JIRA_TOKEN', '')
        if not token:
            return jsonify({"ok": False, "error": "Token required (first-time setup)"}), 400

    try:
        jira_client.save_config(domain, email, token)
        test = jira_client.test_connection()
        if test.get("ok"):
            return jsonify({"ok": True, "message": f"Connected as {test['name']}"})
        return jsonify({"ok": False, "error": test.get("error", "Test failed")}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/jira/issue/<key>', methods=['GET'])
def jira_issue(key):
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing — set them in Settings"})
    return jsonify(jira_client.get_issue_detail(key))


@app.route('/teams', methods=['GET'])
def list_teams():
    return jsonify({"teams": teams_store.list_teams()})


@app.route('/teams', methods=['POST'])
def create_team():
    name = (request.json or {}).get('name', '').strip()
    result = teams_store.create_team(name)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/teams/<name>', methods=['DELETE'])
def delete_team(name):
    return jsonify(teams_store.delete_team(name))


@app.route('/teams/<name>/search-users', methods=['GET'])
def team_search_users(name):
    """Search JIRA users to add to a team"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"users": []})
    return jsonify({"users": jira_client.search_users(query, max_results=10)})


@app.route('/teams/<name>/members', methods=['POST'])
def add_team_member(name):
    member = request.json or {}
    if not member.get('accountId') or not member.get('displayName'):
        return jsonify({"error": "accountId and displayName required"}), 400
    result = teams_store.add_member(name, member)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/teams/<name>/members/<account_id>', methods=['DELETE'])
def remove_team_member(name, account_id):
    return jsonify(teams_store.remove_member(name, account_id))


@app.route('/teams/<name>/status', methods=['GET'])
def team_status(name):
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing"}), 400
    team = teams_store.get_team(name)
    if not team:
        return jsonify({"error": f"Team '{name}' not found"}), 404
    workload = jira_client.get_team_workload(team['members'])
    if isinstance(workload, dict) and 'error' in workload:
        return jsonify(workload), 500
    return jsonify({
        "team": team,
        "workload": workload,
        "idle_count": sum(1 for w in workload if w['open_count'] == 0),
        "busy_count": sum(1 for w in workload if w['open_count'] > 0),
    })


@app.route('/jira/comment/<issue_key>', methods=['POST'])
def post_comment(issue_key):
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing"}), 400
    body = ((request.json or {}).get('body') or '').strip()
    if not body:
        return jsonify({"error": "Comment body is required"}), 400
    result = jira_client.add_comment(issue_key.upper(), body)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/jira/worklog/<issue_key>', methods=['POST'])
def add_worklog(issue_key):
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing"}), 400
    data = request.json or {}
    time_spent = (data.get('time') or '').strip()
    comment = (data.get('comment') or '').strip()
    if not time_spent:
        return jsonify({"error": "Time is required (e.g. 30m, 1h, 2h 30m)"}), 400
    result = jira_client.add_worklog(issue_key.upper(), time_spent, comment)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/jira/worklog/<issue_key>', methods=['GET'])
def get_worklogs(issue_key):
    if not jira_client.is_configured():
        return jsonify({"worklogs": []})
    return jsonify({"worklogs": jira_client.get_worklogs(issue_key.upper())})


@app.route('/jira/production', methods=['GET'])
def jira_production():
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing"}), 400
    return jsonify(jira_client.get_production_issues())


@app.route('/jira/search', methods=['GET'])
def universal_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({"error": "Query required"}), 400
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing"}), 400
    return jsonify(jira_client.universal_search(q))


@app.route('/jira/member/<account_id>', methods=['GET'])
def jira_member_details(account_id):
    """Detailed workload for a single member — today's work + all open tasks"""
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing"}), 400
    domain = jira_client.JIRA_DOMAIN

    open_jql = (f'assignee = "{account_id}" AND statusCategory != Done '
                f'ORDER BY updated DESC')
    today_jql = (f'assignee = "{account_id}" AND '
                 f'(updated >= startOfDay() OR duedate = now()) '
                 f'ORDER BY updated DESC')

    open_res = jira_client.search_issues(open_jql, max_results=30)
    today_res = jira_client.search_issues(today_jql, max_results=15)

    open_issues = open_res.get("issues", []) if isinstance(open_res, dict) else []
    today_issues = today_res.get("issues", []) if isinstance(today_res, dict) else []

    return jsonify({
        "open_tasks": open_issues,
        "today_tasks": today_issues,
        "has_today_work": len(today_issues) > 0,
        "open_count": len(open_issues),
    })


@app.route('/jira/user/<path:name>', methods=['GET'])
def jira_user(name):
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing — set them in Settings"})
    return jsonify(jira_client.get_tasks_by_user(name))


@app.route('/jira/dashboard', methods=['GET'])
def jira_dashboard():
    if not jira_client.is_configured():
        return jsonify({"error": "JIRA credentials missing — set them in Settings"})
    return jsonify(jira_client.get_dashboard())


@app.route('/jira', methods=['POST'])
def jira():
    cmd = request.json.get('cmd', '')
    if not jira_client.is_configured():
        return jsonify({
            "error": "JIRA credentials missing.\n\n"
                     "STEP 1: Create an API token\n"
                     "  https://id.atlassian.com/manage-profile/security/api-tokens\n"
                     "  → 'Create API token' → name it 'MonoceptGPT' → copy the value\n\n"
                     "STEP 2: Edit the file\n"
                     "  ~/tiny_llm/.env\n"
                     "  Fill JIRA_EMAIL and JIRA_TOKEN\n\n"
                     "STEP 3: Restart the server\n"
                     "  pkill -f 'python3 -u app.py'\n"
                     "  cd ~/tiny_llm && python3 app.py"
        })
    jql = jira_client.parse_jira_command(cmd)
    result = jira_client.search_issues(jql, max_results=15)
    result["jql"] = jql
    return jsonify(result)


@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt', '').lower()
    # ensure prompt ends with space for better generation
    if not prompt.endswith(' '):
        prompt = prompt + ' '

    def stream():
        for ch in generate_stream(prompt, max_tokens=200, temperature=0.8):
            yield ch
            time.sleep(0.015)  # thoda typing effect

    return Response(stream_with_context(stream()), mimetype='text/plain')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MonoceptGPT server ready!")
    print("🌐 Open: http://localhost:8080")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=8080, debug=False)
