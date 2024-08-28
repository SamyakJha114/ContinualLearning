import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)
        ff_output = self.ff(x)
        x = self.norm2(ff_output + x)
        return x

class ExpertLayer(nn.Module):
    def __init__(self, embed_size):
        super(ExpertLayer, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        return self.ff(x)

class DeMIXLayer(nn.Module):
    def __init__(self, num_experts, embed_size):
        super(DeMIXLayer, self).__init__()
        self.experts = nn.ModuleList([ExpertLayer(embed_size) for _ in range(num_experts)])
        self.num_experts = num_experts

    def forward(self, x, expert_id):
        assert 0 <= expert_id < self.num_experts, "Invalid expert id"
        return self.experts[expert_id](x)

class DeMIXTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, num_experts):
        super(DeMIXTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.demix_layer = DeMIXLayer(num_experts, embed_size)

    def forward(self, x, expert_id):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)
        demix_output = self.demix_layer(x, expert_id)
        x = self.norm2(demix_output + x)
        return x
    
class GPTDeMIX(nn.Module):
    def __init__(self, embed_size, heads, num_layers, num_experts, vocab_size, max_length):
        super(GPTDeMIX, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DeMIXTransformerBlock(embed_size, heads, num_experts) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x, expert_id):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, expert_id)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
