import torch
import torch.nn as nn

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
    def __init__(self, num_domains, embed_size):
        super(DeMIXLayer, self).__init__()
        self.num_experts = num_domains * 2  # Two experts per domain
        self.experts = nn.ModuleList([ExpertLayer(embed_size) for _ in range(self.num_experts)])

    def forward(self, x, domain_id):
        # Calculate expert_ids based on domain_id
        expert_id_1 = domain_id * 2
        expert_id_2 = domain_id * 2 + 1

        # Set requires_grad for only the selected expert pair during training
        for i, expert in enumerate(self.experts):
            if i not in [expert_id_1, expert_id_2]:
                for param in expert.parameters():
                    param.requires_grad = False
            else:
                for param in expert.parameters():
                    param.requires_grad = True

        # Get the outputs of both experts and combine them
        expert_output_1 = self.experts[expert_id_1](x)
        expert_output_2 = self.experts[expert_id_2](x)

        # Average or concatenate the outputs from the two experts
        expert_output = (expert_output_1 + expert_output_2) / 2

        return expert_output

class DeMIXTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, num_domains):
        super(DeMIXTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.demix_layer = DeMIXLayer(num_domains, embed_size)

    def forward(self, x, domain_id):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)
        demix_output = self.demix_layer(x, domain_id)
        x = self.norm2(demix_output + x)
        return x
    
class GPTDeMIX(nn.Module):
    def __init__(self, embed_size, heads, num_layers, num_domains, vocab_size, max_length):
        super(GPTDeMIX, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DeMIXTransformerBlock(embed_size, heads, num_domains) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x, domain_id):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, domain_id)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
