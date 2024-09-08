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

class CATLayer(nn.Module):
    def __init__(self, num_domains, embed_size, heads):
        super(CATLayer, self).__init__()
        self.num_experts = num_domains * 2  # Two experts per domain
        self.experts = nn.ModuleList([ExpertLayer(embed_size) for _ in range(self.num_experts)])
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)

    def forward(self, x, domain_id, training=True):
        # Calculate expert_ids based on domain_id
        expert_id_1 = domain_id * 2
        expert_id_2 = domain_id * 2 + 1

        # Get outputs from all experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        expert_outputs = expert_outputs.permute(1, 2, 0, 3).contiguous()

        # Combine batch size and sequence length into a single dimension
        batch_size, seq_len, num_experts, embed_size = expert_outputs.shape
        attn_input = expert_outputs.view(batch_size * seq_len, num_experts, embed_size)

        # Create a causal mask that allows current expert pair and previous ones
        causal_mask = torch.zeros((self.num_experts, self.num_experts), device=x.device)
        for i in range(0, self.num_experts, 2):
            causal_mask[i:i+2, :i+2] = 1

        # Apply attention mechanism
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, attn_mask=causal_mask)

        # Reshape back to (batch_size, seq_len, num_experts, embed_size)
        attn_output = attn_output.view(batch_size, seq_len, num_experts, embed_size)

        # Select the output corresponding to the given domain_id (expert pair)
        expert_output_1 = attn_output[:, :, expert_id_1, :]
        expert_output_2 = attn_output[:, :, expert_id_2, :]

        # Average or concatenate the outputs from the two experts (you can choose based on your needs)
        expert_output = (expert_output_1 + expert_output_2) / 2

        # Set requires_grad for only the selected expert pair during training
        for i, expert in enumerate(self.experts):
            if i not in [expert_id_1, expert_id_2]:
                for param in expert.parameters():
                    param.requires_grad = False
            else:
                for param in expert.parameters():
                    param.requires_grad = True

        return expert_output

class CATTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, num_domains):
        super(CATTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.cat_layer = CATLayer(num_domains, embed_size, heads)

    def forward(self, x, domain_id, training=True):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)
        cat_output = self.cat_layer(x, domain_id, training)
        x = self.norm2(cat_output + x)
        return x

class CAT(nn.Module):
    def __init__(self, embed_size, heads, num_layers, num_domains, vocab_size, max_length):
        super(CAT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            CATTransformerBlock(embed_size, heads, num_domains) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x, domain_id, training=True):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, domain_id, training)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
