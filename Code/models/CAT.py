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
    def __init__(self, num_experts, embed_size, heads):
        super(CATLayer, self).__init__()
        self.experts = nn.ModuleList([ExpertLayer(embed_size) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)

    def forward(self, x, expert_id, training=True):
        if training:
            # Get outputs from all experts
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
            expert_outputs = expert_outputs.permute(1, 2, 0, 3).contiguous()

            # Combine batch size and sequence length into a single dimension
            batch_size, seq_len, num_experts, embed_size = expert_outputs.shape
            attn_input = expert_outputs.view(batch_size * seq_len, num_experts, embed_size)

            # Create a causal mask to mask future experts
            causal_mask = torch.tril(torch.ones((self.num_experts, self.num_experts), device=x.device))

            # Apply attention mechanism
            attn_output, _ = self.attention(attn_input, attn_input, attn_input, attn_mask=causal_mask)

            # Reshape back to (batch_size, seq_len, num_experts, embed_size)
            attn_output = attn_output.view(batch_size, seq_len, num_experts, embed_size)

            # Select the output corresponding to the given expert_id
            expert_output = attn_output[:, :, expert_id, :]

        else:
            # Directly get the output of the corresponding expert during inference
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
            expert_outputs = expert_outputs.permute(1, 2, 0, 3).contiguous()

            # Combine batch size and sequence length into a single dimension
            batch_size, seq_len, num_experts, embed_size = expert_outputs.shape

            attn_input = expert_outputs.view(batch_size * seq_len, num_experts, embed_size)

            # Apply attention mechanism
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)

            # Reshape back to (batch_size, seq_len, num_experts, embed_size)
            attn_output = attn_output.view(batch_size, seq_len, num_experts, embed_size)

            # Select the output corresponding to the given expert_id
            expert_output = attn_output[:, :, expert_id, :]

        # Set requires_grad for only the selected expert during training
        for i, expert in enumerate(self.experts):
            if i != expert_id:
                for param in expert.parameters():
                    param.requires_grad = False
            else:
                for param in expert.parameters():
                    param.requires_grad = True

        return expert_output

class CATTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, num_experts):
        super(CATTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.cat_layer = CATLayer(num_experts, embed_size, heads)

    def forward(self, x, expert_id, training=True):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)
        cat_output = self.cat_layer(x, expert_id, training)
        x = self.norm2(cat_output + x)
        return x

class CAT(nn.Module):
    def __init__(self, embed_size, heads, num_layers, num_experts, vocab_size, max_length):
        super(CAT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            CATTransformerBlock(embed_size, heads, num_experts) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x, expert_id, training=True):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, expert_id, training)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def init_param(self, expert_id):
        """
        Initialize the weights of the expert with index expert_id using the weights of the (expert_id - 1)-th expert
        across all layers.
        """
        if expert_id <= 0 or expert_id >= self.layers[0].cat_layer.num_experts:
            raise ValueError("Invalid expert_id. Must be between 1 and the total number of experts minus 1.")

        # Iterate through all layers
        for layer in self.layers:
            # For each layer, copy the parameters from expert (expert_id - 1) to expert (expert_id)
            prev_expert_id = expert_id - 1
            current_expert = layer.cat_layer.experts[expert_id]
            previous_expert = layer.cat_layer.experts[prev_expert_id]
            
            # Copy parameters from previous expert to current expert
            for current_param, previous_param in zip(current_expert.parameters(), previous_expert.parameters()):
                current_param.data.copy_(previous_param.data)
