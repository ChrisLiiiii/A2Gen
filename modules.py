import torch
import torch.nn as nn


# Context-aware Attention Module (CAM)
class CAM(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(CAM, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.gate = nn.Linear(input_dim, input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, seq_features, context_features):
        query = seq_features + context_features.unsqueeze(0)  # Adding context to query
        attn_output, _ = self.multi_head_attention(query, seq_features, seq_features)
        gated_output = self.gate(attn_output) * attn_output
        output = self.mlp(gated_output)
        return output


# Hierarchical Sequence Encoder (HSE)
class HSE(nn.Module):
    def __init__(self, input_dim):
        super(HSE, self).__init__()
        self.cam_action = CAM(input_dim)
        self.cam_item = CAM(input_dim)

    def forward(self, action_seq, item_features, target_item_features):
        action_representation = self.cam_action(action_seq, item_features)
        combined_representation = torch.cat([action_representation, item_features], dim=-1)
        item_representation = self.cam_item(combined_representation, target_item_features)
        return item_representation


# Explicit Autoregressive Generator (EAG)
class EAG(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EAG, self).__init__()
        self.cam_gen = CAM(input_dim)
        self.mlp_classification = nn.Linear(input_dim, num_classes)
        self.mlp_regression = nn.Linear(input_dim, 1)

    def forward(self, action_seq, context_features):
        logits = self.cam_gen(action_seq, context_features)
        action_type = self.mlp_classification(logits)
        action_time = self.mlp_regression(logits)
        return action_type, action_time


# Implicit Autoregressive Generator (IAG)
class IAG(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IAG, self).__init__()
        self.cam_gen = CAM(input_dim)
        self.mlp_classification = nn.Linear(input_dim, num_classes)
        self.mlp_regression = nn.Linear(input_dim, 1)

    def forward(self, logits_seq, context_features):
        logits = self.cam_gen(logits_seq, context_features)
        action_type = self.mlp_classification(logits)
        action_time = self.mlp_regression(logits)
        return action_type, action_time



