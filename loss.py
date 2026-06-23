import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SiglipLoss(nn.Module):
    """
    Sigmoid Loss for Language Image Pre-training (SigLIP).
    Computes pairwise sigmoid loss instead of contrastive infoNCE.
    """
    def __init__(self, init_t: float = 10.0, init_b: float = -10.0):
        super().__init__()
        # Learnable temperature (stored as log) and bias
        self.t = nn.Parameter(torch.tensor(np.log(init_t)))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_v: Video representations of shape (B, d)
            z_t: Text representations of shape (B, d)
        Returns:
            loss: Scalar loss
        """
        # 1. L2 Normalize the representations
        z_v = F.normalize(z_v, p=2, dim=-1)
        z_t = F.normalize(z_t, p=2, dim=-1)
        
        # 2. Compute logits
        # logits: (B, B) where entry i,j is similarity between video i and text j
        logits = torch.matmul(z_v, z_t.t()) * self.t.exp() + self.b
        
        # 3. Create Ground Truth Labels
        B = z_v.size(0)
        # Identity matrix: 1 on diagonal (positive pairs), 0 elsewhere
        targets = torch.eye(B, device=z_v.device)
        # Map 1 -> 1 and 0 -> -1 for the logsigmoid formulation
        labels = 2 * targets - 1
        
        # 4. Compute Sigmoid Binary Cross Entropy Loss
        # The paper uses: -mean(logsigmoid(labels * logits))
        # Equivalent to binary cross entropy with positive and negative pairs.
        loss = -F.logsigmoid(labels * logits).mean()
        
        return loss
