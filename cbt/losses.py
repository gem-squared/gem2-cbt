"""Loss = L_LM + lambda * L_boundary  (CBT-v0)."""
import torch.nn.functional as F


def lm_loss(lm_logits, lm_targets):
    B, T, V = lm_logits.shape
    return F.cross_entropy(
        lm_logits.view(B * T, V), lm_targets.view(B * T), ignore_index=-100
    )


def boundary_loss(bnd_logits, labels):
    return F.cross_entropy(bnd_logits, labels)


def total_loss(lm_logits, lm_targets, bnd_logits, labels, lam=0.5):
    L = lm_loss(lm_logits, lm_targets)
    if bnd_logits is not None:
        L = L + lam * boundary_loss(bnd_logits, labels)
    return L
