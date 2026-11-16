"""Test k2 pruned RNN-T loss on GPU.

Run: python tests/test_k2_pruned_gpu.py
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import k2

from model.transducer import Joiner


def test_k2_pruned_loss_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return False

    device = torch.device("cuda")
    encoder_dim, predictor_dim, hidden_dim, vocab_size = 384, 512, 512, 5000
    B, T_max, S_max = 2, 80, 30
    s_range = 5

    joiner = Joiner(encoder_dim, predictor_dim, hidden_dim, vocab_size).to(device)

    encoder_out = torch.randn(B, T_max, encoder_dim, device=device)
    predictor_out = torch.randn(B, S_max + 1, predictor_dim, device=device)
    encoder_out_lens = torch.tensor([T_max, 60], dtype=torch.long, device=device)
    target_lens = torch.tensor([S_max, 20], dtype=torch.long, device=device)
    targets = torch.randint(1, vocab_size, (B, S_max), device=device)

    boundary = torch.zeros(B, 4, dtype=torch.long, device=device)
    boundary[:, 2] = target_lens
    boundary[:, 3] = encoder_out_lens

    # Step 1: simple projections
    am_proj = joiner.simple_am_proj(encoder_out)
    lm_proj = joiner.simple_lm_proj(predictor_out)
    print(f"AM proj: {am_proj.shape}, LM proj: {lm_proj.shape}")

    # Step 2: simple loss for prune ranges
    simple_loss, (px_grad, py_grad) = k2.rnnt_loss_simple(
        lm=lm_proj.float(),
        am=am_proj.float(),
        symbols=targets.long(),
        termination_symbol=0,
        boundary=boundary,
        reduction="mean",
        return_grad=True,
    )
    print(f"Simple loss: {simple_loss.item():.4f}")

    # Step 3: get prune ranges
    ranges = k2.get_rnnt_prune_ranges(
        px_grad=px_grad,
        py_grad=py_grad,
        boundary=boundary,
        s_range=s_range,
    )
    print(f"Ranges: {ranges.shape}")

    # Step 4: prune am/lm
    am_pruned, lm_pruned = k2.do_rnnt_pruning(
        am=encoder_out, lm=predictor_out, ranges=ranges
    )
    print(f"AM pruned: {am_pruned.shape}, LM pruned: {lm_pruned.shape}")

    # Step 5: joiner on pruned inputs
    pruned_logits = joiner.forward_pruned(am_pruned, lm_pruned)
    print(f"Pruned logits: {pruned_logits.shape}")

    # Step 6: pruned loss
    pruned_loss = k2.rnnt_loss_pruned(
        logits=pruned_logits.float(),
        symbols=targets.long(),
        ranges=ranges,
        termination_symbol=0,
        boundary=boundary,
        reduction="mean",
    )
    print(f"Pruned loss: {pruned_loss.item():.4f}")

    # Step 7: backward
    total_loss = simple_loss * 0.5 + pruned_loss
    total_loss.backward()
    print(f"Backward pass succeeded! Total loss: {total_loss.item():.4f}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print("ALL TESTS PASSED - k2 pruned loss works on GPU!")
    return True


if __name__ == "__main__":
    success = test_k2_pruned_loss_gpu()
    sys.exit(0 if success else 1)
