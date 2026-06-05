"""Equivalence and gradient tests for the compiled chunked scan in fast_kernels mode.

Requires CUDA; skips gracefully on CPU-only systems.
Run: python experiments/test_scan.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from qlm.model import QuantumChannelLM


def test_equivalence():
    """Scan (fast_kernels=True) NLL matches reference (fast_kernels=False) within 1e-4."""
    if not torch.cuda.is_available():
        print("[SKIP] test_equivalence: no CUDA device")
        return

    torch.manual_seed(42)
    V, n, W = 32, 16, 4
    B, L, tbptt = 4, 32, 8

    tokens = torch.randint(0, V, (B, L)).cuda()

    m_ref = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=False).cuda()
    m_ref.eval()
    with torch.no_grad():
        out_ref = m_ref.forward(tokens, tbptt=tbptt)
    nll_ref = out_ref["nll_sum"].item()

    # Use float32 compute_dtype so bf16 rounding doesn't affect the comparison.
    m_scan = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=True,
                              compute_dtype=torch.float32).cuda()
    m_scan.load_state_dict(m_ref.state_dict())
    m_scan.eval()
    with torch.no_grad():
        out_scan = m_scan.forward(tokens, tbptt=tbptt)
    nll_scan = out_scan["nll_sum"].item()

    diff = abs(nll_ref - nll_scan)
    tol = 1e-4 * abs(nll_ref) + 1e-4
    print(f"[test_equivalence] nll_ref={nll_ref:.6f}  nll_scan={nll_scan:.6f}"
          f"  diff={diff:.2e}  tol={tol:.2e}")
    assert diff < tol, f"NLL mismatch: diff={diff} > tol={tol}"
    print("[PASS] test_equivalence")


def test_equivalence_decohere():
    """Scan with decohere=True matches reference."""
    if not torch.cuda.is_available():
        print("[SKIP] test_equivalence_decohere: no CUDA device")
        return

    torch.manual_seed(7)
    V, n, W = 16, 12, 3
    B, L, tbptt = 3, 24, 8

    tokens = torch.randint(0, V, (B, L)).cuda()

    m_ref = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=False).cuda()
    m_ref.eval()
    with torch.no_grad():
        out_ref = m_ref.forward(tokens, decohere=True, tbptt=tbptt)
    nll_ref = out_ref["nll_sum"].item()

    m_scan = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=True,
                              compute_dtype=torch.float32).cuda()
    m_scan.load_state_dict(m_ref.state_dict())
    m_scan.eval()
    with torch.no_grad():
        out_scan = m_scan.forward(tokens, decohere=True, tbptt=tbptt)
    nll_scan = out_scan["nll_sum"].item()

    diff = abs(nll_ref - nll_scan)
    tol = 1e-4 * abs(nll_ref) + 1e-4
    print(f"[test_equivalence_decohere] nll_ref={nll_ref:.6f}  nll_scan={nll_scan:.6f}"
          f"  diff={diff:.2e}")
    assert diff < tol, f"NLL mismatch (decohere): diff={diff} > tol={tol}"
    print("[PASS] test_equivalence_decohere")


def test_gradient_flow():
    """Gradients flow correctly through the compiled scan with TBPTT."""
    if not torch.cuda.is_available():
        print("[SKIP] test_gradient_flow: no CUDA device")
        return

    torch.manual_seed(0)
    V, n, W = 16, 8, 2
    m = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=True,
                         compute_dtype=torch.float32).cuda()
    tokens = torch.randint(0, V, (2, 16)).cuda()
    out = m.forward(tokens, tbptt=8)
    out["loss"].backward()
    gnorm = sum(p.grad.abs().sum().item()
                for p in m.parameters() if p.grad is not None)
    print(f"[test_gradient_flow] total grad L1 = {gnorm:.4f}")
    assert gnorm > 0, "gradient norm is zero — gradients not flowing"
    print("[PASS] test_gradient_flow")


def test_return_probs_unaffected():
    """fast_kernels + return_probs falls back correctly and matches the reference."""
    if not torch.cuda.is_available():
        print("[SKIP] test_return_probs_unaffected: no CUDA device")
        return

    torch.manual_seed(3)
    V, n, W = 12, 8, 2
    B, L = 2, 10

    tokens = torch.randint(0, V, (B, L)).cuda()

    m_ref = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=False).cuda()
    m_ref.eval()
    with torch.no_grad():
        out_ref = m_ref.forward(tokens, return_probs=True)

    m_fast = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=True,
                              compute_dtype=torch.float32).cuda()
    m_fast.load_state_dict(m_ref.state_dict())
    m_fast.eval()
    with torch.no_grad():
        out_fast = m_fast.forward(tokens, return_probs=True)

    nll_diff = abs(out_ref["nll_sum"].item() - out_fast["nll_sum"].item())
    probs_diff = (out_ref["probs"] - out_fast["probs"]).abs().max().item()
    print(f"[test_return_probs_unaffected] nll_diff={nll_diff:.2e}"
          f"  probs_max_diff={probs_diff:.2e}")
    assert nll_diff < 1e-4, f"NLL mismatch in return_probs path: {nll_diff}"
    assert probs_diff < 1e-4, f"probs mismatch in return_probs path: {probs_diff}"
    print("[PASS] test_return_probs_unaffected")


def test_no_tbptt():
    """Scan with tbptt=0 (single full-sequence chunk) matches reference."""
    if not torch.cuda.is_available():
        print("[SKIP] test_no_tbptt: no CUDA device")
        return

    torch.manual_seed(11)
    V, n, W = 20, 10, 3
    B, L = 3, 20

    tokens = torch.randint(0, V, (B, L)).cuda()

    m_ref = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=False).cuda()
    m_ref.eval()
    with torch.no_grad():
        out_ref = m_ref.forward(tokens, tbptt=0)
    nll_ref = out_ref["nll_sum"].item()

    m_scan = QuantumChannelLM(V, dim=n, kraus=W, fast_kernels=True,
                              compute_dtype=torch.float32).cuda()
    m_scan.load_state_dict(m_ref.state_dict())
    m_scan.eval()
    with torch.no_grad():
        out_scan = m_scan.forward(tokens, tbptt=0)
    nll_scan = out_scan["nll_sum"].item()

    diff = abs(nll_ref - nll_scan)
    tol = 1e-4 * abs(nll_ref) + 1e-4
    print(f"[test_no_tbptt] nll_ref={nll_ref:.6f}  nll_scan={nll_scan:.6f}"
          f"  diff={diff:.2e}")
    assert diff < tol, f"NLL mismatch (no tbptt): diff={diff} > tol={tol}"
    print("[PASS] test_no_tbptt")


if __name__ == "__main__":
    print("Running scan tests (first call compiles the chunk kernel — may take a moment)...")
    test_equivalence()
    test_equivalence_decohere()
    test_gradient_flow()
    test_return_probs_unaffected()
    test_no_tbptt()
    print("\nAll scan tests passed.")
