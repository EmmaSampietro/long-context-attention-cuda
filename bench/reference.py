"""NumPy ground-truth for attention. Used by bench/validate.py.

Pure numpy — no PyTorch dependency. Supports all three kernel modes: dense
(every token can attend to every other), windowed (restricted local range
+ optional global tokens), and blocksparse (masked by a block matrix).
Supports an optional causal mask as in autoregressive models.

For approximation-quality work that needs Cholesky / Gaussian processes,
see bench/approx_quality.py (which still uses torch).
"""

import numpy as np

def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation that is also safe for rows (or
    axes) that are fully masked (i.e., all values are -inf).

    This mimics the behavior of correct GPU kernels, where an all-masked row
    should give a probability vector of all zeros rather than NaNs
    (contrast: plain np.exp(-inf)/np.sum gives NaN/0 division).
    """
    # Compute the max along the softmax axis for numerical stability.
    # If all values in a row are -inf, the max will also be -inf.
    m = np.max(x, axis=axis, keepdims=True)
    # If max is -inf (all-masked row): subtracting -inf yields NaN.
    # Instead, replace -inf with 0 to get exp(-inf - 0) = 0 everywhere on that row.
    m = np.where(np.isfinite(m), m, 0.0)
    # Exponentiate after stabilizing so large values do not overflow.
    e = np.exp(x - m)
    # Sum along the softmax axis; this will be zero if everything was -inf.
    s = np.sum(e, axis=axis, keepdims=True)
    # Where the sum is positive, compute e/s as usual. Otherwise (all masked),
    # output zeros to avoid NaNs.
    return np.where(s > 0, e / s, 0.0)

def _build_window_mask(N: int, w: int, G: int) -> np.ndarray:
    """
    Construct a boolean mask for "windowed" attention.

    - Each token i can attend to any token j within a window, i.e., |i-j| <= w.
    - The first G tokens are "global tokens": everyone can attend to them, and
      they themselves can attend everywhere (these are like [CLS]/[SEP] in
      some models).
    - Output: [N, N] mask where True means "i may attend to j".

    Args:
        N: sequence length
        w: local attention window half-width (token i attends to i-w:i+w)
        G: number of global tokens (special tokens at beginning)

    Returns:
        mask: [N, N] boolean array for allowed attendances.
    """
    idx = np.arange(N)
    # Local: each position attends to positions in its +/-w window
    local = np.abs(idx[:, None] - idx[None, :]) <= w   # [N, N] bool
    if G > 0:
        # Global columns: everyone can attend to the first G tokens (columns)
        global_cols = idx[None, :] < G  # shape [1, N], True means col is global
        # Global rows: rows 0..G-1 correspond to global tokens, which can attend anywhere
        global_rows = idx[:, None] < G  # shape [N, 1], True means row is global
        # Tokens can attend if local, or if it's a global col, or if it's a global row
        return local | global_cols | global_rows
    return local

def _expand_block_mask(block_mask: np.ndarray, B: int) -> np.ndarray:
    """
    Upsample a blockwise mask to a full dense mask.

    Given a block sparse mask of shape [num_blocks, num_blocks], expand each
    bool in the block mask to a (B x B) region of True/False in the full [N, N]
    mask.

    Args:
        block_mask: [num_blocks, num_blocks] boolean array
        B: block size

    Returns:
        full_mask: [N, N] boolean array (N = num_blocks * B)
    """
    # Repeat along rows and columns
    return np.repeat(np.repeat(block_mask, B, axis=0), B, axis=1)

def reference_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mode: str = "dense",
    w: int = None,
    G: int = 0,
    block_mask: np.ndarray = None,
    B: int = 64,
    causal: bool = False,
) -> np.ndarray:
    """
    Pure-NumPy attention reference implementation.

    Args:
        Q, K, V: Arrays of shape [..., N, d]. The ... can be batch or head dims.
        mode: "dense", "windowed", or "blocksparse".
            - dense: standard attention (every position can attend everywhere)
            - windowed: restricts to |i-j| <= w and optional global tokens (G)
            - blocksparse: arbitrary pattern, specified by block_mask and B
        w: window size (only for "windowed" mode)
        G: number of global tokens (only for "windowed" mode)
        block_mask: [N/B, N/B] mask for "blocksparse" mode
        B: block size for "blocksparse" mode
        causal: If True, exclude attending to "future" tokens (upper triangle)
    Returns:
        Output: [..., N, d] array of attended values.
    """

    # --- Input shapes and dtypes ---
    d = Q.shape[-1]        # Hidden dimension (head size)
    N = Q.shape[-2]        # Sequence length

    # Always use float32 for all math -- float64 is rarely justified
    # and slows down large reference computations.
    Q = Q.astype(np.float32, copy=False)
    K = K.astype(np.float32, copy=False)
    V = V.astype(np.float32, copy=False)

    # --- Compute unnormalized attention scores ---
    # [Optional batch/head]/N/d @ [Optional batch/head]/d/N  --> [Optional]/N/N
    # The scaling by 1/sqrt(d) ensures gradients are in a reasonable scale as per "Attention is All You Need".
    S = (Q @ np.swapaxes(K, -2, -1)) / np.sqrt(np.float32(d))

    # --- Apply attention pattern/mask ---
    # Set S[i, j] = -inf where attention is not allowed; this ensures its softmax is exactly zero
    if mode == "windowed":
        assert w is not None, "windowed mode requires w"
        mask = _build_window_mask(N, w, G)
        # S[mask == False] = -inf
        S = np.where(mask, S, np.float32(-np.inf))
    elif mode == "blocksparse":
        assert block_mask is not None, "blocksparse mode requires block_mask"
        mask = _expand_block_mask(block_mask, B)
        S = np.where(mask, S, np.float32(-np.inf))
    elif mode != "dense":
        # Guard against accidental typo
        raise ValueError(f"unknown mode: {mode}")

    # --- Apply causal mask if requested ---
    if causal:
        # Causal attention means no attending to future tokens:
        # For i < j (i.e. above the diagonal), prohibit attendance.
        # np.triu produces True above the diagonal
        cmask = np.triu(np.ones((N, N), dtype=bool), k=1)
        S = np.where(cmask, np.float32(-np.inf), S)

    # --- Softmax normalization ---
    # Convert unnormalized attention scores to probabilities for each row
    # Each row sums to 1 *unless* the row is fully masked, in which case it will be all zeros.
    P = _stable_softmax(S, axis=-1)   # [..., N, N] shape

    # --- Weighted average of values ---
    # Standard attention output: for each query, sum values weighted by attention probs
    # P @ V: (..., N, N) @ (..., N, d) --> (..., N, d)
    return P @ V
