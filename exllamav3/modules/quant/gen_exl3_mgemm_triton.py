"""Generate exl3_mgemm_triton.py from exl3_mgemm_triton.py.in.

The M==1 decode branch tree of the grouped kernel is copied VERBATIM out of
exllamav3's _fused_dequant_gemm_kernel: hand transcription would be the main
correctness risk in this prototype. Only the trellis base pointer (loaded from
the int64 pointer table by the caller-side preamble) and the output row differ,
and neither appears inside the copied ranges.

Per branch the generator takes:
  - the `if/elif/else` condition line, verbatim
  - the pre-`if M1:` setup block, verbatim
  - the `if M1:` body up to (not including) its store, dedented by one level
    (the M1 guard disappears: this kernel is M==1 only)
  - a store that writes the expert's output row

Branch boundaries are DISCOVERED, not hard-coded line numbers: the earlier
table rotted every time exl3_triton.py gained a line, and the failure mode was
a silently stale generated file (the leak guard aborts before the write, so the
checked-in .py simply stops matching its source). _find_branches locates the
condition lines of the kernel's `if K_BITS ==` chain, then each branch's
`if M1:` and the store that ends its body, and asserts the shape it expects.
"""
import os
import hashlib
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3_triton.py")
TEMPLATE = "exl3_mgemm_triton.py.in"
OUT = "exl3_mgemm_triton.py"

# The store the generated branches end with. The grouped kernel's y_ptr is
# already advanced to this expert's row, so the shared epilogue helper takes
# the same arguments it does in exl3_triton.py.
STORE_ACC = (
    "        _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n, acc, svh_ptr,\n"
    "                       had_r_scale, FUSE_OUT_HAD, BLOCK_N)\n"
)
STORE_OUT = (
    "        _store_out_had(y_ptr, offs_n * stride_yn, offs_n, mask_n,\n"
    "                       tl.reshape(out, (BLOCK_N,)), svh_ptr,\n"
    "                       had_r_scale, FUSE_OUT_HAD, BLOCK_N)\n"
)

# The condition lines of the kernel's branch chain, in order. The last entry is
# the generic gather fallback, whose condition is a bare `else:`.
CONDS = [
    ("bits=4",     "    if K_BITS == 4 and FULL:"),
    ("bits=6",     "    elif K_BITS == 6 and FULL:"),
    ("bits=1/2/8", "    elif (K_BITS == 1 or K_BITS == 2 or K_BITS == 8) and FULL:"),
    ("bits=3",     "    elif K_BITS == 3 and FULL:"),
    ("bits=5/7",   "    elif (K_BITS == 5 or K_BITS == 7) and FULL:"),
    ("generic",    "    else:"),
]

# A branch's M1 body ends at the first line of one of these forms; everything
# from there to the branch's `else:` (the M > 1 half) is dropped and replaced
# by the generator's own store.
BODY_END = ("if SPLITS == 1:", "_store_out_had(", "tl.store(")


def _find_branches(L):
    """[(label, cond_i, setup_a, setup_b, body_a, body_b)] as 1-based inclusive
    line numbers, discovered from the source rather than tabulated."""
    kern = next(i for i, l in enumerate(L)
                if l.startswith("def _fused_dequant_gemm_kernel("))
    out, at = [], kern
    for label, cond in CONDS:
        try:
            ci = L.index(cond, at)
        except ValueError:
            raise SystemExit(f"branch {label}: condition line not found: {cond!r}")
        at = ci + 1
        try:
            mi = L.index("        if M1:", ci)
        except ValueError:
            raise SystemExit(f"branch {label}: no `if M1:` after line {ci + 1}")
        # the M1 body runs to the first store-ish line at 12-space depth
        bi = next((j for j in range(mi + 1, len(L))
                   if L[j].startswith("            ") and L[j].strip().startswith(BODY_END)),
                  None)
        if bi is None:
            raise SystemExit(f"branch {label}: no store found after `if M1:`")
        out.append((label, ci + 1, ci + 2, mi, mi + 2, bi))
    return out


# (Historical) The K_BITS==8 M1 reduction left `s` as
# (nj, c3, cl) where the shared tail expects (c3, nj, cl), so the output tile is
# a permutation of the right values whenever NN > 1 (BLOCK_N > 16). Verified
# against ext.reconstruct: the repo kernel is off by ~85-175% of peak at
# BLOCK_N 32/64/128 for K_BITS=8 while its own M>1 tensor-core branch is
# correct to 3e-4, and every other width (1-7) is correct at M==1. Accidentally
# correct at BLOCK_N=16 (NN==1), which is why it went unnoticed. Now fixed in
# exl3_triton.py itself (tests/test_exl3_triton_bits8.py), so FIXES is empty
# and the M1 branch is copied verbatim with no deviation.
#
# This is the ONE place the copy deviates from exl3_triton.py; everything else
# is verbatim.
# Upstream exl3_triton.py now carries the (2, 0, 1) permute fix, so the M1 branch
# is copied with no deviation at all.
FIXES = []
# Names that only exist in the non-M1 / split-K halves of the source kernel and
# must never appear in a copied branch. The fused-Hadamard names (FUSE_HAD,
# suh_ptr, svh_ptr, _had_x_tile, _x_sub16, _store_out_had, had_r_scale) are
# DELIBERATELY copied through: the template defines all of them, resolving the
# scale vectors per expert out of the same int64 pointer table as the trellis.
FORBIDDEN = ("BLOCK_M", "offs_m", "mask_m", "pid_split", "stride_ys", "SPLITS", "M1")


def main():
    with open(SRC) as f:
        text = f.read()
    L = text.split("\n")
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]

    branches = _find_branches(L)

    def rng(a, b):
        return "\n".join(L[a - 1:b]) + "\n"

    def dedent4(s):
        return "\n".join(l[4:] if l.startswith("    ") else l for l in s.split("\n"))

    parts = []
    for label, cond, sa, sb, ba, bb in branches:
        store = STORE_OUT if "out = tl.permute" in rng(ba, bb) else STORE_ACC
        parts.append(L[cond - 1] + "\n")
        parts.append(rng(sa, sb))
        parts.append(dedent4(rng(ba, bb)))
        parts.append(store)
        print(f"  {label:11s} cond {cond:5d}  setup {sa}-{sb}  body {ba}-{bb}",
              file=sys.stderr)
    body = "".join(parts)

    for before, after in FIXES:
        if before not in body:
            print("FIX no longer applies (upstream changed?):\n" + before, file=sys.stderr)
            return 1
        body = body.replace(before, after)

    for name in FORBIDDEN:
        for i, line in enumerate(body.split("\n")):
            code = line.split("#")[0]
            if name in code:
                print(f"LEAK {name} at generated line {i + 1}: {line}", file=sys.stderr)
                return 1

    with open(TEMPLATE) as f:
        tmpl = f.read()
    assert "# <<<BRANCHES>>>" in tmpl
    banner = (
        f"    # --- BEGIN generated by _gen.py from exl3_triton.py "
        f"(sha256[:16] {sha}) ---\n"
    )
    out = tmpl.replace(
        "# <<<BRANCHES>>>\n",
        banner + body + "    # --- END generated ---\n",
    )
    with open(OUT, "w") as f:
        f.write(out)
    print(f"wrote {OUT}: {len(body.splitlines())} generated lines, src sha {sha}")
    subprocess.run([sys.executable, "-c", f"import ast;ast.parse(open('{OUT}').read())"],
                   check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
