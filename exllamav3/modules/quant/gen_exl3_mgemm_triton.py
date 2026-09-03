"""Generate exl3_mgemm_triton.py from exl3_mgemm_triton.py.in.

The M==1 decode branch tree of the grouped kernel is copied VERBATIM out of
exllamav3's _fused_dequant_gemm_kernel: hand transcription would be the main
correctness risk in this prototype. Only the trellis base pointer (loaded from
the int64 pointer table by the caller-side preamble) and the output row differ,
and neither appears inside the copied ranges.

Per branch the generator takes:
  - the `if/elif/else` condition line, verbatim
  - the pre-`if M1:` setup block, verbatim
  - the `if M1:` body, dedented by one level (the M1 guard disappears: this
    kernel is M==1 only)
  - a store that writes the expert's output row

Line ranges refer to exl3_triton.py as of the revision recorded in _SRC_SHA.
Rerun with --check to verify the source has not moved under them.
"""
import os
import hashlib
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3_triton.py")
TEMPLATE = "exl3_mgemm_triton.py.in"
OUT = "exl3_mgemm_triton.py"

STORE_ACC = (
    "        tl.store(y_ptr + offs_n * stride_yn, "
    "acc.to(y_ptr.dtype.element_ty), mask=mask_n)\n"
)
STORE_OUT = (
    "        tl.store(y_ptr + offs_n * stride_yn, "
    "tl.reshape(out, (BLOCK_N,)).to(y_ptr.dtype.element_ty), mask=mask_n)\n"
)

# (label, cond_line, setup_first, setup_last, body_first, body_last, store)
BRANCHES = [
    ("bits=4",     818,  819,  843,  845,  886, STORE_ACC),
    ("bits=6",     926,  927,  961,  963, 1004, STORE_OUT),
    ("bits=1/2/8", 1052, 1053, 1075, 1077, 1126, STORE_ACC),
    ("bits=3",     1165, 1166, 1190, 1192, 1223, STORE_ACC),
    ("bits=5/7",   1255, 1256, 1281, 1283, 1301, STORE_ACC),
    ("generic",    1329, 1330, 1412, 1414, 1431, STORE_ACC),
]

# Anchors that must be found at (line, expected prefix); catches a moved source.
ANCHORS = [
    (818, "    if K_BITS == 4 and"),
    (844, "        if M1:"),
    (886, "            acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))"),
    (926, "    elif K_BITS == 6 and"),
    (962, "        if M1:"),
    (1004, "            out = tl.permute(tl.join(h0, h1), (0, 2, 1))"),
    (1052, "    elif (K_BITS == 1 or K_BITS == 2 or K_BITS == 8)"),
    (1076, "        if M1:"),
    (1165, "    elif K_BITS == 3 and"),
    (1191, "        if M1:"),
    (1255, "    elif (K_BITS == 5 or K_BITS == 7)"),
    (1282, "        if M1:"),
    (1329, "    else:"),
    (1413, "        if M1:"),
]

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
# Names that only exist in the non-M1 / split-K halves of the source kernel.
FORBIDDEN = ("BLOCK_M", "offs_m", "mask_m", "pid_split", "stride_ys", "SPLITS", "M1")


def main():
    with open(SRC) as f:
        text = f.read()
    L = text.split("\n")
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]

    bad = False
    for ln, prefix in ANCHORS:
        if not L[ln - 1].startswith(prefix):
            print(f"ANCHOR MISMATCH at line {ln}: expected {prefix!r}, got {L[ln - 1]!r}",
                  file=sys.stderr)
            bad = True
    if bad:
        print("exl3_triton.py has moved; fix the line ranges in _gen.py", file=sys.stderr)
        return 1

    def rng(a, b):
        return "\n".join(L[a - 1:b]) + "\n"

    def dedent4(s):
        return "\n".join(l[4:] if l.startswith("    ") else l for l in s.split("\n"))

    parts = []
    for label, cond, sa, sb, ba, bb, store in BRANCHES:
        parts.append(L[cond - 1] + "\n")
        parts.append(rng(sa, sb))
        parts.append(dedent4(rng(ba, bb)))
        parts.append(store)
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
