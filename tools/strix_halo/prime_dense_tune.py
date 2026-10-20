"""Create/populate the dense-GEMM tune cache *before* the first server start.

Why this exists
---------------
Every dense product exllamav3 runs goes through hgemm, whose C++ tuner (hgemm.cu dtune)
screens the rocBLAS solutions on the live operands the first time it sees a key

    (n, k, rows rounded up to 256, (ldc == n) | (padded A pitch) << 1, out fp32)

and appends the winner to EXL3_DENSE_GEMM_TUNE_FILE
(default ~/.cache/exllamav3/dense_gemm_tune.txt). Inside a request that screening costs
0.05-4 s per key. Measured stage times for a first start with no tune file: 27 s weights +
1 s drafter + 1 s vision tower + 139 s warm-up (its 4.6K-token prefill walks the row classes)
+ 538 s startup priming = ~12 min to /health, against ~38 s with a populated file (warm-up
9 s, priming 0.3 s).

tools/glm/serve.py's own prime_dense_tune() cannot help on an empty file -- it only fills in
missing row classes for the (n, k, ...) shapes the file already holds. This script
enumerates the shapes from the pack itself, building the module lists straight from
config.json without loading a single weight (a 100 GB pack costs ~1 GiB of RSS, ~0.3 s), and
runs that same screening up front with small preallocated buffers.

What it primes
--------------
* every dense Linear of the target and MTP components (the filter of
  exllamav3/model/dense_warmup.py), rows 256 .. --lm-class
* every dense Linear of the vision tower (rows = patches, up to max_image_tokens * 4)
* the vision tower's GEMM convolutions: the flat 3D patch embed (rows = patches) and the
  reshape2d spatial-merge downsample (rows = vision tokens)

Rows below 256 never reach the tuner (MIN_M in hgemm.cu), so they are not generated.

Entries are only valid for the rocBLAS version + GPU arch that wrote them: the tuner tags
every line and ignores lines whose tag does not match. Re-run after a ROCm upgrade (or use
--rescreen).

    python tools/strix_halo/prime_dense_tune.py --model <pack dir> --dry-run
    python tools/strix_halo/prime_dense_tune.py --model <pack dir>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

MIN_M = 256          # hgemm.cu: rows below this never reach the tuner
PAD_ELEMS = 64       # exl3.PAD_ELEMS: width a row-padded activation adds
CLASS_STEP = 256


@dataclass
class Shape:
    n: int
    k: int
    f32: set[int]                     # 1 = fp32 output, 0 = fp16 output
    key4: set[int]                    # bit0 ldc == n, bit1 padded A pitch
    classes: list[int]
    label: str = ""

    def keys(self):
        return [(self.n, self.k, mc, k4, f32)
                for mc in self.classes for k4 in sorted(self.key4) for f32 in sorted(self.f32)]


def m_class(m: int) -> int:
    """hgemm.cu: rows are rounded up to a 256 row class."""
    return ((m + CLASS_STEP - 1) // CLASS_STEP) * CLASS_STEP


def classes_up_to(top: int) -> list[int]:
    """Every row class the tuner can be asked for at or below `top` (top included, rounded up)."""
    return list(range(MIN_M, m_class(top) + 1, CLASS_STEP))


def default_tune_path() -> Path:
    p = os.environ.get("EXL3_DENSE_GEMM_TUNE_FILE")
    return Path(p) if p else Path(os.environ.get("HOME", "/tmp")) / ".cache/exllamav3/dense_gemm_tune.txt"


def read_keys(path: Path) -> list[tuple[str, int, int, int, int, int]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 7:
            out.append((f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])))
    return out


def newest_tag(keys) -> str | None:
    """The tag the tuner would write now = the tag of the newest line in the file."""
    return keys[-1][0] if keys else None


def _conv_flags(out_dtype, k: int) -> tuple[set[int], set[int]]:
    """(f32, key4) for a Conv GEMM. The conv's own out_dtype IS the GEMM output dtype
    (Conv.forward allocates y with it), unlike a Linear, whose inner module decides."""
    f32 = {1} if out_dtype == torch.float32 else {0}
    return f32, ({1, 3} if k % 128 == 0 else {1})


def _linear_flags(k: int, union: bool) -> tuple[set[int], set[int]]:
    """(f32, key4) for a dense Linear GEMM.

    Measured on GLM-5.3-Flash (all 20 dense Linear shapes in a populated tune file): the GEMM
    output is fp16 -- f32 = 0 -- even where the Linear module declares out_dtype = torch.float32
    (that dtype is a later cast, not the GEMM's C tensor). f32 = 1 occurs only on the vision
    Conv layers. --union-flags also primes f32 = 1 here, at twice the screening cost.

    A padded activation pitch is only possible where the producer pads (EXL3 linears,
    k % 128 == 0); eligibility can depend on loaded weights, so both key4 variants are primed.
    """
    f32 = {0, 1} if union else {0}
    return f32, ({1, 3} if k % 128 == 0 else {1})


def linear_shapes(models: dict, top: int, union: bool = False) -> list[Shape]:
    """Dense Linears, same filter as exllamav3/model/dense_warmup.py._dense_linears."""
    from exllamav3.modules.linear import Linear

    seen: dict[tuple, Shape] = {}
    for comp, model in models.items():
        for m in model:
            if not isinstance(m, Linear) or ".experts." in m.key or m.out_features == 0 \
                    or m.out_features > 65536 or m.in_features == 0:
                continue
            n, k = m.out_features, m.in_features
            f32, key4 = _linear_flags(k, union)
            sig = (n, k, tuple(sorted(f32)), tuple(sorted(key4)))
            if sig not in seen:
                seen[sig] = Shape(n, k, f32, key4, classes_up_to(top), f"{comp or 'target'}: {m.key}")
    return list(seen.values())


def vision_rows_for(cfg, width: int, height: int) -> tuple[int, int]:
    """(vision tokens, patches) the largest canvas of an image that fits `width x height` produces."""
    from exllamav3.architecture.mm_processing.glm5_next import glm5_vision_canvas

    v, pp = cfg.vision, cfg.vision_pp
    th, tw = glm5_vision_canvas(pp.temporal_patch_size, height, width, pp.temporal_patch_size,
                                pp.patch_size * pp.merge_size, pp.min_image_tokens,
                                pp.max_image_tokens)
    patches = (th // pp.patch_size) * (tw // pp.patch_size)
    return patches // (v.spatial_merge_size ** 2), patches


def vision_rows(m, token_top: int, patch_top: int) -> int:
    """Row ceiling for one vision module.

    The tower feeds whole patches to the blocks and to the flat patch embed, but the merger
    sits behind the spatial-merge downsample and therefore sees one row per merged token.
    Priming the merger at the patch count would quadruple its key count for nothing.
    """
    return token_top if ".merger" in m.key else patch_top


def vision_conv_shapes(cfg, model, token_top: int, patch_top: int) -> list[Shape]:
    """The vision tower's GEMM convolutions as (n, k, rows) key space."""
    from exllamav3.modules.conv import Conv

    out = []
    for m in model:
        if not isinstance(m, Conv) or not (m.flat or m.reshape2d):
            continue
        f32, key4 = _conv_flags(m.out_dtype, m.dim)
        # a flat 3D patch embed runs one row per patch; reshape2d runs one row per merged token
        # the flat patch embed is fed patches; the reshape2d downsample merges them
        rows = patch_top if m.flat else token_top
        out.append(Shape(m.out_channels, m.dim, f32, key4, classes_up_to(rows),
                         f"vision: {m.key} "
                         f"({'patch embed, rows = patches' if m.flat else 'spatial merge, rows = tokens'})"))
    return out


def screen(shapes: list[Shape], have: list, budget_s: float, log=print) -> dict:
    """One hgemm per missing key: the tuner screens that key and appends the winner."""
    from exllamav3.ext import exllamav3_ext as ext

    tag = newest_tag(have)
    known = {(t, n, k, mc, k4, f32) for (t, n, k, mc, k4, f32) in have if t == tag}
    todo = []
    for s in shapes:
        for key in s.keys():
            if (tag, *key) not in known:
                todo.append((s, key))
    total = sum(len(s.keys()) for s in shapes)
    log(f"tag in use: {tag or '(none yet -- the tuner will stamp its own)'}")
    log(f"keys to screen: {len(todo)} of {total} for these shapes")
    if not todo:
        log("nothing to do: every key is already in the file for this tag")
        return {"screened": 0, "seconds": 0.0, "todo": 0}

    t0, done = time.perf_counter(), 0
    gen = torch.Generator(device="cuda").manual_seed(0)
    by_shape: dict[tuple, list] = {}
    for s, key in todo:
        by_shape.setdefault((s.n, s.k), []).append((s, key))
    for (n, k), items in sorted(by_shape.items()):
        max_mc = max(key[2] for _, key in items)
        padded = any(key[3] & 2 for _, key in items)
        # Two activation buffers: a contiguous one (lda == k) and a wider one whose [:, :k] view
        # carries the padded row pitch (lda != k) that the other key4 variant encodes.
        a_buf = torch.empty((max_mc, k), device="cuda", dtype=torch.half).normal_(0, 0.02, generator=gen)
        a_wide = (torch.empty((max_mc, k + PAD_ELEMS), device="cuda", dtype=torch.half)
                  .normal_(0, 0.02, generator=gen)) if padded else None
        b_buf = torch.empty((k, n), device="cuda", dtype=torch.half).normal_(0, 0.02, generator=gen)
        c_bufs = {dt: torch.empty((max_mc, n), device="cuda", dtype=dt)
                  for dt in {torch.float32 if key[4] else torch.half for _, key in items}}
        for s, key in items:
            _, _, mc, k4, f32 = key
            a = a_wide[:mc, :k] if (k4 & 2) else a_buf[:mc]
            ext.hgemm(a, b_buf, c_bufs[torch.float32 if f32 else torch.half][:mc])
            done += 1
            if budget_s and time.perf_counter() - t0 > budget_s:
                log(f"  budget {budget_s:.0f} s reached at {done}/{len(todo)} keys")
                break
            if done % 128 == 0:
                torch.cuda.synchronize()
                el = time.perf_counter() - t0
                log(f"  {done}/{len(todo)} keys, {el:.0f} s ({el / done * 1000:.0f} ms/key)")
        del a_buf, a_wide, b_buf
        c_bufs.clear()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {"screened": done, "seconds": time.perf_counter() - t0, "todo": len(todo)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="EXL3 pack directory (only config.json is read)")
    ap.add_argument("--tune-file", default=None,
                    help="dense GEMM tune file (default: $EXL3_DENSE_GEMM_TUNE_FILE)")
    ap.add_argument("--lm-class", type=int, default=4096,
                    help="largest LM prefill row class to prime (default 4096, the serving chunk)")
    ap.add_argument("--max-image", default="",
                    help="cover every image that fits WxH, e.g. 2048x2048: derives the vision row "
                         "ceilings from the pack's canvas rules instead of a raw class number")
    ap.add_argument("--vision-class", type=int, default=0,
                    help="manual override: largest row class to prime for the vision tower "
                         "(default 0 = the whole patch range, max_image_tokens * 4 = 32000 on "
                         "GLM-5.3, which is the expensive end)")
    ap.add_argument("--only", choices=("all", "vision", "lm"), default="all")
    ap.add_argument("--vision-parts", choices=("all", "convs", "linears"), default="all",
                    help="which half of the vision tower to prime. The convolutions are 1-2 keys "
                         "per row class and were the layer that made MIOpen slow; the block Linears "
                         "are ~7 keys per class and only 0.2-3.3 s each, so leaving them lazy costs "
                         "seconds per new image size instead of a long generation")
    ap.add_argument("--budget-s", type=float, default=0.0, help="stop after N seconds (0 = no limit)")
    ap.add_argument("--union-flags", action="store_true",
                    help="also prime f32 = 1 for dense Linears (2x the keys; measured unnecessary)")
    ap.add_argument("--rescreen", action="store_true",
                    help="ignore what the file already holds and screen every key again")
    ap.add_argument("--dry-run", action="store_true", help="list the plan, screen nothing")
    args = ap.parse_args()

    from exllamav3 import Config, Model

    path = Path(args.tune_file) if args.tune_file else default_tune_path()
    cfg = Config.from_directory(args.model)
    t0 = time.perf_counter()
    models: dict = {}
    for comp in (None, "mtp", "vision"):
        try:
            models[comp] = Model.from_config(cfg, **({} if comp is None else {"component": comp}))
        except Exception as exc:                                     # noqa: BLE001
            print(f"  ({comp or 'target'} component not built: {type(exc).__name__}: {exc})")
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30
    n_mod = sum(1 for m in models.values() for _ in m)
    print(f"built {n_mod} module objects in "
          f"{time.perf_counter() - t0:.2f} s, rss {rss:.2f} GiB (no weights loaded)")

    shapes: list[Shape] = []
    if args.only in ("all", "lm"):
        shapes += linear_shapes({k: v for k, v in models.items() if k in (None, "mtp")},
                                args.lm_class, args.union_flags)
    if args.only in ("all", "vision") and models.get("vision") is not None:
        pp = getattr(cfg, "vision_pp", None)
        token_top = int(getattr(pp, "max_image_tokens", 8192) or 8192)
        patch_top = token_top * 4          # the flat patch embed runs one row per patch
        if args.max_image:
            w, h = (int(x) for x in args.max_image.lower().replace("*", "x").split("x"))
            token_top, patch_top = vision_rows_for(cfg, w, h)
            print(f"max-image {w}x{h} -> vision rows: tokens {token_top} "
                  f"(class {m_class(token_top)}), patches {patch_top} (class {m_class(patch_top)})")
        if args.vision_class:
            token_top = patch_top = args.vision_class
        if args.vision_parts in ("all", "convs"):
            shapes += vision_conv_shapes(cfg, models["vision"], token_top, patch_top)
        if args.vision_parts in ("all", "linears"):
            for m in models["vision"]:
                pass
            for sh in linear_shapes({"vision": models["vision"]}, patch_top, args.union_flags):
                top = token_top if ".merger" in sh.label else patch_top
                sh.classes = classes_up_to(top)
                shapes.append(sh)

    total = sum(len(s.keys()) for s in shapes)
    lm_keys = sum(len(s.keys()) for s in shapes if s.label.startswith(("target", "mtp")))
    print(f"shapes: {len(shapes)}, keys if nothing were cached: {total} "
          f"(language model {lm_keys}, vision {total - lm_keys})")
    print("cost: ~0.05-4 s per key, rising with the row class (measured here: the language model "
          "+ conv set, 893 keys, 1106 s; the vision rows 4352..8192, 189 keys, 668 s)")
    print("bound the job with --only lm / --only vision, --vision-class and --budget-s")
    print(f"tune file: {path}")
    have = [] if args.rescreen else read_keys(path)
    if args.dry_run:
        for s in sorted(shapes, key=lambda s: -len(s.keys())):
            print(f"  n={s.n:<6} k={s.k:<6} f32={sorted(s.f32)} key4={sorted(s.key4)} "
                  f"rows={s.classes[0]}..{s.classes[-1]} ({len(s.keys())} keys)  <- {s.label}")
        print(f"\n{total} keys would be screened; nothing written (--dry-run)")
        return 0

    st = screen(shapes, have, args.budget_s)
    if st["screened"]:
        after = read_keys(path)
        print(f"\nscreened {st['screened']} keys in {st['seconds']:.0f} s "
              f"({st['seconds'] / max(st['screened'], 1) * 1000:.0f} ms/key)")
        print(f"tune file now holds {len(after)} lines; the next server start skips this screening")
    return 0


if __name__ == "__main__":
    sys.exit(main())
