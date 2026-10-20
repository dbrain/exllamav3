# EXL3_CONV_GEMM — image input without the solver searches

An image enters the language model as a run of vision tokens (40 to 8,000 depending on its size), and
the first input of a resolution is slow for two unrelated reasons:

* **The MIOpen convolution solver search.** The tower ends in a 2x2 spatial-merge convolution
  (`model.visual.downsample`), the only `F.conv2d` in GLM-5.3. This platform ships no MIOpen find-db
  for gfx1151 (the ROCm SDK's `share/miopen/db` holds no `.fdb.txt` at all), so the first call for a
  convolution shape screens the candidate solvers on the live operands. That shape's batch is the
  vision token count, so the cost is about **0.3 s per vision token** — 68 s for a 400x400 image,
  416 s for 1024x1024 — and it is keyed on the **exact canvas**: a 1 % change in resolution pays it
  again (392.7 s measured for 1000x1000 right after 1024x1024).
* **The dense GEMM screening.** `hgemm`'s tuner (`EXL3_DENSE_GEMM_TUNE`) times every rocBLAS
  solution for each `(n, k, row class)` key it meets. This is not specific to images, but a new image
  size lands on row classes nothing has used before: 20-45 s for the language model and the vision
  projections of one new size, and up to **600 s at startup**.

Both disappear with environment variables. Nothing else in the server changes.

**Since Kyojin 1.5 the GLM server applies the recommended setup below by default** (`EXL3_CONV_GEMM=1`,
`EXL3_DENSE_GEMM_TUNE=0`, set by `tools/glm/serve.py` unless the variable is already in the environment).
The "default" columns and wording in this page describe the engine without those two values, which is
what `EXL3_CONV_GEMM=0 EXL3_DENSE_GEMM_TUNE=1` still gives.

## Recommended setup

```bash
export EXL3_CONV_GEMM=1            # the spatial-merge convolution becomes a GEMM: no MIOpen search
export EXL3_DENSE_GEMM_TUNE=0      # no dense GEMM screening, at startup or per shape
export EXL3_SERVE_DTUNE_PRIME_S=0  # implied by the line above; kept to be explicit
python tools/glm/serve.py \
    --model ~/models/glm53-exl3-td205 --port 18080
```

The first start is ~45 s, the first input of a new size costs seconds instead of minutes, and no cache
file is created. Nothing has to be prepared in advance, and nothing depends on the ROCm version.

`EXL3_DENSE_GEMM_TUNE=0` is not free, but in this model it is nearly free: the untuned path is
20-35 % slower on the GEMMs it would have tuned, and those are a small part of the work because the
expert layers are EXL3-quantized and never use it. Measured end to end: text prefill 549 tok/s
untuned against 539 tok/s tuned (no difference), and a 5,497-token image prefill 560 against 597
tok/s (about 6 %). Decode is untouched — its rows are below the tuner's 256-row floor.

Setting `EXL3_DENSE_GEMM_TUNE=0` also removes the 600 s startup pre-tuning on its own, so
`EXL3_SERVE_DTUNE_PRIME_S=0` is redundant; it is listed to make the intent explicit.

`EXL3_CONV_GEMM=1` runs that convolution as one `hgemm` GEMM. With stride == kernel the output is 1x1,
so it is `x_flat @ w_flat.T`, the same contraction the flat 3D patch embedding next to it already
runs. MIOpen leaves the model. Default off: with the switch unset the forward is unchanged.

## Alternative: `MIOPEN_FIND_MODE=FAST`

```bash
export MIOPEN_FIND_MODE=FAST
```

It removes the MIOpen half of the cost by telling that one convolution to take the first applicable
solver instead of screening the candidates: a first 1024x1024 drops from 431.3 s to 21.7 s, a first
400x400 from 91.4 s to 24.3 s.

* `MIOPEN_FIND_MODE` is global: it changes solver selection for every MIOpen operation in the process
  and permanently forgoes tuned solvers.

## Alternative: pre-generate the tune file offline

For the ~6 % of prefill throughput that the tuned path buys, the tune file can be filled in advance
instead of at request time. This option **requires the tuner to stay on**, so set `EXL3_DENSE_GEMM_TUNE=1` (default).

```bash
# <run the pre-generation first>
# the tune file has to exist before the server starts;
# its commands are below

export EXL3_CONV_GEMM=1
export EXL3_SERVE_DTUNE_PRIME_S=0
python tools/glm/serve.py --model ~/models/glm53-exl3-td205 --port 18080
```

`serve.py` reads the tune file at `$EXL3_DENSE_GEMM_TUNE_FILE`, by default
`~/.cache/exllamav3/dense_gemm_tune.txt`. Its own `prime_dense_tune()` can only fill in missing row
classes for the shapes the file already holds, and it only ever writes the unpadded-A variant of a
key, while the serving path uses the padded-A variant for most EXL3 linears — so on an empty file the
600 s it spends buy almost nothing for the first requests. `tools/strix_halo/prime_dense_tune.sh`
closes both gaps: it is a thin wrapper that activates the venv, sets `EXL3_ROCM_SDK`, sources
`tools/strix_halo/env.sh` (the same environment the servers use) and runs `prime_dense_tune.py`
with every argument passed through.

```bash
# size the job first: lists the shapes, the key counts and the cost, writes nothing
bash tools/strix_halo/prime_dense_tune.sh --model <pack> --dry-run

# the recommended coverage: the language model, then the vision tower up to 2048x2048
bash tools/strix_halo/prime_dense_tune.sh --model <pack> --only lm
bash tools/strix_halo/prime_dense_tune.sh --model <pack> --only vision --max-image 2048x2048
```

| flag | meaning |
|---|---|
| `--model <pack>` | EXL3 pack directory; only its `config.json` is read, no weights are loaded (1.1 GiB RSS, 0.3 s for a 100 GB pack) |
| `--only lm` / `--only vision` / `--only all` | which half to prime; the LM half is 512 keys, the vision half 778 keys for 2048x2048 |
| `--max-image WxH` | derive the vision row ceilings from the pack's own canvas rules (2048x2048 -> 5476 vision tokens, 21904 patches) |
| `--lm-class N` | largest LM prefill row class, default 4096 — the ceiling the chunked prefill actually reaches |
| `--vision-parts convs` / `linears` / `all` | the convolutions are 1-2 keys per row class, the block Linears about 7 |
| `--budget-s N` | stop after N seconds; a run is resumable, keys already in the file for the current tag are skipped |
| `--dry-run` / `--union-flags` / `--rescreen` | plan only / also prime the unused fp32 variant / ignore what the file holds |

Measured on this box (gfx1151, rocBLAS `5.4.0.f000f7786e`):

| run | keys | time |
|---|---|---|
| `--only lm` | 512 | 785 s |
| `--only vision --max-image 2048x2048 --vision-parts convs` | 130 | 46 s |
| `--only vision --max-image 2048x2048` (the rest: block Linears) | 648 | 990 s |
| **total for the 2048x2048 coverage** | **1290** | **1778 s (~30 min)** |

The file is tagged `<rocBLAS version>|<gcnArchName>|exact1`, so **re-run the primer after a ROCm
upgrade** — otherwise every entry is ignored and the screening happens again. That is the main cost of
this option, and the reason the two configurations above are preferred. Coverage also has to match the
images actually sent: `--max-image 2048x2048` reaches 21904 patches (row class 22016) and does nothing
for a 2888x4177 image, which needs the full patch range.

## Comparison with the default configuration

First-token seconds for the **first** input of each of these resolutions:

| | default | `FAST` | GEMM(recommend) | GEMM(pre-generated) |
|---|---|---|---|---|
| start | 702 s | 626 s | **45 s** | 45 s |
| 400x400 | 91.4 s | 24.3 s | 23.8 s | 23.9 s |
| 800x600 | 193.4 s | 2.5 s | 2.2 s | 2.2 s |
| 1024x1024 | 431.3 s | 21.7 s | 3.5 s | 3.5 s |
| 1000x1000 | 392.7 s | 4.9 s | 3.2 s | 3.2 s |
| 2048x2048 | ~27 min¹ | 80.6 s | **35.1 s** | ~35 s |
| 2888x4177 | ~40 min¹ | 72.2 s | **24.7 s** | ~25 s |
| prefill, 5.5K-token prompt | 597 tok/s | 597 tok/s | 560 tok/s | 597 tok/s |
| needs preparation | — | no | no | **30 min, per ROCm version** |

¹ extrapolated at the measured 0.3 s per vision token (5476 and 7992 tokens); not run to completion.
The two large-image cells of the GEMM(pre-generated) column are estimates from an equivalent cache state
(a warm file written by a 2048x2048 run), because that configuration was measured on the smaller
sizes only.

**Time basis.** All four columns were measured with the same protocol, and every cell is the **first**
time that size was sent in its process. The four small sizes went in one after another — 400x400,
800x600, 1024x1024, 1000x1000 — within a single start, and the two large ones the same way within
another start. The figures therefore include the one-off work a request triggers, in particular the
~25 s that a fresh process pays on its first image request: that is why the first row of each group
(400x400, 2048x2048) is more expensive than the larger sizes that follow it. It is a property of the
process and not of the size, and it sits in every column alike, so the comparison within a row still
holds. Repeating a size afterwards is steady state: 9.9 s for a 2048x2048 under GEMM(recommend),
against 35.1 s for its first input.

## Additional notes

**`EXL3_DENSE_GEMM_TUNE=0` has one side effect beyond switching off the tuner.** In
   `exllamav3/modules/quant/fp16.py` it also moves plain fp16 Linears from `ext.hgemm` to torch's
   hipBLASLt path, which the comment there calls 1.3-1.8x slower on small-K/small-N shapes. GLM-5.3 is
   unaffected — the vision tower's 172 Linears are all `LinearEXL3` (`vision_bits = 5`) and the
   language model is EXL3 too (`bits = 2.05`) — but a pack in which some tower layer stays fp16 should
   use `MIOPEN_FIND_MODE=FAST` on its own, which leaves the GEMM path untouched.
