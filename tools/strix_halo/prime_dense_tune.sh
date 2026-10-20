#!/usr/bin/env bash
# Prime the dense-GEMM tune cache before the first server start.
#
# Same environment as the servers (venv + ROCm SDK + strix_halo env), then run
# prime_dense_tune.py with every argument passed through.
#
#   bash tools/strix_halo/prime_dense_tune.sh --model <pack dir> --dry-run
#   bash tools/strix_halo/prime_dense_tune.sh --model <pack dir>
#
# EXL3_DENSE_GEMM_TUNE_FILE decides which file is written (default
# ~/.cache/exllamav3/dense_gemm_tune.txt, the same one the server reads).
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"
export EXL3_ROCM_SDK="$(rocm-sdk path --root)"
# shellcheck disable=SC1091
source tools/strix_halo/env.sh

exec python tools/strix_halo/prime_dense_tune.py "$@"
