#!/bin/bash
# CPU test of every branch of docker/entrypoint.sh. No GPU, no model, no network: the server is a stub script.
# Run: bash docker/test_entrypoint.sh   (works on the host or inside the image: docker run --rm --entrypoint bash IMG docker/test_entrypoint.sh)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
FAIL=0
ok()   { echo "ok   $1"; }
bad()  { echo "FAIL $1"; FAIL=1; }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# fake application root: env.sh, a stub server, and the real entrypoint + argument script
R=$T/root; mkdir -p "$R/tools/strix_halo" "$R/tools/qwen" "$R/docker"
cp "$HERE/entrypoint.sh" "$HERE/kyojin_args.py" "$R/docker/"
cat >"$R/tools/strix_halo/env.sh" <<'EOF'
[ "${1:-}" = "--check" ] && { echo ENV_CHECK_RAN; exit 0; }
export KYOJIN_TEST_ENV_SOURCED=1
EOF
cat >"$R/tools/qwen/serve.py" <<'EOF'
import os, signal, sys, time
a = sys.argv[1:]
print("STUB args:", " ".join(a), flush=True)
print("STUB env: sourced=%s triton=%s" % (os.environ.get("KYOJIN_TEST_ENV_SOURCED"), os.environ.get("TRITON_CACHE_DIR")), flush=True)
if "--rc" in a:
    sys.exit(int(a[a.index("--rc") + 1]))
if "--deaf" in a:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("STUB deaf", flush=True)
if "--hang" in a or "--deaf" in a:
    time.sleep(60)
EOF
mkdir -p "$T/bin"; ln -s "$(command -v python3)" "$T/bin/python"
export PATH="$T/bin:$PATH"
PACK=$T/models/Qwen3.8-Flash-Next-EXL3-Yamz; mkdir -p "$PACK"
echo '{"model_type":"qwen4_exp"}' >"$PACK/config.json"; echo w >"$PACK/model.safetensors"
run() { # run NAME [env...] -- [args]: leaves $T/out.NAME and $T/rc.NAME
    local n=$1; shift; local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    env -i PATH="$PATH" HOME="$T/home.$n" KYOJIN_ROOT="$R" KYOJIN_MODEL=qwen MODEL_DIR="$T/models" "${envs[@]}" bash "$R/docker/entrypoint.sh" "$@" >"$T/out.$n" 2>&1
    echo $? >"$T/rc.$n"
}
rc() { cat "$T/rc.$1"; }

# 1 shell and check
mkdir -p "$T/home.shell" "$T/home.check"
run shell -- shell -c 'echo SHELL_OK'
check "shell branch runs bash with the arguments" "grep -q SHELL_OK $T/out.shell && [ $(rc shell) = 0 ]"
run check -- check
check "check branch runs env.sh --check" "grep -q ENV_CHECK_RAN $T/out.check"

# 2 normal start: arguments, appended args, env.sh sourced, Triton cache, exit code propagated
mkdir -p "$T/home.run"
run run -- --rc 7
check "server gets model, context 131072, port 8731, sampling, then the appended args" \
    "grep -q 'STUB args: --model $PACK --host 0.0.0.0 --port 8731 -c 131072 .*--default-temperature 1.0 .*--rc 7' $T/out.run"
check "env.sh is sourced and TRITON_CACHE_DIR points into the cache" "grep -q \"sourced=1 triton=$T/home.run/.cache/triton\" $T/out.run"
check "server exit code is the container exit code" "[ $(rc run) = 7 ] && grep -q 'server exited with code 7' $T/out.run"
mkdir -p "$T/home.run0"; run run0 KYOJIN_CONTEXT=4096 -- --rc 0
check "exit 0 and context override" "[ $(rc run0) = 0 ] && grep -q -- '-c 4096' $T/out.run0"

# 3 configuration errors stop before any server starts
mkdir -p "$T/home.err" "$T/empty"
run err MODEL_DIR="$T/empty" KYOJIN_DOWNLOAD=0 -- --rc 0
check "config error exits 2 and never starts the server" "[ $(rc err) = 2 ] && ! grep -q STUB $T/out.err && grep -q 'configuration error' $T/out.err"

# 4 SIGTERM reaches the server and the exit code is the signal's
mkdir -p "$T/home.term"
env -i PATH="$PATH" HOME="$T/home.term" KYOJIN_ROOT="$R" KYOJIN_MODEL=qwen MODEL_DIR="$T/models" setsid bash "$R/docker/entrypoint.sh" --hang >"$T/out.term" 2>&1 &
EP=$!
for i in $(seq 1 50); do grep -q 'STUB env' "$T/out.term" 2>/dev/null && break; sleep 0.2; done
S=$(date +%s.%N); kill -TERM "$EP"; wait "$EP"; RC=$?; E=$(python3 -c "import time;print(round(time.time()-$S,1))")
check "SIGTERM stops the server (exit 143) in under 3 s (took ${E}s)" "[ $RC = 143 ] && python3 -c 'import sys;sys.exit(0 if $E<3 else 1)'"
check "stop is logged" "grep -q 'signal received' $T/out.term"

# 4b a server that ignores SIGTERM is killed after KYOJIN_STOP_TIMEOUT
mkdir -p "$T/home.deaf"
env -i PATH="$PATH" HOME="$T/home.deaf" KYOJIN_ROOT="$R" KYOJIN_MODEL=qwen MODEL_DIR="$T/models" KYOJIN_STOP_TIMEOUT=2 setsid bash "$R/docker/entrypoint.sh" --deaf >"$T/out.deaf" 2>&1 &
EP=$!
for i in $(seq 1 50); do grep -q 'STUB deaf' "$T/out.deaf" 2>/dev/null && break; sleep 0.2; done
S=$(date +%s.%N); kill -TERM "$EP"; wait "$EP"; RC=$?; E=$(python3 -c "import time;print(round(time.time()-$S,1))")
check "a server deaf to SIGTERM is killed after the timeout (exit 137, took ${E}s)" "[ $RC = 137 ] && python3 -c 'import sys;sys.exit(0 if 1.5<$E<6 else 1)' && grep -q 'SIGKILL' $T/out.deaf"

# 5 SIGTERM during the configuration step (a long download): stub the argument script with a sleeper
R2=$T/root2; cp -r "$R" "$R2"; printf 'import time\ntime.sleep(60)\n' >"$R2/docker/kyojin_args.py"
mkdir -p "$T/home.dl"
env -i PATH="$PATH" HOME="$T/home.dl" KYOJIN_ROOT="$R2" KYOJIN_MODEL=qwen MODEL_DIR="$T/models" setsid bash "$R2/docker/entrypoint.sh" >"$T/out.dl" 2>&1 &
EP=$!; sleep 1; S=$(date +%s.%N); kill -TERM "$EP"; wait "$EP"; RC=$?; E=$(python3 -c "import time;print(round(time.time()-$S,1))")
check "SIGTERM during the download step exits 143 in under 3 s (took ${E}s)" "[ $RC = 143 ] && python3 -c 'import sys;sys.exit(0 if $E<3 else 1)'"

# 6 kernel cache choice
mkdir -p "$T/home.c1"; run c1 -- --rc 0
check "writable models mount: cache is a hidden folder there, linked from ~/.cache" \
    "[ -L $T/home.c1/.cache ] && [ \"\$(readlink $T/home.c1/.cache)\" = $T/models/.kyojin-cache ] && [ -d $T/models/.kyojin-cache ] && grep -q 'kernel cache: $T/models/.kyojin-cache' $T/out.c1"
mkdir -p "$T/home.c2"; run c2 -- --rc 0
check "second start reuses the same cache folder" "[ \"\$(readlink $T/home.c2/.cache)\" = $T/models/.kyojin-cache ]"
mkdir -p "$T/home.c3"; run c3 KYOJIN_CACHE_DIR="$T/mycache" -- --rc 0
check "KYOJIN_CACHE_DIR forces the folder" "[ \"\$(readlink $T/home.c3/.cache)\" = $T/mycache ] && [ -d $T/mycache ]"
mkdir -p "$T/home.c4/.cache/old"; echo x >"$T/home.c4/.cache/old/f"; run c4 -- --rc 0
check "a non-empty ~/.cache is kept and reported" "[ -d $T/home.c4/.cache/old ] && grep -q 'cannot use' $T/out.c4 && [ $(rc c4) = 0 ]"
if [ "$(id -u)" != 0 ]; then
mkdir -p "$T/home.c5"; chmod 555 "$T/models"; run c5 -- --rc 0; chmod 755 "$T/models"
check "read-only models mount: the container's own ~/.cache is used" "[ ! -L $T/home.c5/.cache ] && grep -q 'kernel cache: $T/home.c5/.cache' $T/out.c5 && [ $(rc c5) = 0 ]"
else echo "skip read-only models mount test (running as root)"; fi
mkdir -p "$T/home.c6/.cache" "$T/fakebin"; printf '#!/bin/sh\nexit 0\n' >"$T/fakebin/mountpoint"; chmod +x "$T/fakebin/mountpoint"
PATH="$T/fakebin:$PATH" run c6 -- --rc 0
check "a volume mounted on ~/.cache wins" "[ ! -L $T/home.c6/.cache ] && grep -q 'kernel cache: $T/home.c6/.cache' $T/out.c6"

# 7 unwritable HOME (podman --userns=keep-id): needs a non-root uid, root ignores permissions
if [ "$(id -u)" != 0 ]; then
    mkdir -p "$T/home.ro"; chmod 555 "$T/home.ro"; run ro -- --rc 0; chmod 755 "$T/home.ro"
    check "unwritable HOME: a container-local home is used and the server still starts" "[ $(rc ro) = 0 ] && grep -q 'HOME not writable' $T/out.ro && grep -q 'STUB args' $T/out.ro"
else
    echo "skip unwritable HOME test (running as root)"
fi

[ "$FAIL" = 0 ] && echo "ALL ENTRYPOINT TESTS PASSED" || { echo "SOME ENTRYPOINT TESTS FAILED"; exit 1; }
