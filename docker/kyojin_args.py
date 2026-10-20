#!/usr/bin/env python3
"""Turn the container environment into the command line of one Kyojin server.

Used by docker/entrypoint.sh. Every variable is optional: the image bakes the model it serves (KYOJIN_MODEL), and
each model has defaults for a 128 GB machine (context, sampling of the model card, hub repository). The only mount a
user needs is /models. If no pack is there, it is downloaded from the hub into /models.

Run it directly to see the command a configuration produces (it downloads a missing pack):

    KYOJIN_MODEL=qwen MODEL_DIR=/models python docker/kyojin_args.py
"""
import contextlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

SERVERS = {"qwen": "tools/qwen/serve.py", "glm": "tools/glm/serve.py", "mimo": "tools/mimo/serve.py"}
# model_type values found in the config.json of the published packs
TYPE_HINTS = (("qwen", "qwen"), ("glm", "glm"), ("mimo", "mimo"))

# Per-model defaults. Context: the value of the README / model card quickstart (-c 131072), which leaves about
# 14 GiB (Qwen) and 10 GiB (GLM) free on a 128 GB machine. Sampling: the values of the model cards.
PROFILES = {
    "qwen": {"repo": "yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz", "CONTEXT": "131072", "DEFAULT_TEMPERATURE": "1.0",
             "DEFAULT_TOP_P": "0.95", "DEFAULT_TOP_K": "20", "DEFAULT_MIN_P": "0.0"},
    "glm": {"repo": "yamz-labs/GLM-5.3-Flash-EXL3-Yamz", "CONTEXT": "131072", "DEFAULT_TEMPERATURE": "1.0",
            "DEFAULT_TOP_P": "0.95"},
    "mimo": {"repo": "yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz", "CONTEXT": "131072", "DEFAULT_TEMPERATURE": "1.0",
             "DEFAULT_TOP_P": "0.95"},
}

# env name -> (flag, {family: flag override or None when the family has no such option}, is_switch)
OPTIONS = {
    "MODEL_ID": ("--model-id", {}, False),
    "DEFAULT_TEMPERATURE": ("--default-temperature", {}, False),
    "DEFAULT_TOP_P": ("--default-top-p", {}, False),
    "DEFAULT_TOP_K": ("--default-top-k", {"glm": None, "mimo": None}, False),
    "DEFAULT_MIN_P": ("--default-min-p", {"glm": None, "mimo": None}, False),
    "REASONING_EFFORT": ("--default-reasoning-effort", {"mimo": None}, False),
    "DRAFT_TOKENS": ("--ndt", {"glm": "--num-draft"}, False),
    "DRAFT_POLICY": ("--draft-policy", {"glm": None, "mimo": None}, False),
    "SLOT_SAVE_PATH": ("--slot-save-path", {}, False),
    "NO_VISION": ("--no-vision", {"glm": None, "mimo": None}, True),
    "NO_THINKING": ("--no-thinking", {"glm": None, "mimo": None}, True),
    "NO_UNCENSOR": ("--no-uncensor", {"mimo": None}, True),
}
# Positive switches: KYOJIN_ENABLE_X=1 keeps the feature on (the default), =0 is KYOJIN_NO_X=1.
ENABLE_SWITCHES = {"ENABLE_VISION": "NO_VISION", "ENABLE_THINKING": "NO_THINKING", "ENABLE_UNCENSOR": "NO_UNCENSOR"}
TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}
# Weight files of a pack are listed in this index when the pack is sharded
INDEX = "model.safetensors.index.json"
GLM_EH_SIDECAR = "mtp_eh_proj.st"  # optional file of the GLM pack, read by tools/glm/serve.py


class ConfigError(Exception):
    pass


def log(msg: str) -> None:
    print(f"[kyojin] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- pack lookup

def detect_family(pack: Path):
    """Family named by config.json, or None when it names none of ours. Unreadable config.json is an error."""
    try:
        cfg = json.loads((pack / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read {pack}/config.json: {e}")
    text = " ".join([str(cfg.get("model_type", ""))] + [str(a) for a in cfg.get("architectures", [])]).lower()
    for hint, family in TYPE_HINTS:
        if hint in text:
            return family
    return None


def missing_files(pack: Path) -> list:
    """Weight files a pack should hold and does not. An unsharded pack needs at least one safetensors file."""
    index = pack / INDEX
    if index.is_file():
        try:
            names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        except (OSError, ValueError, KeyError):
            return [INDEX + " (unreadable)"]
        return [n for n in names if not (pack / n).is_file()]
    return [] if any(pack.glob("*.safetensors")) else ["*.safetensors"]


def check_family(pack: Path, family: str) -> None:
    found = detect_family(pack)
    if found is not None and found != family:
        raise ConfigError(f"{pack} is a {found} pack but this image serves {family}: use the {found} image, or set KYOJIN_MODEL={found}")


def local_pack(model_dir: Path, family: str, repo_name: str, repo_forced: bool):
    """The pack folder to serve, or None when there is none (then it has to be downloaded)."""
    if (model_dir / "config.json").is_file():
        check_family(model_dir, family)
        return model_dir
    if (model_dir / repo_name / "config.json").is_file():
        check_family(model_dir / repo_name, family)
        return model_dir / repo_name
    if repo_forced:
        return None
    # one /models folder can hold the packs of several models: take the one of this family
    packs = [p.parent for p in sorted(model_dir.glob("*/config.json")) if detect_family(p.parent) == family]
    if len(packs) > 1:
        raise ConfigError(f"MODEL_DIR={model_dir} holds {len(packs)} {family} packs ({', '.join(p.name for p in packs)}): "
                          f"point MODEL_DIR at one, or set KYOJIN_HF_REPO to the one to use")
    return packs[0] if packs else None


class HubDownloader:
    """Hub access, behind two methods so that tests can replace it."""

    def size(self, repo: str):
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        return sum(s.size or 0 for s in info.siblings)

    def download(self, repo: str, dest: Path) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=str(dest))  # resumes: finished files are kept, partial ones continue


def fetch_pack(repo: str, dest: Path, hub, reason: str, env) -> None:
    if env.get("KYOJIN_DOWNLOAD", "1").strip().lower() in FALSE:
        raise ConfigError(f"{reason} and KYOJIN_DOWNLOAD=0 forbids the download: mount the pack at /models")
    parent = dest.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ConfigError(f"{reason}, but {parent} is not writable, so the pack cannot be downloaded: "
                          f"mount a writable /models (-v /path/to/models:/models) or mount the pack")
    try:
        total = hub.size(repo)
    except Exception as e:  # offline, private repo, unknown repo: say so before trying the real download
        raise ConfigError(f"{reason}, and {repo} cannot be read on the hub ({type(e).__name__}: {e}). "
                          f"Set KYOJIN_HF_REPO to a valid repository, HF_TOKEN for a private one, or mount the pack")
    gb = total / 1e9
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < total:
        raise ConfigError(f"{reason}: {repo} needs {gb:.1f} GB and {parent} has {free / 1e9:.1f} GB free")
    log(f"{reason}. Downloading {repo} ({gb:.1f} GB) into {dest}. This can take a long time; "
        f"if it stops, start the container again and it resumes.")
    try:
        hub.download(repo, dest)
    except Exception as e:
        raise ConfigError(f"download of {repo} failed ({type(e).__name__}: {e}); start again to resume")
    log(f"download of {repo} finished")


def resolve_pack(model_dir: Path, family: str, env, hub=None) -> Path:
    if not model_dir.is_dir():
        raise ConfigError(f"MODEL_DIR={model_dir} does not exist: mount the model folder (-v /path/to/models:/models)")
    profile = PROFILES[family]
    forced = env.get("KYOJIN_HF_REPO", "").strip()
    repo = forced or profile["repo"]
    repo_name = repo.rstrip("/").split("/")[-1]
    pack = local_pack(model_dir, family, repo_name, bool(forced))
    hub = hub or HubDownloader()
    if pack is None:
        dest = model_dir / repo_name
        fetch_pack(repo, dest, hub, f"no {family} pack found in {model_dir}", env)
        pack = dest
    else:
        gone = missing_files(pack)
        if gone:
            fetch_pack(repo, pack, hub, f"the pack {pack} is incomplete ({len(gone)} file(s) missing, first: {gone[0]})", env)
    gone = missing_files(pack)
    if gone:
        raise ConfigError(f"the pack {pack} is still incomplete after the download ({gone[0]} missing)")
    return pack


# ---------------------------------------------------------------- command line

def flag_value(name: str, env, family: str):
    """KYOJIN_<name> from the environment, else the default of the model."""
    v = env.get("KYOJIN_" + name, "").strip()
    return v if v else PROFILES[family].get(name, "")


def bare(name: str, env, default: str) -> str:
    """Names without prefix (PORT, HOST, CONTEXT, MODEL_DIR) also work as KYOJIN_<name>."""
    for key in ("KYOJIN_" + name, name):
        if env.get(key, "").strip():
            return env[key].strip()
    return default


def parse_bool(name: str, env):
    v = env.get("KYOJIN_" + name, "").strip().lower()
    if not v:
        return None
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ConfigError(f"KYOJIN_{name}={v!r}: expected 1 or 0")


def switches(env, family: str) -> tuple:
    """Flags from NO_* and ENABLE_* (both spellings). Returns (flags, notes)."""
    flags, notes = [], []
    for enable, no in ENABLE_SWITCHES.items():
        flag, overrides, _ = OPTIONS[no]
        flag = overrides.get(family, flag)
        off, on = parse_bool(no, env), parse_bool(enable, env)
        disable = off is True or on is False
        keep = on is True or off is False
        if disable and keep:
            raise ConfigError(f"KYOJIN_{no} and KYOJIN_{enable} contradict each other: set one of them")
        if flag is None:
            if disable:
                raise ConfigError(f"KYOJIN_{no if off else enable} is not supported by the {family} server")
            if on:
                notes.append(f"KYOJIN_{enable}=1 ignored: the {family} server has no such option")
        elif disable:
            flags.append(flag)
    return flags, notes


def build(env, hub=None) -> tuple:
    """Return (argv, notes). argv starts after the interpreter: [server.py, flags...]."""
    family = env.get("KYOJIN_MODEL", "").strip().lower() or "auto"
    model_dir = Path(bare("MODEL_DIR", env, "/models")).expanduser()
    if family == "auto":
        # no model baked in: read it from a pack given as MODEL_DIR itself (or the only pack inside)
        packs = [model_dir] if (model_dir / "config.json").is_file() else [p.parent for p in sorted(model_dir.glob("*/config.json"))]
        found = {detect_family(p) for p in packs} - {None}
        if len(found) != 1:
            raise ConfigError("cannot tell the model family from the packs in MODEL_DIR: set KYOJIN_MODEL=qwen|glm|mimo")
        family = found.pop()
    if family not in SERVERS:
        raise ConfigError(f"KYOJIN_MODEL={family!r}: expected qwen, glm, mimo or auto")
    port = bare("PORT", env, "8731")
    if not port.isdigit():
        raise ConfigError(f"PORT={port!r} is not a number")
    ctx = bare("CONTEXT", env, PROFILES[family]["CONTEXT"])
    if not ctx.isdigit() or int(ctx) <= 0:
        raise ConfigError(f"CONTEXT={ctx!r} must be a positive number of tokens")
    flags, notes = switches(env, family)
    options = []
    for name, (flag, overrides, is_switch) in OPTIONS.items():
        if is_switch:
            continue
        value = flag_value(name, env, family)
        if not value:
            continue
        flag = overrides.get(family, flag)
        if flag is None:
            raise ConfigError(f"KYOJIN_{name} is not supported by the {family} server")
        options += [flag, value]
    pack = resolve_pack(model_dir, family, env, hub)
    argv = [SERVERS[family], "--model", str(pack), "--host", bare("HOST", env, "0.0.0.0"), "--port", port, "-c", ctx]
    argv += options + flags + shlex.split(env.get("KYOJIN_EXTRA_ARGS", ""))
    if family == "glm" and not (pack / GLM_EH_SIDECAR).is_file() and "EXL3_MTP_EH_FP16" not in env:
        notes.append(f"WARNING: {pack} has no {GLM_EH_SIDECAR}: speculative decode is about 15 % slower without it; "
                     "download the pack again to get the file")
    return argv, notes + [f"family={family} pack={pack} context={ctx}"]


def main() -> int:
    quiet_stdout = "--null" in sys.argv[1:]
    try:
        # anything printed by the hub client must not end up in the argument list read by the entrypoint
        with contextlib.redirect_stdout(sys.stderr) if quiet_stdout else contextlib.nullcontext():
            argv, notes = build(os.environ)
    except ConfigError as e:
        log(f"configuration error: {e}")
        return 2
    for n in notes:
        log(n)
    if quiet_stdout:
        sys.stdout.write("\0".join(argv) + "\0")
    else:
        print(" ".join(shlex.quote(a) for a in ["python"] + argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
