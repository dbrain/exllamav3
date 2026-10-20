"""CPU tests for the env -> argument mapping, the pack lookup and the download. Run: python -m pytest docker/test_kyojin_args.py
The hub is never contacted: a fake downloader stands in for it."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import kyojin_args as ka

TYPES = {"qwen": "qwen4_exp", "glm": "glm5_next", "mimo": "mimo_v2"}


def pack(tmp_path, model_type, name="pack", files=("model.safetensors",)):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type, "architectures": ["X"]}))
    for f in files:
        (d / f).write_text("w")
    return d


class FakeHub:
    def __init__(self, size=1e9, fail=None, model_type="qwen4_exp", sharded=False):
        self.calls, self.size_value, self.fail, self.model_type, self.sharded = [], size, fail, model_type, sharded

    def size(self, repo):
        self.calls.append(("size", repo))
        if self.fail == "size":
            raise OSError("offline")
        return self.size_value

    def download(self, repo, dest):
        self.calls.append(("download", repo, Path(dest)))
        if self.fail == "download":
            raise OSError("connection reset")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(json.dumps({"model_type": self.model_type}))
        if self.sharded:
            (dest / ka.INDEX).write_text(json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}))
            (dest / "s1.safetensors").write_text("w"); (dest / "s2.safetensors").write_text("w")
        else:
            (dest / "model.safetensors").write_text("w")


def env_for(tmp_path, family="qwen", **extra):
    return {"KYOJIN_MODEL": family, "MODEL_DIR": str(tmp_path), **extra}


# ---- defaults per model (the card values)

def test_qwen_defaults_for_128gb(tmp_path):
    p = pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build(env_for(tmp_path))
    assert argv[:9] == ["tools/qwen/serve.py", "--model", str(p), "--host", "0.0.0.0", "--port", "8731", "-c", "131072"]
    for flag, v in (("--default-temperature", "1.0"), ("--default-top-p", "0.95"), ("--default-top-k", "20"), ("--default-min-p", "0.0")):
        assert argv[argv.index(flag) + 1] == v
    assert "--no-vision" not in argv and "--no-thinking" not in argv


@pytest.mark.parametrize("family", ["glm", "mimo"])
def test_glm_mimo_defaults(tmp_path, family):
    pack(tmp_path, TYPES[family])
    argv, _ = ka.build(env_for(tmp_path, family))
    assert argv[0] == f"tools/{family}/serve.py" and argv[argv.index("-c") + 1] == "131072"
    assert argv[argv.index("--default-temperature") + 1] == "1.0" and argv[argv.index("--default-top-p") + 1] == "0.95"
    assert "--default-top-k" not in argv and "--default-min-p" not in argv


def test_glm_pack_without_draft_sidecar_is_reported(tmp_path):
    p = pack(tmp_path, TYPES["glm"])
    warned = lambda env: any("mtp_eh_proj.st" in n and "WARNING" in n for n in ka.build(env)[1])
    assert warned(env_for(tmp_path, "glm"))
    assert not warned({**env_for(tmp_path, "glm"), "EXL3_MTP_EH_FP16": "0"})   # an explicit choice is not second-guessed
    (p / ka.GLM_EH_SIDECAR).write_text("w")
    assert not warned(env_for(tmp_path, "glm"))
    pack(tmp_path / "q", TYPES["qwen"])
    assert not any("WARNING" in n for n in ka.build(env_for(tmp_path / "q", "qwen"))[1])


@pytest.mark.parametrize("family", ka.SERVERS)
def test_profile_complete_and_flags_exist_in_servers(family):
    """Each flag emitted for a family, from the profile or from an option, must be an add_argument of that server."""
    root = Path(__file__).parent.parent
    src = (root / ka.SERVERS[family]).read_text()
    prof = ka.PROFILES[family]
    assert prof["repo"].startswith("yamz-labs/") and prof["CONTEXT"].isdigit()
    for env_name, (flag, over, _) in ka.OPTIONS.items():
        f = over.get(family, flag)
        if f is not None:
            assert f'"{f}"' in src, f"{family} server has no {f}"
    assert '"-c"' in src
    for name in prof:
        if name.startswith("DEFAULT_"):
            flag, over, _ = ka.OPTIONS[name]
            assert over.get(family, flag) is not None, f"{family} profile sets {name} but the server lacks it"


# ---- model selection

def test_baked_model_used_even_if_pack_is_other_family(tmp_path):
    p = pack(tmp_path, "glm5_next")
    with pytest.raises(ka.ConfigError, match="glm pack but this image serves qwen"):
        ka.build(env_for(tmp_path, MODEL_DIR=str(p)))
    q = pack(tmp_path, "glm5_next", "Qwen3.8-Flash-Next-EXL3-Yamz")  # named like the qwen repo, but a glm pack
    with pytest.raises(ka.ConfigError, match="glm pack but this image serves qwen"):
        ka.build(env_for(tmp_path))


def test_auto_detects_from_the_only_pack(tmp_path):
    pack(tmp_path, "mimo_v2")
    argv, _ = ka.build({"KYOJIN_MODEL": "auto", "MODEL_DIR": str(tmp_path)})
    assert argv[0] == "tools/mimo/serve.py"
    argv, _ = ka.build({"MODEL_DIR": str(tmp_path)})  # unset = auto
    assert argv[0] == "tools/mimo/serve.py"


def test_auto_cannot_decide(tmp_path):
    pack(tmp_path, "mimo_v2", "a"); pack(tmp_path, "qwen4_exp", "b")
    with pytest.raises(ka.ConfigError, match="set KYOJIN_MODEL"):
        ka.build({"MODEL_DIR": str(tmp_path)})
    with pytest.raises(ka.ConfigError, match="set KYOJIN_MODEL"):
        ka.build({"MODEL_DIR": str(tmp_path / "empty-nonexistent")})


def test_unknown_family(tmp_path):
    with pytest.raises(ka.ConfigError, match="expected qwen"):
        ka.build(env_for(tmp_path, "llama"))


def test_explicit_model_case_insensitive(tmp_path):
    pack(tmp_path, "glm5_next")
    argv, _ = ka.build(env_for(tmp_path, "GLM"))
    assert argv[0] == "tools/glm/serve.py"


# ---- pack lookup

def test_pack_is_model_dir_itself(tmp_path):
    p = pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build({"KYOJIN_MODEL": "qwen", "MODEL_DIR": str(p)})
    assert argv[2] == str(p)


def test_pack_named_after_the_repo_wins(tmp_path):
    pack(tmp_path, "qwen4_exp", "other")
    p = pack(tmp_path, "qwen4_exp", "Qwen3.8-Flash-Next-EXL3-Yamz")
    argv, _ = ka.build(env_for(tmp_path))
    assert argv[2] == str(p)


def test_shared_models_folder_picks_own_family(tmp_path):
    pack(tmp_path, "glm5_next", "g"); q = pack(tmp_path, "qwen4_exp", "q"); pack(tmp_path, "mimo_v2", "m")
    hub = FakeHub()
    argv, _ = ka.build(env_for(tmp_path, "qwen"), hub)
    assert argv[2] == str(q) and hub.calls == []


def test_two_packs_of_the_family(tmp_path):
    pack(tmp_path, "qwen4_exp", "a"); pack(tmp_path, "qwen4_exp", "b")
    with pytest.raises(ka.ConfigError, match="2 qwen packs"):
        ka.build(env_for(tmp_path), FakeHub())


def test_missing_model_dir(tmp_path):
    with pytest.raises(ka.ConfigError, match="does not exist"):
        ka.build(env_for(tmp_path, MODEL_DIR=str(tmp_path / "nope")))


def test_unreadable_config(tmp_path):
    d = tmp_path / "p"; d.mkdir(); (d / "config.json").write_text("{not json")
    with pytest.raises(ka.ConfigError, match="cannot read"):
        ka.build(env_for(tmp_path, MODEL_DIR=str(d)))


def test_pack_with_unknown_model_type_is_accepted_when_family_is_set(tmp_path):
    p = pack(tmp_path, "gpt2")
    argv, _ = ka.build(env_for(tmp_path, MODEL_DIR=str(p)))
    assert argv[2] == str(p)


# ---- download

def test_download_when_no_pack(tmp_path):
    hub = FakeHub(sharded=True)
    argv, notes = ka.build(env_for(tmp_path), hub)
    dest = tmp_path / "Qwen3.8-Flash-Next-EXL3-Yamz"
    assert hub.calls == [("size", "yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz"), ("download", "yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz", dest)]
    assert argv[2] == str(dest)


@pytest.mark.parametrize("family,repo", [("glm", "yamz-labs/GLM-5.3-Flash-EXL3-Yamz"), ("mimo", "yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz")])
def test_download_repo_per_model(tmp_path, family, repo):
    hub = FakeHub(model_type=TYPES[family])
    ka.build(env_for(tmp_path, family), hub)
    assert hub.calls[0] == ("size", repo)


def test_download_repo_override(tmp_path):
    hub = FakeHub()
    argv, _ = ka.build(env_for(tmp_path, KYOJIN_HF_REPO="someone/My-Pack"), hub)
    assert hub.calls[1] == ("download", "someone/My-Pack", tmp_path / "My-Pack") and argv[2] == str(tmp_path / "My-Pack")


def test_forced_repo_ignores_other_packs_in_folder(tmp_path):
    pack(tmp_path, "qwen4_exp", "old")
    hub = FakeHub()
    argv, _ = ka.build(env_for(tmp_path, KYOJIN_HF_REPO="someone/My-Pack"), hub)
    assert argv[2] == str(tmp_path / "My-Pack")


def test_log_line_with_size_comes_before_the_download(tmp_path, capsys):
    ka.build(env_for(tmp_path), FakeHub(size=1.5e9))
    err = capsys.readouterr().err
    assert "Downloading yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz (1.5 GB)" in err and "resumes" in err


def test_incomplete_pack_resumes(tmp_path):
    d = tmp_path / "Qwen3.8-Flash-Next-EXL3-Yamz"; d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    (d / ka.INDEX).write_text(json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}))
    (d / "s1.safetensors").write_text("w")
    hub = FakeHub(sharded=True)
    argv, _ = ka.build(env_for(tmp_path), hub)
    assert hub.calls[1] == ("download", "yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz", d) and (d / "s2.safetensors").exists()


def test_complete_sharded_pack_does_not_touch_the_hub(tmp_path):
    d = pack(tmp_path, "qwen4_exp", files=("s1.safetensors",))
    (d / ka.INDEX).write_text(json.dumps({"weight_map": {"a": "s1.safetensors"}}))
    hub = FakeHub()
    ka.build(env_for(tmp_path), hub)
    assert hub.calls == []


def test_pack_without_weights_is_incomplete(tmp_path):
    pack(tmp_path, "qwen4_exp", files=())
    hub = FakeHub()
    ka.build(env_for(tmp_path), hub)
    assert [c[0] for c in hub.calls] == ["size", "download"]


def test_unreadable_index_is_incomplete(tmp_path):
    d = pack(tmp_path, "qwen4_exp"); (d / ka.INDEX).write_text("nope")
    assert ka.missing_files(d)[0].startswith(ka.INDEX)


def test_download_forbidden(tmp_path):
    hub = FakeHub()
    with pytest.raises(ka.ConfigError, match="KYOJIN_DOWNLOAD=0"):
        ka.build(env_for(tmp_path, KYOJIN_DOWNLOAD="0"), hub)
    assert hub.calls == []


def test_download_into_read_only_mount(tmp_path):
    ro = tmp_path / "ro"; ro.mkdir(); ro.chmod(0o555)
    try:
        with pytest.raises(ka.ConfigError, match="not writable"):
            ka.build(env_for(tmp_path, MODEL_DIR=str(ro)), FakeHub())
    finally:
        ro.chmod(0o755)


def test_hub_unreachable_is_a_clear_error(tmp_path):
    with pytest.raises(ka.ConfigError, match="cannot be read on the hub"):
        ka.build(env_for(tmp_path), FakeHub(fail="size"))


def test_download_failure_says_how_to_resume(tmp_path):
    with pytest.raises(ka.ConfigError, match="start again to resume"):
        ka.build(env_for(tmp_path), FakeHub(fail="download"))


def test_not_enough_disk(tmp_path):
    with pytest.raises(ka.ConfigError, match="GB free"):
        ka.build(env_for(tmp_path), FakeHub(size=1e18))


def test_download_that_leaves_pack_incomplete(tmp_path):
    class Half(FakeHub):
        def download(self, repo, dest):
            super().download(repo, dest)
            (Path(dest) / ka.INDEX).write_text(json.dumps({"weight_map": {"a": "gone.safetensors"}}))
    with pytest.raises(ka.ConfigError, match="still incomplete"):
        ka.build(env_for(tmp_path), Half())


def test_hub_downloader_is_lazy():
    ka.HubDownloader()  # building it must not import or contact the hub


# ---- options

def test_port_context_options(tmp_path):
    pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build(env_for(tmp_path, PORT="9000", HOST="127.0.0.1", CONTEXT="32768", KYOJIN_DEFAULT_TEMPERATURE="0.6",
                               KYOJIN_NO_VISION="1", KYOJIN_DRAFT_TOKENS="2", KYOJIN_REASONING_EFFORT="low", KYOJIN_NO_THINKING="0"))
    assert argv[argv.index("--port") + 1] == "9000" and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("-c") + 1] == "32768"
    assert "--no-vision" in argv and "--no-thinking" not in argv
    assert argv[argv.index("--ndt") + 1] == "2" and argv[argv.index("--default-temperature") + 1] == "0.6"
    assert argv.count("--default-temperature") == 1


def test_prefixed_aliases(tmp_path):
    pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build(env_for(tmp_path, KYOJIN_PORT="9100", KYOJIN_CONTEXT="4096", KYOJIN_HOST="::1", PORT="1", CONTEXT="2"))
    assert argv[argv.index("--port") + 1] == "9100" and argv[argv.index("-c") + 1] == "4096" and argv[argv.index("--host") + 1] == "::1"


def test_prefixed_model_dir(tmp_path):
    p = pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build({"KYOJIN_MODEL": "qwen", "KYOJIN_MODEL_DIR": str(p)})
    assert argv[2] == str(p)


def test_glm_draft_flag_name(tmp_path):
    pack(tmp_path, "glm5_next")
    argv, _ = ka.build(env_for(tmp_path, "glm", KYOJIN_DRAFT_TOKENS="3"))
    assert argv[argv.index("--num-draft") + 1] == "3" and "--ndt" not in argv


def test_unsupported_option_is_an_error(tmp_path):
    pack(tmp_path, "glm5_next")
    with pytest.raises(ka.ConfigError, match="not supported by the glm"):
        ka.build(env_for(tmp_path, "glm", KYOJIN_NO_VISION="1"))
    with pytest.raises(ka.ConfigError, match="not supported by the glm"):
        ka.build(env_for(tmp_path, "glm", KYOJIN_DEFAULT_TOP_K="40"))


def test_extra_args_passthrough_keeps_quotes(tmp_path):
    pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build(env_for(tmp_path, KYOJIN_EXTRA_ARGS="--default-max-tokens 512 --model-id 'my model'"))
    assert argv[-4:] == ["--default-max-tokens", "512", "--model-id", "my model"]


@pytest.mark.parametrize("env,msg", [({"CONTEXT": "abc"}, "CONTEXT"), ({"CONTEXT": "0"}, "CONTEXT"), ({"PORT": "x"}, "PORT"),
                                     ({"KYOJIN_NO_VISION": "maybe"}, "expected 1 or 0")])
def test_bad_values(tmp_path, env, msg):
    pack(tmp_path, "qwen4_exp")
    with pytest.raises(ka.ConfigError, match=msg):
        ka.build(env_for(tmp_path, **env))


# ---- positive switches

@pytest.mark.parametrize("env,flags", [
    ({}, []),
    ({"KYOJIN_ENABLE_VISION": "1"}, []),
    ({"KYOJIN_ENABLE_VISION": "0"}, ["--no-vision"]),
    ({"KYOJIN_ENABLE_VISION": "false"}, ["--no-vision"]),
    ({"KYOJIN_NO_VISION": "1"}, ["--no-vision"]),
    ({"KYOJIN_NO_VISION": "0"}, []),
    ({"KYOJIN_ENABLE_THINKING": "0"}, ["--no-thinking"]),
    ({"KYOJIN_ENABLE_UNCENSOR": "0"}, ["--no-uncensor"]),
    ({"KYOJIN_ENABLE_VISION": "1", "KYOJIN_NO_VISION": "0"}, []),
    ({"KYOJIN_ENABLE_VISION": "0", "KYOJIN_NO_VISION": "1"}, ["--no-vision"]),
])
def test_qwen_switches(tmp_path, env, flags):
    pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build(env_for(tmp_path, **env))
    assert [f for f in ("--no-vision", "--no-thinking", "--no-uncensor") if f in argv] == flags


@pytest.mark.parametrize("env", [{"KYOJIN_ENABLE_VISION": "1", "KYOJIN_NO_VISION": "1"}, {"KYOJIN_ENABLE_VISION": "0", "KYOJIN_NO_VISION": "0"}])
def test_contradicting_switches(tmp_path, env):
    pack(tmp_path, "qwen4_exp")
    with pytest.raises(ka.ConfigError, match="contradict"):
        ka.build(env_for(tmp_path, **env))


def test_enable_vision_is_ignored_where_the_server_has_none(tmp_path):
    """His llama-swap line sets KYOJIN_ENABLE_VISION=1 for every model; GLM and MiMo have no vision tower."""
    pack(tmp_path, "glm5_next")
    argv, notes = ka.build(env_for(tmp_path, "glm", KYOJIN_ENABLE_VISION="1"))
    assert "--no-vision" not in argv and any("ignored" in n for n in notes)
    with pytest.raises(ka.ConfigError, match="not supported by the glm"):
        ka.build(env_for(tmp_path, "glm", KYOJIN_ENABLE_VISION="0"))


# ---- command line of the script

def test_cli_null_output_and_errors(tmp_path, monkeypatch, capsys):
    pack(tmp_path, "qwen4_exp")
    monkeypatch.setattr(sys, "argv", ["x", "--null"])
    for k, v in env_for(tmp_path).items():
        monkeypatch.setenv(k, v)
    assert ka.main() == 0
    cap = capsys.readouterr()
    assert cap.out.split("\0")[0] == "tools/qwen/serve.py" and cap.out.endswith("\0") and "family=qwen" in cap.err
    monkeypatch.setenv("CONTEXT", "x")
    assert ka.main() == 2
    cap = capsys.readouterr()
    assert cap.out == "" and "configuration error" in cap.err


def test_cli_plain_output(tmp_path, monkeypatch, capsys):
    pack(tmp_path, "qwen4_exp")
    monkeypatch.setattr(sys, "argv", ["x"])
    for k, v in env_for(tmp_path).items():
        monkeypatch.setenv(k, v)
    assert ka.main() == 0
    assert capsys.readouterr().out.startswith("python tools/qwen/serve.py --model ")
