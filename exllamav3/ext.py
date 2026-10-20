from __future__ import annotations
import importlib.machinery
import importlib.util
import torch
from torch.utils.cpp_extension import load
import os
import sys
from .util.arch_list import maybe_set_arch_list_env
from .util.cuda_flags import cuda_cflags, extension_sources, hip_include_flags, patch_hip_ninja_file_writer, use_rocm_sdk_devel

extension_name = "exllamav3_ext"
verbose = False  # Print wall of text when compiling
ext_debug = False  # Compile with debug options

# Determine if we're on Windows

windows = (os.name == "nt")

# Determine if extension is already installed or needs to be built

def is_precompiled_extension_available():
    spec = importlib.util.find_spec(extension_name)
    if not spec or not spec.origin or not spec.loader:
        return False
    return any(
        spec.origin.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

if is_precompiled_extension_available():
    import exllamav3_ext
else:

    # Kludge to get compilation working on Windows

    if windows:

        def find_msvc():

            # Possible locations for MSVC, in order of preference

            program_files_x64 = os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", r"C:\Program Files"))
            program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

            msvc_dirs = \
            [
                a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\\Tools\\MSVC\\"
                for b in ["2022", "2019", "2017"]
                for a in [program_files_x64, program_files_x86]
                for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
            ]

            for msvc_dir in msvc_dirs:
                if not os.path.exists(msvc_dir): continue

                # Prefer the latest version

                versions = sorted(os.listdir(msvc_dir), reverse = True)
                for version in versions:

                    compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                    if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                        return compiler_dir

            # No path found

            return None

        import subprocess

        # Check if cl.exe is already in the path

        try:

            subprocess.check_output(["where", "/Q", "cl"])

        # If not, try to find an installation of Visual Studio and append the compiler dir to the path

        except subprocess.CalledProcessError as e:

            cl_path = find_msvc()
            if cl_path:
                if verbose:
                    print(" -- Injected compiler path:", cl_path)
                os.environ["path"] += ";" + cl_path
            else:
                print(" !! Unable to find cl.exe; compilation will probably fail", file = sys.stderr)

    # compiler flags

    extra_cflags = []
    extra_cuda_cflags = cuda_cflags(
        cuda_home = torch.utils.cpp_extension.CUDA_HOME,
        debug = ext_debug,
        hip = bool(torch.version.hip),
    )

    if windows:
        # TODO: preprocessor and lean_and_mean flags are needed for Windows cu132 build, verify that they don't break
        #       older cu128 builds
        # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
        # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
        # Defined globally so it holds regardless of include order in any TU (mirrors setup.py).
        extra_cflags += ["/Ox", "/Zc:preprocessor", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX"]
        extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX", "-Xcompiler=/Zc:preprocessor"]
        if ext_debug:
            extra_cflags += ["/Zi"]
            extra_cuda_cflags += []
    elif torch.version.hip:
        # torch hands the C++ flags to hipcc as well, and -Ofast implies fast-math (see hip_cflags)
        extra_cflags += ["-O3"]
    else:
        extra_cflags += ["-Ofast"]
        extra_cuda_cflags += []
        if ext_debug:
            extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
            extra_cuda_cflags += []

    # Windows: torch's JIT runs bare cl for the C++ sources and never passes -ccbin, so keep nvcc on the same cl.exe.
    # ROCm: CUDAHOSTCXX belongs to nvcc; hipcc takes no -ccbin
    if not windows and not torch.version.hip and (cuda_host_cxx := os.environ.get("CUDAHOSTCXX")):
        extra_cuda_cflags += ["-ccbin", cuda_host_cxx]
    elif windows and os.environ.get("CUDAHOSTCXX"):
        print(
            " !! CUDAHOSTCXX is not used by the JIT build on Windows; "
            "nvcc uses the same cl.exe as the C++ sources (see doc/env_vars.md)",
            file = sys.stderr
        )

    if verbose and not torch.version.hip:
        extra_cuda_cflags += ["--ptxas-options=-v"]

    # linker flags

    extra_ldflags = []

    if windows:
        extra_ldflags += ["cublas.lib"]
        if sys.base_prefix != sys.prefix:
            extra_ldflags += [f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]
    elif torch.version.hip:
        # The ROCm extension calls hipBLAS directly (hgemm.cu, graph.cu) and rocBLAS for the hgemm algo tuner
        extra_ldflags += ["-lhipblas", "-lrocblas"]

    # sources

    library_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(library_dir, extension_name)
    sources = extension_sources(sources_dir, hip = bool(torch.version.hip))
    if torch.version.hip:
        # With no target list, torch builds for every architecture it supports, wave64 ones included, which
        # the kernels cannot compile for. maybe_set_rocm_arch_env fills it in from the visible devices, so an
        # empty list here means HIP sees none (CUDA_VISIBLE_DEVICES is honored by HIP as well)
        if not os.environ.get("PYTORCH_ROCM_ARCH"):
            raise RuntimeError(
                "No ROCm device is visible to build the extension for. Check CUDA_VISIBLE_DEVICES and "
                "HIP_VISIBLE_DEVICES (HIP honors both), or set PYTORCH_ROCM_ARCH (e.g. gfx1100) explicitly."
            )
        use_rocm_sdk_devel(torch.utils.cpp_extension)
        extra_cflags += hip_include_flags(sources_dir)
        extra_cuda_cflags += hip_include_flags(sources_dir)
        # Flags hipcc would add from its environment go on the command line instead, where the ccache wrapper
        # (hip_compiler_wrapper) hashes them; left to hipcc, a changed define is a cache hit on the old object
        extra_cuda_cflags += os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", "").split()

    # Load extension

    maybe_set_arch_list_env()
    if torch.version.hip:
        # Dependency files for the HIP compile rule (header edits rebuild their includers) and the
        # ccache/sccache wrapper torch applies to CUDA builds only
        patch_hip_ninja_file_writer(torch.utils.cpp_extension)
    try:
        exllamav3_ext = load(
            name = extension_name,
            sources = sources,
            extra_include_paths = [sources_dir],
            verbose = verbose,
            extra_ldflags = extra_ldflags,
            extra_cuda_cflags = extra_cuda_cflags,
            extra_cflags = extra_cflags
        )
    except IndexError as e:
        # With no list given (or "native"), torch derives the architectures from the visible GPUs and
        # fails with an IndexError when there are none
        if torch.version.cuda and os.environ.get("TORCH_CUDA_ARCH_LIST", "") in ("", "native") and \
                not torch.cuda.device_count():
            raise RuntimeError(
                f"No CUDA device is visible to determine the architectures to build {extension_name} for. "
                "Set TORCH_CUDA_ARCH_LIST to the target architectures, e.g. TORCH_CUDA_ARCH_LIST=\"8.6;8.9+PTX\""
            ) from e
        raise


# Kyojin's ROCm module paths (hip_gemv, fused MoE, GDN/KDA HIP chains) are taken only when a module has no
# bound C++ block (self.bc is None), so the BC_* classes stay masked on ROCm unless EXL3_HIP_BC=1. BC_SAM has
# no Python-side equivalent and stays native.

if torch.version.hip:
    from . import ext_fallbacks as _fb

    if os.environ.get("EXL3_HIP_BC", "0") != "1":
        _bc_none = _fb._BCNone()
        for _name in [
            'BC_Mamba2', 'BC_GatedDeltaNet', 'BC_GatedDeltaNetSplit',
            'BC_MLP', 'BC_GatedMLP', 'BC_BlockSparseMLP',
            'BC_Attention', 'BC_GatedRMSNorm',
            'BC_LinearEXL3', 'BC_LinearFP16',
            'BC_DSV4Compressor', 'BC_DSV4Attention', 'BC_DSV4BatchAttention',
            'BC_MLAttention',
        ]:
            setattr(exllamav3_ext, _name, _bc_none)

    # r35: runtime-mutable knobs (in-process A/B)
    ROCM_KNOBS = {"silu_lim": os.environ.get("EXL3_FUSE_SILU_LIM", "1") != "0",
                  # fp16 gate (KDA z under EXL3_KDA_F16_GATES) into the fused gated norm; "0" = the torch
                  # fallback chain (r2 glm-next: cost 4K prefill -2.5 %, the F16_GATES "loss")
                  "gnorm_f16g": os.environ.get("EXL3_FUSE_GNORM_F16G", "1") != "0"}

    # Native RMSNorm (exl3_dec.cu) for the common shapes; everything else goes to the upstream kernel (norm.cu)
    if hasattr(exllamav3_ext, "exl3_dec_rms_norm") and os.environ.get("EXL3_DEC_NORM", "1") != "0":
        _fb_rms_norm = exllamav3_ext.rms_norm
        _fb_rms_norm_res_in = exllamav3_ext.rms_norm_res_in
        _norm_dt = (torch.float16, torch.float32)

        def _w_ok(w, dim):
            return w is None or (w.dtype in (torch.float16, torch.bfloat16) and w.numel() == dim and w.is_contiguous())

        def _rms_norm_rocm(x, w, y, eps, constant_bias, constant_scale, span_heads, add_residual, w_groups = 1):
            if (not span_heads and w_groups == 1 and x.is_cuda and x.is_contiguous() and y.is_contiguous() and
                    x.shape == y.shape and x.dtype in _norm_dt and y.dtype in _norm_dt and
                    x.shape[-1] <= 8192 and _w_ok(w, x.shape[-1])):
                exllamav3_ext.exl3_dec_rms_norm(x, w, y, None, eps, constant_bias, constant_scale,
                                                1 if add_residual else 0)
                return
            _fb_rms_norm(x, w, y, eps, constant_bias, constant_scale, span_heads, add_residual, w_groups)

        def _rms_norm_res_in_rocm(x, w, y, r, eps, constant_bias, constant_scale):
            if (x.is_cuda and x.is_contiguous() and y.is_contiguous() and r.is_contiguous() and
                    x.shape == y.shape == r.shape and x.dtype in _norm_dt and r.dtype in _norm_dt and
                    y.dtype == torch.float16 and x.shape[-1] <= 8192 and _w_ok(w, x.shape[-1])):
                exllamav3_ext.exl3_dec_rms_norm(x, w, y, r, eps, constant_bias, constant_scale, 2)
                return
            _fb_rms_norm_res_in(x, w, y, r, eps, constant_bias, constant_scale)

        setattr(exllamav3_ext, "rms_norm", _rms_norm_rocm)
        setattr(exllamav3_ext, "rms_norm_res_in", _rms_norm_res_in_rocm)

    # Fused elementwise ops (fused_elt_rocm.cu): one launch instead of the fallback's torch chain
    if hasattr(exllamav3_ext, "fused_gated_rms_norm") and os.environ.get("EXL3_FUSE_GNORM", "1") != "0":
        _native_gated_rms_norm = exllamav3_ext.gated_rms_norm

        def _fb_gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act):
            if (x.dtype == torch.bfloat16 and w.dtype in (torch.float32, torch.bfloat16) and
                    y.dtype in (torch.float16, torch.float32) and g.dtype in (torch.bfloat16, torch.float32) and
                    x.is_contiguous() and w.is_contiguous() and y.is_contiguous() and g.is_contiguous() and
                    x.shape[-1] % 4 == 0):
                _native_gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act)
            else:
                _fb.gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act)

        def _gated_rms_norm_rocm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act = 0, y_pitch = 0):
            if (x.is_cuda and x.dtype == torch.bfloat16 and y.dtype in (torch.float16, torch.float32) and
                    w.dtype in (torch.bfloat16, torch.float32) and
                    (g.dtype in (torch.bfloat16, torch.float32) or
                     (g.dtype == torch.float16 and ROCM_KNOBS["gnorm_f16g"])) and
                    x.is_contiguous() and (y.is_contiguous() or (y_pitch and y.dtype == torch.float16 and x.dim() >= 3)) and g.is_contiguous() and w.is_contiguous() and
                    x.shape == y.shape == g.shape and x.shape[-1] <= 512 and x.shape[-1] % 128 == 0 and w_groups >= 1 and
                    w.numel() == w_groups * x.shape[-1] and gate_act in (0, 1)):
                exllamav3_ext.fused_gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act,
                                                   y_pitch, x.shape[-2] if y_pitch else 1)
                return
            assert not y_pitch, "padded gated-norm output needs the fused kernel"
            _fb_gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act)

        def _gn_pad_ok(x, w, y_dtype, g, w_groups, gate_act = 0):
            """True when _gated_rms_norm_rocm can write a row-padded fp16 y: the fused kernel's own conditions."""
            return (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() >= 3 and y_dtype == torch.float16 and
                    w.dtype in (torch.bfloat16, torch.float32) and
                    (g.dtype in (torch.bfloat16, torch.float32) or
                     (g.dtype == torch.float16 and ROCM_KNOBS["gnorm_f16g"])) and
                    x.is_contiguous() and g.is_contiguous() and w.is_contiguous() and x.shape == g.shape and
                    x.shape[-1] <= 512 and x.shape[-1] % 128 == 0 and w_groups >= 1 and
                    w.numel() == w_groups * x.shape[-1] and gate_act in (0, 1))
        _gated_rms_norm_rocm.pad_ok = _gn_pad_ok
        setattr(exllamav3_ext, "gated_rms_norm", _gated_rms_norm_rocm)

    # deinterleave_qg: hand HIP copy kernel (modules/ple_fn/ple_hip.hip), pure copy so bitwise equal; EXL3_DQ_HIP=1
    if True:  # flag read per call so one process can A/B it
        _fb_deinterleave_qg = exllamav3_ext.deinterleave_qg

        def _deinterleave_qg_rocm(qg, q, g, head_dim):
            if (os.environ.get("EXL3_DQ_HIP", "0") == "1" and head_dim % 8 == 0 and qg.is_cuda and qg.dtype == torch.float16 and q.dtype == torch.float16
                    and g.dtype == torch.float16 and qg.is_contiguous() and q.is_contiguous() and g.is_contiguous()
                    and q.numel() == g.numel() and q.numel() * 2 == qg.numel() and q.numel() > 0
                    and qg.data_ptr() % 16 == 0 and q.data_ptr() % 16 == 0 and g.data_ptr() % 16 == 0
                    and qg.numel() % (2 * head_dim) == 0):
                try:
                    from .modules.ple_fn import ple_hip
                    ple_hip.deinterleave_qg(qg, q, g, head_dim)
                    return
                except Exception:
                    pass
            _fb_deinterleave_qg(qg, q, g, head_dim)

        setattr(exllamav3_ext, "deinterleave_qg", _deinterleave_qg_rocm)

    if hasattr(exllamav3_ext, "fused_silu_mul") and os.environ.get("EXL3_FUSE_SILU", "1") != "0":
        _fb_silu_mul = exllamav3_ext.silu_mul

        def _silu_mul_rocm(x, y, z, act_limit = 0.0):
            if ((act_limit == 0.0 or ROCM_KNOBS["silu_lim"]) and x.is_cuda and
                    x.dtype in (torch.float16, torch.float32) and
                    y.dtype == x.dtype and z.dtype == torch.float16 and x.is_contiguous() and
                    y.is_contiguous() and z.is_contiguous() and x.shape == y.shape == z.shape):
                exllamav3_ext.fused_silu_mul(x, y, z, float(act_limit))
                return
            _fb_silu_mul(x, y, z, float(act_limit))

        setattr(exllamav3_ext, "silu_mul", _silu_mul_rocm)

    # Keep the public API stable while using the native QSA kernel (k <= 512, moeA5) only for its
    # validated gfx12 wave32 shape. Everything else retains the general PyTorch fallback.
    if hasattr(exllamav3_ext, "dsa_topk_gfx12"):
        _native_dsa_topk = exllamav3_ext.dsa_topk
        # EXL3_DSA_TOPK_V2=1 (default off; env read per call): the row-in-registers kernel (dsa_topk_gfx12_v2, one memory
        # pass instead of four, bitwise-identical selection) for rows of 8192 < width <= 32768 with a 16-byte aligned base
        # (narrower rows are not faster: 82 vs 115 us at 2048). topk_v2_switch['on'] lets a harness flip it without the env.
        # EXL3_DSA_TOPK_V2_THREADS = 512 | 1024.
        topk_v2 = {"on": False, "threads": int(os.environ.get("EXL3_DSA_TOPK_V2_THREADS", "512"))}
        setattr(exllamav3_ext, "topk_v2_switch", topk_v2)

        def _dsa_topk_rocm(scores, indices, k, t_ptr = None, t_seq = 0):
            if (
                hasattr(exllamav3_ext, "dsa_topk_gfx12") and
                _fb.dsa_topk_gfx12_supported(scores, indices, k, t_ptr, t_seq)
            ):
                env_v2 = os.environ.get("EXL3_DSA_TOPK_V2")      # read per call so one process can A/B it
                use_v2 = topk_v2["on"] if env_v2 is None else env_v2 == "1"
                if (use_v2 and 8192 < scores.shape[1] <= 32768 and scores.data_ptr() % 16 == 0 and
                        hasattr(exllamav3_ext, "dsa_topk_gfx12_v2")):
                    return exllamav3_ext.dsa_topk_gfx12_v2(scores, indices, k, topk_v2["threads"])
                return exllamav3_ext.dsa_topk_gfx12(scores, indices, k)
            return _native_dsa_topk(scores, indices, k, t_ptr, t_seq)

        setattr(exllamav3_ext, "dsa_topk", _dsa_topk_rocm)

    if os.environ.get("EXL3_QWRITE_TRITON", "1") != "0":
        _native_quant_cache_paged = exllamav3_ext.quant_cache_paged

        def _quant_cache_paged_rocm(k_in, k_out, k_out_scales, v_in, v_out, v_out_scales,
                                    cache_seqlens, block_table, page_size, seq_len, compand_a = 0.0, in_contiguous = False):
            if (
                compand_a == 0.0 and seq_len > 0 and k_in.is_cuda and
                k_out.ndim == 3 and v_out.ndim == 3 and k_out_scales.shape[-1] in (8, 16, 32, 64) and
                k_out.shape[-1] == k_out_scales.shape[-1] * 8 and v_out.shape[-1] == k_out.shape[-1] and
                v_out_scales.shape == k_out_scales.shape and
                k_in.dtype == torch.float16 and v_in.dtype == torch.float16 and
                k_in.is_contiguous() and v_in.is_contiguous() and page_size == 256 and
                k_out.dtype == torch.int32 and v_out.dtype == torch.int32 and
                k_out_scales.dtype == torch.float16 and v_out_scales.dtype == torch.float16 and
                cache_seqlens.dtype == torch.int32 and block_table.dtype == torch.int32 and
                k_in.shape == v_in.shape and
                k_in.numel() >= block_table.shape[0] * seq_len * (k_out_scales.shape[-1] * 32) and
                (in_contiguous or k_in.numel() == k_out.shape[0] * page_size * k_out_scales.shape[-1] * 32)
            ):
                from .modules.attention_fn.qwrite_triton import quant_cache_paged8
                quant_cache_paged8(k_in, k_out, k_out_scales, v_in, v_out, v_out_scales,
                                   cache_seqlens, block_table, seq_len, in_contiguous)
                return
            _native_quant_cache_paged(k_in, k_out, k_out_scales, v_in, v_out, v_out_scales,
                                      cache_seqlens, block_table, page_size, seq_len, compand_a, in_contiguous)

        setattr(exllamav3_ext, "quant_cache_paged", _quant_cache_paged_rocm)
