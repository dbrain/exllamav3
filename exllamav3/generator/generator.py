from __future__ import annotations
import logging
import os
import time
import torch
try:
    from ..modules.moe_fused import check_barrier as _moe_fused_check   # fused MoE half-layer grid barrier: raise if it ever timed out
except Exception:                                                    # noqa: BLE001
    _moe_fused_check = None
from ..model.model import Model
from ..cache.cache import Cache
from ..cache.recurrent import RecurrentCache, PairedState, host_pool, mp_host_pool_release
from ..modules.gated_delta_net import GDNState
from ..tokenizer.tokenizer import Tokenizer
from ..constants import PAGE_SIZE
from ..util import cuda_sync_active
from ..ext import exllamav3_ext as ext

logger = logging.getLogger(__name__)
_PADDED_VOCAB_LOGGED = set()


def _log_padded_vocab(model_vocab: int, piece_count: int):
    """One start-up line per process and vocabulary shape: the head is wider than the tokenizer's table."""
    if (model_vocab, piece_count) in _PADDED_VOCAB_LOGGED:
        return
    _PADDED_VOCAB_LOGGED.add((model_vocab, piece_count))
    logger.warning(
        "vocabulary: the model head has %d columns but the tokenizer defines %d pieces; ids %d..%d are padding. "
        "They are masked at sampling and in the drafters; a padding id that still got through decodes to empty text",
        model_vocab, piece_count, piece_count, model_vocab - 1
    )


from ..util.memory import malloc_trim
from .pagetable import PageTable, is_content_hash
from .cpu_cache import CPUPageCache
from .disk_cache import DiskPageCache
from .draft_confidence import DraftConfidenceCalibrator
from .job import Job, ScoreGroup, _ttrace_begin, _ttrace_end, _TAIL_REPLY as _TAIL_REPLY_GEN
# EXL3_REPLY_FIRST=1 (default off): a job that finishes inside iterate() is released (deallocate_pages: reply-end tail
# stash, draft flush, page / slot release) by flush_deferred() AFTER its last results are returned, not before. The server calls it
# once the results are handed out; every iterate / cancel / clear_queue runs it first, so later work sees the same cache state in
# the same order (exact by construction). Pays where the stash is a blocking copy (HIP .cpu(), CUDA EXL3_TAIL_PIN=0).
_REPLY_FIRST = os.environ.get("EXL3_REPLY_FIRST", "0") == "1"
from . import spec_sample
from .filter import Filter
from concurrent.futures import ThreadPoolExecutor
from .sampler import Sampler
from .visualizer import CacheVisualizer
import time
from ..util.device_copy import to_device
import threading
from ..tokenizer import MMEmbedding
from ..util import profile_opt
from ..ext import exllamav3_ext as ext

# EXL3_MTP_FUSE_CATCHUP: 0 = catch-up draft prefill after each accepted round (old path),
# 1 = fold the catch-up rows into the next draft forward, 2 = same with the R-row union MoE path.
# Unset = auto: 2 when the union MoE is on (EXL3_DEC_MOE_UNION or _DEV), else 0 (1 lost the step 5 A/B).
MTP_FUSE_CATCHUP = None if os.environ.get("EXL3_MTP_FUSE_CATCHUP") is None else \
    int(os.environ["EXL3_MTP_FUSE_CATCHUP"])

def _host_lean():
    """EXL3_HOST_LEAN=1: host-side decode-overhead trims (default off). Read per call so one
    loaded process can A/B arms at run time. All lean paths keep kernel launches, RNG draw
    order and numerics identical; they only batch host readbacks and reuse staging buffers."""
    return os.environ.get("EXL3_HOST_LEAN", "0") == "1"


def _mtp_fuse_catchup():
    if MTP_FUSE_CATCHUP is not None: return MTP_FUSE_CATCHUP
    on = lambda k: os.environ.get(k, "0") != "0"
    return 2 if on("EXL3_DEC_MOE_UNION") or on("EXL3_DEC_MOE_UNION_DEV") else 0

def block_table_pages(max_seq_len: int) -> int:
    """Block table width for a batch whose longest sequence reaches max_seq_len tokens: whole pages,
    padded to a multiple of 16 so the pinned staging buffers cover a few distinct widths only."""
    return ((max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE + 15) // 16 * 16


# Test hook: True forces the per-position serial speculative-window readback instead of the batched
# (launch-all-samplers, one pinned stage, one synchronize) path
_NO_BATCH_READBACK = False

class Generator:

    @staticmethod
    def _validate_ngram_drafting_support() -> None:
        try:
            sam = ext.BC_SAM()
        except AttributeError as exc:
            raise NotImplementedError(
                "N-gram drafting requires the BC_SAM extension binding, which is unavailable on this build."
            ) from exc
        if sam is None:
            raise NotImplementedError(
                "N-gram drafting requires BC_SAM, which is unavailable on this platform/build."
            )


    # Sparse drafting head and its per-round flag (set up in the constructor when enabled; class-level
    # defaults keep iterate() valid on stubbed instances)
    sparse_head = None
    _sparse_flag = False

    def __init__(
        self,
        model: Model,
        cache: Cache,
        tokenizer: Tokenizer,
        max_batch_size: int = 256,
        max_chunk_size: int | None = None,
        shared_chunk_size: int | None = None,
        prefill_cooldown: int | None = None,
        max_q_size: int = 8,
        draft_model: Model | None = None,
        draft_cache: Cache | None = None,
        num_draft_tokens: int | None = None,
        show_visualizer: bool = False,
        enable_defrag: bool = True,
        tail_ckpt: int | None = None,
        cpu_cache_size: int = 0,
        disk_cache_dir: str | None = None,
        disk_cache_size: int = 50 * 1024**3,
        disk_cache_flush_on_idle: bool = True,
        disk_cache_checkpoint_interval: int = 4096,
        disk_cache_write_interval: float = 2.0,
        recurrent_cache_size: int = 4 * 1024**3,
        recurrent_checkpoint_interval: int = None,
        recurrent_checkpoint_interval_pp: int = 32768,
        ngram_match_min: int = 0,
        dynamic_draft_tokens: bool = False,
        draft_confidence: float = 0.4,
        record_draft_stats: bool = False,
        ngram_corpus: str | None = None,
        hybrid_draft_tokens: int = 0,
        draft_max_rows: int = 0,
        sparse_draft_head: int = 0,
        **kwargs
    ):
        """
        Initialize generator

        :param model:
            The model (loaded)

        :param cache:
            Paged cache

        :param tokenizer:
            Tokenizer

        :param max_batch_size:
            The maximum number of sequences to process in parallel. The generator will also limit this
            dynamically considering the available cache space.

        :param max_chunk_size:
            Maximum number of tokens to process in parallel during prefill (prompt ingestion). Should not
            exceed the model's max_input_len but can be lowered to trade off prompt speed for a shorter
            interruption to ongoing jobs when a new job is started. Default: the model's
            caps["prefill_chunk_size"] if set, else 2048. See shared_chunk_size for lowering it
            only while other jobs are generating.

        :param shared_chunk_size:
            Prefill chunk size to use while another active job is already generating, so that a new prompt
            interrupts ongoing generation for shorter stretches at a time. Must be a multiple of the page size
            and no larger than max_chunk_size. Prompt ingestion with no job generating keeps the full
            max_chunk_size, which remains the efficient choice for cold prefill. Default is None, which selects
            half of max_chunk_size rounded up to the page size. Set it to max_chunk_size to keep full chunks
            while other jobs generate

        :param prefill_cooldown:
            Number of iterations to wait between prefill passes while another active job is generating. Every
            call to iterate() still runs one generation step, so the generating jobs keep streaming at an even
            pace while prompt ingestion takes one chunk every prefill_cooldown + 1 calls. With several prompts
            waiting, they take turns, one chunk per pass. Default is 4. Set to 0 for a prefill pass on every
            iteration, as when no job is generating

        :param max_q_size:
            Maximum number of tokens to evaluate per sequence during generation. Leave this at the default
            (8) unless there's a good reason to increase it.

        :param draft_model:
            Draft model. Enables speculative decoding with draft, and must be specified along with
            draft_cache. Note that speculative decoding with many parallel jobs is likely not advantageous.

        :param draft_cache:
            Cache allocated for draft model. Must be same size as main cache.

        :param num_draft_tokens:
            Number of future tokens to draft. Default is 4 unless the draft model has a preference (e.g.
            from DFlash block size)

        :param ngram_match_min:
            Minimum number of tokens to match for n-gram draft (0 = disabled).

        :param ngram_corpus:
            Optional frozen SAM file shared by n-gram jobs; requires ngram_match_min > 0.

        :param hybrid_draft_tokens:
            Long suffix-match drafting on top of whichever drafting mode is active (or on its own). Every
            round, each job's suffix automaton is consulted; when the longest match of the current suffix
            against the job's earlier context is at least this many tokens, the draft becomes the same number
            of tokens that followed that earlier occurrence, unless they disagree with the regular drafter's
            own guess. The intent is to catch verbatim repetition of context (reciting a drafted reply, editing
            source) and gamble a long verification pass on advancing many tokens at prefill speed. A recurrent
            model's cache needs max_history of at least this many tokens.

        :param draft_max_rows:
            Ceiling on the number of decoding rows in a speculative round, counting the verified draft
            positions, i.e. bsz * (1 + draft tokens) (0 disables the ceiling). Draft length is capped when
            possible to keep the batch from exceeding the ceiling.

        :param sparse_draft_head:
            Number of vocabulary columns in a dynamic sparse LM head for drafters that borrow the target's
            lm_head (MTP, DFlash), 0 to disable. The subset follows the frequencies of the tokens seen in
            prompts and generated text; verification always uses the full head, so output is unaffected.

        :param dynamic_draft_tokens:
            Adapt the per-round draft length to the workload. The draft is cut using a drafter-provided
            confidence score, calibrated online against observed acceptance rates (see draft_confidence). A
            DFlash drafter still runs at its fixed diffusion block size and only the verified window shrinks;
            AR and MTP drafters stop the drafting loop itself early, saving one drafter forward per pruned
            position. The acceptance behavior of the surviving positions is unchanged. Has no effect on n-gram
            drafting, whose draft length already adapts through the match length.

        :param draft_confidence:
            Used with dynamic_draft_tokens and a draft model: target acceptance probability for drafted
            positions, evaluated against an online mapping from drafter confidence scores to observed
            acceptance rates. Scale-free and portable across model pairs. For DFlash (block produced at fixed
            cost) the block is cut at the first position whose estimated conditional acceptance drops below
            the target. For AR and MTP drafters (one forward per position, and a position only pays off if
            everything before it is accepted too) drafting stops when the running product of conditional
            estimates - the probability that the next position actually contributes a token - falls below the
            target. Lower values keep longer drafts; higher values truncate more aggressively. Ignored for
            n-gram drafting.

        :param record_draft_stats:
            Append (position, window, accepted) per verification round to job.draft_stats, for analysis.

        :param show_visualizer:
            Open window to render visualization of cache (for debug/demonstration purposes)

        :param enable_defrag:
            Defragment cache periodically

        :param cpu_cache_size:
            Size in bytes of a second-tier page cache in pinned system memory, 0 (default) to disable. Complete
            K/V pages evicted from the GPU cache are stored there and restored on prompt-cache hits instead of
            being recomputed by prefill. In tensor-parallel mode each rank pins its own shard of the tier and
            the budget counts whole pages across all of them

        :param disk_cache_dir:
            Directory for a third, persistent tier behind the CPU page cache (requires cpu_cache_size > 0),
            None (default) to disable. Pages entering the CPU tier and recurrent checkpoints are written
            through to a per-model subdirectory (keyed by a fingerprint of the model, cache configuration and
            kernel sources) and read back, in this or a later process, when a prompt's pages are in no other
            tier. Page images and checkpoints are stored in a layout independent of the tensor-parallel
            split, so a directory is shared between TP loads with different splits and non-TP loads of the
            same model and cache configuration

        :param disk_cache_size:
            Budget in bytes for the whole disk_cache_dir, shared by every model and cache configuration
            stored under it (the least recently used of the others are removed whole when a load finds the
            root over budget). Default is 50 GB

        :param disk_cache_flush_on_idle:
            Also write every complete page still in VRAM (and every checkpoint) to the disk tier whenever the
            generator goes idle, so a context never has to be evicted first to be persisted. Default True;
            flush_cache_to_disk() does the same on demand

        :param disk_cache_checkpoint_interval:
            Minimum spacing, in tokens, between the recurrent checkpoints the disk tier keeps along any chain
            of cached pages, counted from the start of the chain; closer ones (and checkpoints whose page is
            gone) are pruned when the generator goes idle, so a resumed prompt replays at most this many
            tokens. Checkpoints take far more space than pages on hybrid models. Default 4096; 0 keeps all

        :param disk_cache_write_interval:
            While jobs run, each this many seconds, queue the complete pages of the active sequences for the disk
            tier (through the CPU tier). Without this, a page goes to disk only when it leaves VRAM or when the
            generator becomes idle, so a process that stops during a long prefill loses all of it. With this, it
            loses only the pages after the last checkpoint on disk. The last two pages before the position of
            each sequence are not queued: speculative decoding can still rewind them, and the MTP draft cache
            writes the last slot of a page one step later. 0 disables. Default 2.0

        :param recurrent_cache_size:
            Size of recurrent cache, in bytes. Recurrent cache resides in system RAM. Default is 4 GB.
            Ignored if model doesn't use recurrent states

        :param recurrent_checkpoint_interval:
            Minimum number of tokens between recurrent checkpoints in model output and the tail end of
            the prompt. Must be a multiple of the page size. Model architecture determines default

        :param recurrent_checkpoint_interval_pp:
            Minimum number of tokens between recurrent checkpoints during prompt ingestion. Must be a
            multiple of the page size. Checkpoints land on the first chunk boundary at least this far past the
            previous one, so the interval need not be a multiple of the chunk size. Default is 32768 tokens

        :param kwargs:
        """

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        cfg = self.model.config
        self.padded_vocab_size = ((cfg.vocab_size + 31) // 32) * 32

        # Padded head: the logits (and the embedding table) can be wider than the vocabulary the tokenizer defines, and the
        # ids in between have no piece. CustomSampler masks them, the drafters are bounded to vocab_limit (_bound_draft_ids)
        # and Job.receive_sample decodes an unknown id to empty text (note_unknown_token logs the first one)
        self.vocab_limit = None
        self._unknown_token_seen = False
        piece_count = getattr(tokenizer, "actual_vocab_size", None)
        if isinstance(piece_count, int) and 0 < piece_count < cfg.vocab_size:
            self.vocab_limit = piece_count
            _log_padded_vocab(cfg.vocab_size, piece_count)

        # Paging
        self.pagetable = PageTable(self, cache)
        # A cache has exactly one owning generator. Record the takeover and refuse to iterate a superseded generator
        cache.owner_serial = getattr(cache, "owner_serial", 0) + 1
        self.cache_owner_serial = cache.owner_serial
        self.max_total_tokens = PAGE_SIZE * self.pagetable.max_pages
        self._cache_stats_memo = (None, None)

        # Draft model
        self.draft_model = draft_model
        self.draft_cache = draft_cache
        if draft_model is not None:
            draft_model.vocab_limit = self.vocab_limit    # read by the MTP heads (sample_from_state); None = whole head
        if draft_model:
            assert not ngram_match_min, \
                "Cannot use both draft model and n-gram draft."
            assert draft_cache is not None, \
                "Must supply cache for draft model"
            assert draft_cache.max_num_tokens == cache.max_num_tokens, \
                "Cache and draft cache must be same size"
            if draft_model.caps.get("recurrent_states"):
                # An autoregressive drafter with recurrent layers drafts through a token-by-token
                # speculative pass on its own state pool (iterate_draftmodel_gen); a DFlash-family
                # drafter's sliding-window rings take the projected rows directly. MTP heads with
                # recurrent layers have neither path yet
                assert not draft_model.caps.get("mtp_draft"), "Recurrent MTP draft models are not supported."
            if num_draft_tokens:
                self.num_draft_tokens = num_draft_tokens
            else:
                self.num_draft_tokens = draft_model.caps.get("default_draft_size", 4)
            depths = draft_model.caps.get("mtp_depths")
            if depths is not None and self.num_draft_tokens > depths:
                print(f" !! Warning: the MTP head has {depths} depth-specialized layers; draft positions past "
                      f"{depths} reuse the last one, with decreasing acceptance (num_draft_tokens = "
                      f"{self.num_draft_tokens})")
        elif ngram_match_min:
            self._validate_ngram_drafting_support()
            self.num_draft_tokens = num_draft_tokens if num_draft_tokens is not None else 4
        else:
            self.num_draft_tokens = 0

        self.ngram_corpus = None
        if ngram_corpus is not None:
            if ngram_match_min <= 0:
                raise ValueError("ngram_corpus requires ngram_match_min > 0")
            from .ngram import NgramCorpus
            self.ngram_corpus = NgramCorpus(ngram_corpus, tokenizer)
        self.ngram_match_min = ngram_match_min
        # The widest verification pass any round can take: the regular drafter's window or a hybrid draft
        self.hybrid_draft_tokens = max(0, hybrid_draft_tokens or 0)
        self.max_draft_tokens = max(self.num_draft_tokens, self.hybrid_draft_tokens)
        self.dynamic_draft = dynamic_draft_tokens and self.num_draft_tokens > 0
        self.draft_max_rows = max(0, draft_max_rows or 0)
        self._draft_window_last = None
        self.record_draft_stats = record_draft_stats
        # cost-model gate: skip the drafter forward (and the R-row verify) whenever the
        # measured speculative rate would not beat plain decode. Opt-in: EXL3_SPEC_GATE=1
        self.spec_gate = SpecGate() if (draft_model is not None and
                                        os.environ.get("EXL3_SPEC_GATE", "0") != "0") else None
        max_q_size = max(self.max_draft_tokens + 1, max_q_size)

        # Chunking/partitioning
        self.max_batch_size = max_batch_size
        # Default chunk length comes from the model (an architecture may prefer longer chunks when
        # its prefill cost is dominated by per-chunk fixed work); EXL3_PREFILL_CHUNK overrides both
        if max_chunk_size is None:
            max_chunk_size = model.caps.get("prefill_chunk_size", 2048)
        if os.environ.get("EXL3_PREFILL_CHUNK"):
            max_chunk_size = int(os.environ["EXL3_PREFILL_CHUNK"])
        if max_chunk_size > 65535:
            raise ValueError(f"prefill chunk of {max_chunk_size} rows is not supported: a forward is limited to 65535 rows (mixer kernel grid.y); lower EXL3_PREFILL_CHUNK / max_chunk_size")
        self.max_chunk_size = max_chunk_size

        # Prompt ingestion alongside generating jobs, see iterate()
        # Env overrides for callers that predate these arguments (TabbyAPI passes neither). The old
        # behaviour, full chunks on every iteration, is EXL3_SHARED_CHUNK_SIZE = max_chunk_size with
        # EXL3_PREFILL_COOLDOWN = 0
        if shared_chunk_size is None and os.environ.get("EXL3_SHARED_CHUNK_SIZE"):
            shared_chunk_size = min(int(os.environ["EXL3_SHARED_CHUNK_SIZE"]), max_chunk_size)
        if prefill_cooldown is None:
            prefill_cooldown = int(os.environ.get("EXL3_PREFILL_COOLDOWN", "4"))
        if shared_chunk_size is None:
            shared_chunk_size = (max_chunk_size // 2 + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
        assert 0 < shared_chunk_size <= max_chunk_size and shared_chunk_size % PAGE_SIZE == 0, \
            "shared_chunk_size must be a multiple of the page size (256) and no larger than max_chunk_size"
        if prefill_cooldown is None:
            prefill_cooldown = 4
        assert prefill_cooldown >= 0, \
            "prefill_cooldown must not be negative"
        self.shared_chunk_size = shared_chunk_size
        self.prefill_cooldown = prefill_cooldown
        self.prefill_cooldown_left = 0
        self.prefill_rotation = 0

        # Job queues
        self.job_serial = 0
        self.pending_jobs = []
        self.active_jobs = []
        # Score-only jobs (Job(score_only = True)): FIFO, run at the head of iterate(), outside max_batch_size
        self.score_jobs = []
        self.score_trace = os.environ.get("EXL3_SCORE_TRACE", "0") != "0"
        # Off by default: decision states are mostly one-off, so their checkpoints and pages would only evict
        # chat state from the shared caches. Score jobs still read the shared caches either way
        self.score_cache = os.environ.get("EXL3_SYSTEMONE_CACHE", "0") == "1"
        # A group's split-point stash lives in VRAM: it is restored once per question 2..N and freed with the group
        self.score_group_device = os.environ.get("EXL3_SYSTEMONE_GROUP_DEVICE", "1") != "0"
        self.score_yielded = False
        self.deferred_jobs = []   # EXL3_REPLY_FIRST: finished jobs not yet released

        # Filter threads
        self.filter_pool = ThreadPoolExecutor(max_workers = 16)
        self.filter_queue = []

        # Rounds that verified a hybrid (long suffix-match) draft
        self.hybrid_rounds = 0

        # Pinned staging buffer for batched token readback in iterate_gen
        self.sample_pinned = None
        self.staging_buffers = {}
        self.stage_arenas = {}
        self.stage_devices = None

        # Buffers. Pinned: the draft input ids upload non-blocking from here every round (and
        # the DFlash2 selector reads its anchor from the same view), the drafted ids come back
        # into draft_ids_pinned
        if draft_model or ngram_match_min or self.hybrid_draft_tokens:
            self.draft_input_ids_pinned = torch.empty(
                (max_batch_size, 1),
                dtype = torch.long,
                pin_memory = True
            )
            self.draft_ids_pinned = torch.empty(
                (max_batch_size, self.max_draft_tokens),
                dtype = torch.long,
                pin_memory = True
            )

        # CPU page cache tier, and the disk tier behind it
        self.cpu_page_cache = None
        self.disk_page_cache = None
        self.disk_cache_flush_on_idle = disk_cache_flush_on_idle
        self.disk_cache_write_interval = disk_cache_write_interval
        self._write_behind_time = 0.0
        if cpu_cache_size:
            tier_caches = [cache] + ([draft_cache] if draft_cache is not None else [])
            self.cpu_page_cache = CPUPageCache(tier_caches, cpu_cache_size)
            self.cpu_page_cache.attach(self.pagetable)
            self.pagetable.cpu_tier = self.cpu_page_cache
            if disk_cache_dir:
                self.disk_page_cache = DiskPageCache(disk_cache_dir, tier_caches, self.cpu_page_cache, disk_cache_size,
                                                     min_state_distance = disk_cache_checkpoint_interval)
                self.cpu_page_cache.attach_disk(self.disk_page_cache)
        elif disk_cache_dir:
            raise ValueError("disk_cache_dir needs a CPU page cache tier to write through from (cpu_cache_size > 0)")

        # Visualizer
        if show_visualizer:
            self.visualizer = CacheVisualizer(self.pagetable.max_pages)
        else:
            self.visualizer = None

        # Defrag
        self.enable_defrag = enable_defrag

        # Exact prompt-end checkpoints (see Job._tail_stash); EXL3_TAIL_CKPT=0 turns them off
        self.tail_ckpts = []
        self.tail_ckpt_max = int(os.environ.get("EXL3_TAIL_CKPT", "8")) if tail_ckpt is None else int(tail_ckpt)
        # prompts shorter than this never stash (the stash is 20-40 ms of host copy and 142 MiB per entry)
        self.tail_ckpt_min = int(os.environ.get("EXL3_TAIL_MIN", "512"))

        # Recurrent cache. A job's recurrent state is a PairedState over the target and draft sides;
        # both sides checkpoint together into this one pool (see PairedState)
        self.recurrent_cache_size = recurrent_cache_size
        self.target_recurrent = bool(self.model.caps.get("recurrent_states"))
        self.draft_recurrent = bool(draft_model is not None and draft_model.caps.get("recurrent_states"))
        if self.target_recurrent:
            if self.max_draft_tokens > cache.max_history:
                raise ValueError(
                    f"Speculative passes of up to {self.max_draft_tokens} draft tokens need a cache with "
                    f"max_history >= {self.max_draft_tokens} on a recurrent model (this cache has "
                    f"max_history = {cache.max_history})"
                )
        if self.draft_recurrent and not draft_model.caps.get("dflash_draft") and self.num_draft_tokens > draft_cache.max_history:
            raise ValueError(
                f"A recurrent draft model drafting {self.num_draft_tokens} tokens needs a draft cache with "
                f"max_history >= {self.num_draft_tokens} (this cache has max_history = {draft_cache.max_history})"
            )
        if self.target_recurrent or self.draft_recurrent:
            self.recurrent_cache = RecurrentCache(self.model, recurrent_cache_size, draft_model)
            self.recurrent_cache.pagetable = self.pagetable
            self.recurrent_cache.disk_tier = self.disk_page_cache
            if self.disk_page_cache is not None:
                # Which model and cache stash under each side of a PairedState: a tensor-parallel side's
                # checkpoints are written and read by that model's ranks
                self.disk_page_cache.recurrent_sides = {
                    side: (m, c) for side, m, c, on in (
                        ("target", self.model, cache, self.target_recurrent),
                        ("draft", draft_model, draft_cache, self.draft_recurrent),
                    ) if on
                }
            # The new page table owns every page, so every state slot is ours too
            for c, recurrent in ((cache, self.target_recurrent), (draft_cache, self.draft_recurrent)):
                if recurrent:
                    c.reset_states()
                    # Limit batch size if cache has recurrent states
                    self.max_batch_size = min(self.max_batch_size, c.num_slots)
        else:
            self.recurrent_cache = None
        if recurrent_checkpoint_interval is None:
            # The target sets the checkpoint schedule; a recurrent draft under a non-recurrent target
            # lends it its own
            schedule_model = model if self.target_recurrent or not self.draft_recurrent else draft_model
            recurrent_checkpoint_interval = schedule_model.caps.get("default_recurrent_checkpoint_interval", 2048)

        assert recurrent_checkpoint_interval % PAGE_SIZE == 0 and recurrent_checkpoint_interval_pp % PAGE_SIZE == 0, \
            "checkpoint interval must be a multiple of the page size (256)"
        self.recurrent_checkpoint_interval = recurrent_checkpoint_interval
        self.recurrent_checkpoint_interval_pp = recurrent_checkpoint_interval_pp

        # Drafting mode
        if draft_model is not None and draft_model.caps.get("attach_target"):
            draft_model.attach_to(model)
        self.dflash_draft = self.draft_model is not None and self.draft_model.caps.get("dflash_draft", False)
        self.mtp_draft = self.draft_model is not None and self.draft_model.caps.get("mtp_draft", False)

        # Confidence-calibrated draft truncation (draft model + dynamic draft, any mode). For
        # DFlash the fixed-size drafted block is truncated before verification; for AR draft
        # models and MTP the drafting loop itself stops early, saving drafter forwards. Modes
        # whose TP lm_head path returns only the argmax simply never export confidence values,
        # leaving the calibrator inert
        self.draft_calibrator = None
        self._draft_conf_round = None
        # Sampled-draft round (spec_sample): (job, sampler params, draft laws per depth) for the next verify
        self._spec_round = None
        # Lazy drafter KV catch-up: plain 1-row steps do not feed the drafter cache, they park the
        # target hidden rows here and the next drafting round (or a flush point) writes them in one pass
        self._dkv_pending = None
        self.lazy_dkv = os.environ.get("EXL3_LAZY_DKV", "1") == "1"
        if self.dynamic_draft and self.draft_model is not None:
            self.draft_calibrator = DraftConfidenceCalibrator(draft_confidence)

        # Sparse drafting head over a frequency-selected vocabulary subset (layer-split mode only)
        self.sparse_head = None
        self._sparse_flag = False
        if sparse_draft_head and self.draft_model is not None and not model.loaded_tp:
            from ..modules.sparse_head import token_stats_path
            # Drafters with a head of their own (Gemma4 assistant) take the subset there; the others borrow the
            # target's head and the subset mirrors it
            own = getattr(self.draft_model, "sparse_head_linear", None)
            lm = own() if own is not None else model.modules[model.logit_layer_idx]
            self.sparse_head = lm.make_sparse_head(
                sparse_draft_head,
                model.config.vocab_size,
                profile_path = token_stats_path(model.config.directory, model.config.vocab_size),
                embedding = model.modules[0] if self.mtp_draft else None,
            )
            self.sparse_head.benchmark()


    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs) + len(self.score_jobs)

    def num_active_jobs(self):
        return len(self.active_jobs)

    def num_pending_jobs(self):
        return len(self.pending_jobs)

    def get_cache_stats(self) -> dict:
        """
        Snapshot of cache occupancy and prompt-cache efficiency, intended for status displays.

        Returns a dict with:
            page_size: tokens per page
            max_tokens: total cache capacity, in tokens
            used_tokens: tokens in pages referenced by active jobs
            cached_tokens: tokens in complete pages that a later or concurrent job could actually reuse: every
                page on an unbroken chain down to a root, whether or not an active job currently references it.
                Complete pages whose ancestors were evicted (orphans) don't count, since a prefix match walks
                the chain from the root
            free_tokens: tokens in unreferenced pages holding no reusable content
            tier_cached_tokens: reusable tokens held only in the system-memory tier (sysmem K/V cache), by the
                same chain rule; 0 without a tier
            tier_max_tokens: capacity of the system-memory tier, 0 without one
            disk_cached_tokens: reusable tokens held only in the disk tier, by the same chain rule (a chain
                may run through all three tiers; the disk tier is persistent, so these survive a restart);
                0 without a disk tier
            disk_pages, disk_states: pages and recurrent checkpoints the disk tier holds for this model
            disk_used_bytes, disk_max_bytes: the disk tier's occupancy and budget (the budget is for the whole
                directory, every model and cache configuration under it); 0 without a disk tier
            alloc_pages: cumulative pages claimed by started jobs, the pages reserved for generation included
            alloc_prompt_pages: of those, the complete prompt pages, i.e. the ones a prompt-cache hit can serve
            alloc_cached_pages: of those, pages that were reused from the prompt cache
            alloc_tier_pages: of those, pages restored from the system-memory tier (including pages that were
                promoted there from disk first)
            alloc_disk_pages: of those, pages that came from the disk tier, whether the enqueue-time prefetch
                or the allocation itself promoted them; a page counts here the first time it is restored after
                its promotion and as an ordinary tier page from then on
            hit_rate: alloc_cached_pages / alloc_prompt_pages, or None before any job has started. The
                denominator is the prompt pages only, so this matches the per-request cached fraction rather
                than being diluted by the pages each job reserves for its output
            tier_hit_rate: alloc_tier_pages / alloc_prompt_pages, the part of hit_rate served from the
                system-memory tier (disk promotions included), or None before any job has started
            disk_hit_rate: alloc_disk_pages / alloc_prompt_pages, the part of tier_hit_rate that was read from
                disk, or None before any job has started
            active_jobs, pending_jobs: current job counts

        The chain walk is memoized on a signature of the page tables, so repeated calls between changes are
        cheap.
        """
        pt = self.pagetable
        tier = pt.cpu_tier
        disk = self.disk_page_cache

        # The chain walk is linear in the number of pages, which can be large with a big system-memory
        # tier, so it only reruns when something that can change the outcome has moved
        signature = (
            len(pt.referenced_pages),
            len(pt.unreferenced_pages),
            sum(1 for h in pt.referenced_pages if is_content_hash(h)),
            pt.metrics["alloc_pages"],
            pt.metrics["evictions"],
            pt.last_defrag_serial,
            len(tier) if tier is not None else 0,
            (tier.metrics["pushes"], tier.metrics["evictions"]) if tier is not None else None,
            (len(disk), len(disk.states), disk.metrics["page_writes"], disk.metrics["evictions"]) if disk is not None else None,
        )
        memo_signature, memo = self._cache_stats_memo
        if signature == memo_signature:
            return {**memo, "active_jobs": len(self.active_jobs), "pending_jobs": len(self.pending_jobs)}

        # prev-hash links of every complete page, per tier
        gpu_links = {}
        for pages in (pt.referenced_pages, pt.unreferenced_pages):
            for h, page in pages.items():
                if is_content_hash(h):
                    gpu_links[h] = page.prev_hash
        tier_links = {h: e["prev_hash"] for h, e in tier.entries.items()} if tier is not None else {}
        if disk is not None:
            with disk._lock:
                disk_links = {h: m["prev_hash"] for h, m in disk.pages.items()}
        else:
            disk_links = {}

        # A page is reusable when its whole chain is present in either tier
        resumable = {}
        def is_resumable(h):
            chain = []
            while h is not None and h not in resumable:
                chain.append(h)
                if h in gpu_links:
                    h = gpu_links[h]
                elif h in tier_links:
                    h = tier_links[h]
                elif h in disk_links:
                    h = disk_links[h]
                else:
                    resumable[h] = False
                    break
            result = True if h is None else resumable[h]
            for c in chain:
                resumable[c] = result
            return result

        cached = sum(1 for h in gpu_links if is_resumable(h))
        tier_cached = sum(1 for h in tier_links if h not in gpu_links and is_resumable(h))
        disk_cached = sum(1 for h in disk_links if h not in gpu_links and h not in tier_links and is_resumable(h))
        used = len(pt.referenced_pages)
        free = sum(1 for h in pt.unreferenced_pages if not is_content_hash(h))
        alloc = pt.metrics["alloc_pages"]
        alloc_prompt = pt.metrics["alloc_prompt_pages"]
        alloc_cached = pt.metrics["alloc_cached_pages"]
        alloc_tier = pt.metrics["alloc_tier_pages"]
        alloc_disk = pt.metrics["alloc_disk_pages"]
        stats = {
            "page_size": PAGE_SIZE,
            "max_tokens": pt.max_pages * PAGE_SIZE,
            "used_tokens": used * PAGE_SIZE,
            "cached_tokens": cached * PAGE_SIZE,
            "free_tokens": free * PAGE_SIZE,
            "tier_cached_tokens": tier_cached * PAGE_SIZE,
            "tier_max_tokens": tier.max_slots * PAGE_SIZE if tier is not None else 0,
            "disk_cached_tokens": disk_cached * PAGE_SIZE,
            "disk_pages": len(disk) if disk is not None else 0,
            "disk_states": len(disk.states) if disk is not None else 0,
            "disk_used_bytes": disk.total_size if disk is not None else 0,
            "disk_max_bytes": disk.root_size if disk is not None else 0,
            "alloc_pages": alloc,
            "alloc_prompt_pages": alloc_prompt,
            "alloc_cached_pages": alloc_cached,
            "alloc_tier_pages": alloc_tier,
            "alloc_disk_pages": alloc_disk,
            "hit_rate": alloc_cached / alloc_prompt if alloc_prompt else None,
            "tier_hit_rate": alloc_tier / alloc_prompt if alloc_prompt else None,
            "disk_hit_rate": alloc_disk / alloc_prompt if alloc_prompt else None,
            "active_jobs": len(self.active_jobs),
            "pending_jobs": len(self.pending_jobs),
        }
        self._cache_stats_memo = (signature, stats)
        return stats


    def clear_queue(self):
        """
        Abort all active and pending jobs
        """

        self.flush_deferred()
        num_jobs = self.num_remaining_jobs()
        for job in self.active_jobs + self.pending_jobs:
            job.deallocate_pages()
        for job in list(self.score_jobs):
            self.score_retire(job)
        self.active_jobs.clear()
        self.pending_jobs.clear()
        self.score_jobs.clear()
        if num_jobs and not self.num_remaining_jobs():
            self.on_queue_drained()


    def enqueue(
        self,
        job: Job | list[Job]
    ) -> int | list[int]:
        """
        Adds a job or list of jobs to the queue.

        Each job is prepared against this generator before it is appended to the pending queue. Preparation assigns
        the current job_serial, hashes prompt pages for later cache lookup, and validates that the request can fit
        the configured cache and batch limits. The serial is then incremented and returned to the caller; it is also
        included in started, prefill and streaming result dictionaries so clients can correlate events with the
        logical request. Requeued jobs keep their original serial number internally so a long generation still
        appears as one stream.

        returns:
            int: (List of) unique serial number(s) for job(s)
        """

        if isinstance(job, list):
            serials = []
            for j in job:
                serials.append(self.enqueue(j))
            return serials

        job.prepare_for_queue(self, self.job_serial)
        self.job_serial += 1
        if job.score_only:
            if job.score_group is not None:
                job.score_group.pending += 1
            self.score_jobs.append(job)
        else:
            self.pending_jobs.append(job)
        if self.disk_page_cache is not None:
            # Start reading the prompt's pages that only the disk tier holds, so the allocation finds them in
            # the CPU tier (or in flight) rather than waiting on the disk itself
            for seq in job.sequences:
                hashes = seq.page_hashes if seq.max_cached_pages is None else seq.page_hashes[:seq.max_cached_pages]
                self.disk_page_cache.prefetch(hashes, self.pagetable.access_serial)
        return job.serial_number


    def cancel(
        self,
        job: Job
    ):
        """
        Cancel a single pending or active job.

        Pending jobs are removed before they ever allocate cache pages. Active jobs first release their page
        references and recurrent state, then leave active_jobs so future iterations stop sampling them. If the
        cancellation drains the generator completely, the page table is defragmented immediately because no active
        block tables still depend on the current physical page ordering.
        """

        self.flush_deferred()
        num_jobs = self.num_remaining_jobs()

        if job in self.score_jobs:
            self.score_jobs.remove(job)
            self.score_retire(job)
        elif job in self.pending_jobs:
            self.pending_jobs.remove(job)
        elif job in self.active_jobs:
            job.deallocate_pages()
            self.active_jobs.remove(job)
        if num_jobs and not self.num_remaining_jobs():
            self.on_queue_drained()


    def flush_deferred(self):
        """
        EXL3_REPLY_FIRST: release the jobs that finished in the last iterate() (see _REPLY_FIRST). No-op when none.
        """
        if not self.deferred_jobs:
            return
        jobs, self.deferred_jobs = self.deferred_jobs, []
        with torch.inference_mode():
            for job in jobs:
                job.deallocate_pages()
        if not self.num_remaining_jobs():
            self.on_queue_drained()


    @torch.inference_mode
    def iterate(self) -> list[dict]:
        self.flush_deferred()
        results = self._iterate_inner()
        if _moe_fused_check is not None:
            _moe_fused_check(self.model)
        return results

    def _iterate_inner(self) -> list[dict]:
        """
        Performs inference on available jobs.

        :return:
            List of dicts:

            # Job has started
            {
                "job": Job  - reference to job
                "stage": "started"
                "identifier":  - optional identifier
                "serial": int  - job serial number
                "eos": bool  - always False at this stage
            }

            # Prefill is underway
            {
                "job": Job  - reference to job
                "stage": "prefill"
                "curr_progress": int  - prompt tokens ingested so far
                "max_progress": int  - total prompt tokens to ingest
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - always False at this stage
            }

            # Generation is underway
            {
                "job": Job  - reference to job
                "stage": "streaming"
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - True if stop condition has been met

                optional, if eos:
                    "eos_reason":  - one of:
                        "stop_token"  (stop token was reached)
                        "stop_string"  (stop string was completed)
                        "max_new_tokens"  (max_new_tokens reached)
                        "end_filter"  (filter reached end state with eos_after_completed = True)
                        "loop_detected"  (loop detection triggered)
                    optional, if "eos_reason" == "stop_token":
                        "eos_triggering_token_id": int
                        "eos_triggering_token_str": str
                    optional, if "eos_reason" == "stop_string":
                        "eos_triggering_string": str
                    "full_completion": str  - full text completion
                    "new_tokens": int  - number of tokens generated
                    "time_enqueued": float  - time from job was enqueued until it started, in seconds
                    "time_prefill": float  - time to first token, in seconds
                    "time_generate": float  - time to last token, in seconds
                    optional, if SD enabled:
                        "accepted_draft_tokens": int
                        "rejected_draft_tokens": int

                "text": str  - streamed text output. Does not include prefix from healed token, or stop string
                "token_ids": torch.Tensor  - output tokens, shape (1, n)
                "token_probs": torch.Tensor  - last sampling probability of output tokens, shape (1, n)
                "top_k_tokens": torch.Tensor  - shape (1, n, k)
                "top_k_probs": torch.Tensor  - shape (1, n, k)
                "logits": torch.Tensor  - shape (1, n, vocab_size)
            }
        """

        assert self.cache.initialized, \
            "Cache tensors were never allocated. Construct the Cache BEFORE calling model.load()"
        owner_serial = getattr(self.cache, "owner_serial", None)
        if owner_serial is not None and owner_serial != getattr(self, "cache_owner_serial", owner_serial):
            # Superseded: drop every job without touching the cache (the new owner may already have
            # reused these pages and state slots) and refuse to run
            self.pending_jobs.clear()
            self.active_jobs.clear()
            self.score_jobs.clear()
            raise RuntimeError(
                "This Generator no longer owns its Cache: a newer Generator was created over the same "
                "Cache. Close or discard the old Generator before creating a new one."
            )
        assert self.draft_cache is None or self.draft_cache.initialized, \
            "Draft cache tensors were never allocated. Construct the draft Cache BEFORE calling draft_model.load()"

        results = []

        # Score-only jobs first: they wait behind at most the round already in flight, never behind the
        # prefill chunks this round is about to run. Results only reach their consumers when iterate()
        # returns, so a round that produced scores returns right away instead of running a prefill chunk
        # (seconds, at chunk_size 4096) on top. Never twice in a row, so a steady stream of score requests
        # still leaves every other round to the regular jobs
        if self.score_jobs:
            self.iterate_score(results)
            if results and not self.score_yielded and (self.active_jobs or self.pending_jobs):
                self.score_yielded = True
                return results
        self.score_yielded = False

        _t0 = _ttrace_begin()
        self.iterate_start_jobs(results)
        _ttrace_end("start_jobs (alloc pages/restore ckpt)", _t0)

        # Perform one round of prefill. While some active job is already generating, prompt ingestion shares
        # the GPU with it: at most one job with prompt left to ingest runs a pass per iteration, with the
        # shared chunk size, and only once the cooldown since the previous pass has elapsed, so every call
        # still produces one generation step for the generating jobs. The other jobs' passes are deferred;
        # jobs with nothing left to ingest keep their call, which only does bookkeeping. With no job
        # generating, or with a full shared chunk and no cooldown, every job runs a full chunk on every
        # iteration
        prefilling = [job for job in self.active_jobs if not job.is_prefill_done()]
        chunk_size = None
        deferred = []
        if prefilling and len(prefilling) < len(self.active_jobs) and \
                (self.shared_chunk_size < self.max_chunk_size or self.prefill_cooldown > 0):
            if self.shared_chunk_size < self.max_chunk_size:
                chunk_size = self.shared_chunk_size
            if self.prefill_cooldown_left > 0:
                self.prefill_cooldown_left -= 1
                deferred = prefilling
            else:
                self.prefill_cooldown_left = self.prefill_cooldown
                selected = prefilling[self.prefill_rotation % len(prefilling)]
                self.prefill_rotation += 1
                deferred = [job for job in prefilling if job is not selected]
        else:
            self.prefill_cooldown_left = 0
        _t0 = _ttrace_begin()
        for job in list(self.active_jobs):
            if any(job is d for d in deferred):
                continue
            try:
                job.prefill(results, chunk_size)
            except Exception as e:
                if os.environ.get("KYO_TURN_TRACE"):
                    import traceback; traceback.print_exc()
                self.reap_failed_job(job, e, results)
        _ttrace_end("prefill total", _t0)

        # Recurrent checkpoints
        _t0 = _ttrace_begin()
        if self.recurrent_cache is not None:
            self.recurrent_checkpoint()
        _ttrace_end("recurrent_checkpoint", _t0)
        _t0 = _ttrace_begin()

        # Generation with draft model, with the draft window trimmed to the row ceiling (draft_max_rows)
        draft_window = self.row_limited(self.num_draft_tokens) if self.draft_model else 0
        round_start = None
        if self.sparse_head is not None and draft_window > 0:
            self._sparse_flag = self.sparse_head.round_flag()
            round_start = time.perf_counter()
        if self.draft_model and draft_window > 0:
            if self.dflash_draft:
                draft_fn = lambda results: self.iterate_draftmodel_dflash_gen(results, draft_window)
            elif self.mtp_draft:
                draft_fn = lambda results: self.iterate_draftmodel_mtp_gen(results, draft_window)
            else:
                draft_fn = lambda results: self.iterate_draftmodel_gen(results, draft_window)

        # Generation with n-gram draft
        elif self.ngram_match_min:
            draft_fn = self.iterate_ngram_gen

        # Regular generation
        else:
            draft_fn = lambda results: None

        # Long suffix-match draft over the regular one, when a job is repeating its context
        if self.hybrid_draft_tokens:
            inner_fn = draft_fn
            draft_fn = lambda results: self.iterate_hybrid_gen(inner_fn(results), self.row_limited(self.hybrid_draft_tokens))

        if round_start is not None:
            sparse_fn = draft_fn
            def draft_fn(results):
                draft_tokens = sparse_fn(results)
                self._sparse_drafted = draft_tokens.shape[-1] if draft_tokens is not None else 0
                return draft_tokens
            self._sparse_drafted = 0

        # A round in which every generating job needs exactly one more token cannot accept a single
        # drafted token, so the draft forwards and the wider verify window are pure cost. Plain
        # generation still exports the target state, so an MTP carry and the lazy drafter KV stay
        # valid; it is the same path a one-token prompt or a spec-gate plain step takes. Kept out of
        # the spec gate's timing: a last-token round says nothing about spec vs plain throughput
        if (self.draft_model is not None or self.ngram_match_min) and self.all_jobs_need_one_token():
            self._skip_draft_round(results)
        elif self.spec_gate is not None and (self.dflash_draft or self.mtp_draft):
            self._iterate_gated(results, draft_fn)
        else:
            self.iterate_gen(results, draft_fn(results))
        if round_start is not None and self._sparse_drafted:
            # MTP drafts one token per head call; DFlash drafters take one head call for the whole block
            drafted = self._sparse_drafted
            self.sparse_head.note_round(time.perf_counter() - round_start, drafted, drafted if self.mtp_draft else 1)

        _ttrace_end("generation round", _t0)
        # Visualization
        if self.visualizer:
            self.update_visualizer()

        # Disk tier: queue the complete pages of the active sequences (see disk_cache_write_interval)
        if self.disk_page_cache is not None and self.active_jobs and self.disk_cache_write_interval > 0:
            self.write_behind()

        # Finished iteration
        return results


    def close(self):
        """
        Release the host memory a retired generator would otherwise hold until it is garbage collected,
        which can be long after it stops being used: a generator sits in a reference cycle with its page
        table, and the jobs that failed on it keep it reachable through their exception tracebacks. The
        model and cache stay loaded for a replacement generator to use; this generator is not reusable
        afterwards. Safe to call more than once.

        Releases, in order of size: the recurrent checkpoint cache (up to recurrent_cache_size of system
        RAM, in the tensor-parallel ranks too), the CPU page cache tier (cpu_cache_size of pinned memory),
        the pinned staging and draft buffers, and the filter thread pool.
        """
        self.flush_deferred()
        if self.disk_page_cache is not None:
            # Persist what is still only in VRAM or RAM, then let the writer drain before its sources go away.
            # After abort() (generator error), nothing is written
            if self.disk_cache_flush_on_idle and not self.disk_page_cache.write_disabled:
                self.flush_cache_to_disk()
            self.disk_page_cache.close()
            self.disk_page_cache = None
            if self.recurrent_cache is not None:
                self.recurrent_cache.disk_tier = None
        if self.recurrent_cache is not None:
            self.recurrent_cache.close()
        if self.cpu_page_cache is not None:
            self.cpu_page_cache.close()
            self.cpu_page_cache = None
            self.pagetable.cpu_tier = None
        self.sample_pinned = None
        self.staging_buffers = {}
        self.stage_arenas = {}
        self.draft_input_ids_pinned = None
        self.draft_ids_pinned = None
        self.filter_pool.shutdown(wait = False)


    @torch.inference_mode()
    def flush_cache_to_disk(self, wait: bool = False, max_pages: int | None = None) -> dict:
        """
        Queue every complete, content-hashed page that is in VRAM but in neither the CPU nor the disk tier for
        persistence (through the CPU tier, whose store is the write-through point), and every recurrent
        checkpoint not yet on disk. Returns counts of what was queued; with wait = True returns once the
        writer has finished.

        max_pages limits the number of pages queued in one call. Each page takes a CPU-tier slot until its
        write is complete. If the pages are more than the free slots, the tier must wait for the writer, and
        this call stops the generator for that time. With a limit, the call stops early and sets "more" in the
        result. The idle loop then calls again (see idle_work).
        """
        disk = self.disk_page_cache
        tier = self.cpu_page_cache
        queued = {"pages": 0, "states": 0, "more": False}
        if disk is None:
            return queued
        pt = self.pagetable
        pages = list(pt.unreferenced_pages.values()) + list(pt.referenced_pages.values())
        queued["pages"], queued["more"] = self._queue_pages_to_disk(pages, max_pages)
        rc = self.recurrent_cache
        if rc is not None and rc.disk_tier is not None:
            # Checkpoints follow their pages: every complete page is queued above, so an anchored checkpoint
            # is written now, the rest stay candidates. The last checkpoint of each chain (no other checkpoint
            # in RAM is after it on its chain) is written also when it is closer than min_state_distance to
            # the one before: a conversation continues from the end of its chain
            keys = [h for h in rc.keys() if is_content_hash(h)]
            keyset = set(keys)
            covered = set()
            limit = pt.max_pages + len(tier) + len(disk)
            for k in keys:
                h = disk.chain_prev(k)
                steps = 0
                while h is not None and steps <= limit and h not in covered:
                    if h in keyset:
                        covered.add(h)
                    h = disk.chain_prev(h)
                    steps += 1
            for h in keys:
                stashed = rc.get(h)
                if stashed is None or not (disk.has_page(h) or h in disk.pending_pages):
                    continue
                if disk.write_state(h, stashed, force = h not in covered):
                    queued["states"] += 1
        queued["states_pruned"] = disk.prune_states()
        self._disk_backlog = queued["more"]
        if wait:
            disk.flush()
        return queued


    def _queue_pages_to_disk(self, pages, budget: int | None) -> tuple[int, bool]:
        """
        Queue the complete, content-hashed, persistable pages among `pages` that are in VRAM but in neither the
        CPU nor the disk tier, through the CPU tier (its store is the write-through point; the copy to host
        memory goes on the current stream, after the kernels that wrote the page and before any that could
        write it again). Stops after `budget` pages. Returns (pages queued, stopped early).
        """
        disk, tier, pt = self.disk_page_cache, self.cpu_page_cache, self.pagetable
        queued = 0
        for page in pages:
            h = page.phash
            if page.kv_position != PAGE_SIZE or not is_content_hash(h) or not page.persistable:
                continue
            if h in tier or disk.has_page(h) or h in disk.pending_pages:
                continue
            if budget is not None and queued >= budget:
                return queued, True
            tier.store(page, pt.access_serial)
            queued += 1
        return queued, False


    @torch.inference_mode()
    def write_behind(self):
        """
        Queue the complete pages of the active sequences for the disk tier, at most once per
        disk_cache_write_interval seconds, so a process that dies during a long prefill loses only what came
        after the last page (and checkpoint) written. The generator does not wait for the copies or the disk.

        The last two pages before a sequence's position are not queued: speculative decoding can still rewind
        them, and the MTP draft cache writes the last slot of a page one step later. One call queues at most
        a quarter of the CPU tier, less the writes still queued, so the tier never has to wait for the writer
        to free a slot.
        """
        now = time.monotonic()
        if now - self._write_behind_time < self.disk_cache_write_interval:
            return
        self._write_behind_time = now
        disk = self.disk_page_cache
        if disk.write_disabled:
            return
        budget = self.cpu_page_cache.max_slots // 4 - len(disk.pending_pages)
        if budget <= 0:
            return
        pages = []
        for job in self.active_jobs:
            for seq in job.sequences:
                if seq.allocated_pages:
                    pages.extend(seq.allocated_pages[:max(0, seq.kv_position // PAGE_SIZE - 2)])
        self._queue_pages_to_disk(pages, budget)


    def _idle_flush_budget(self) -> int:
        """Pages for one idle flush step: half the CPU tier, less the writes still queued"""
        return max(1, self.cpu_page_cache.max_slots // 2 - len(self.disk_page_cache.pending_pages))


    def has_idle_work(self) -> bool:
        """True if the idle flush stopped early and has more pages to queue"""
        return self.disk_page_cache is not None and getattr(self, "_disk_backlog", False)


    @torch.inference_mode()
    def idle_work(self):
        """Do one more idle flush step, if no job is queued. The async generator calls this while idle"""
        if self.has_idle_work() and not self.num_remaining_jobs():
            self.flush_cache_to_disk(max_pages = self._idle_flush_budget())


    @torch.inference_mode()
    def on_queue_drained(self):
        """
        Idle-transition housekeeping: drop recurrent checkpoints stranded by KV eviction, then defragment the
        page table (both need/prefer a moment with no active block tables), and return whatever host memory
        glibc is retaining from stash/offload churn (issue #277) — sub-threshold residue would otherwise sit
        in the arenas for the whole idle period.
        """
        if self.sparse_head is not None:
            self.sparse_head.save_profile()
        if self.disk_page_cache is not None and self.disk_cache_flush_on_idle and \
                not self.disk_page_cache.write_disabled:
            # Before pruning: a checkpoint stranded in VRAM/RAM terms may be anchored by pages on disk. One
            # step only; the async generator does the rest while idle (idle_work)
            self.flush_cache_to_disk(max_pages = self._idle_flush_budget())
        if self.recurrent_cache is not None:
            self.recurrent_cache.prune_stranded()
            # The pruned checkpoints' buffers went back to the stash pool; drop them so the RAM
            # is actually returned (the next stash reallocates once)
            host_pool.release()
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_host_pool_release, ())
        self.pagetable.defrag()
        # Dynamic expert placement: apply any pending swap sweep now, between generations —
        # a placement change perturbs the logits slightly (same expert, different device
        # numerics) and must never land mid-stream
        from ..modules.block_sparse_mlp_cpu import run_pending_swap_sweeps
        run_pending_swap_sweeps(self.model.config.infer_params)
        malloc_trim()


    def requeue_one(self, job, results) -> bool:
        """
        Put a yielding job back on the queue, shrinking its headroom rather than killing it.

        Returns True if the job was requeued, False if it was reaped.

        A job yields every max_rq_tokens generated tokens so it does not monopolise the page
        table. prepare_for_requeue() turns everything generated so far into the new prompt and
        re-validates — and that validation can fail once two long conversations are competing
        for one page table. Patch 05 made that failure survivable for the SERVICE. This makes
        it survivable for the REQUEST.

        🔴 The page requirement is driven by max_rq_tokens, NOT max_new_tokens:
            Sequence.prepare(has_prefix, max_new_tokens = self.max_rq_tokens)
            max_len     = len(sequence_ids) + max_rq_tokens
            total_pages = ceil(max_len / PAGE_SIZE)
        So a job that no longer fits is asking for too much GENERATION HEADROOM PER YIELD,
        not too many tokens overall. Halving max_rq_tokens makes it yield twice as often and
        reserve half the headroom; the work still completes, just in smaller slices.
        Clamping max_new_tokens instead would do NOTHING AT ALL — that is the trap this
        docstring exists to prevent.
        """
        try:
            self.pending_jobs.insert(0, job.prepare_for_requeue())
            return True
        except Exception as e:
            # prepare_for_requeue() does `rq_job = self` and re-inits IN PLACE before calling
            # prepare_for_queue(), so on a raise `job` IS the fully-formed replacement job and
            # only the final validation failed. Retrying prepare_for_queue() on it is safe.
            boundary = (self.recurrent_checkpoint_interval
                        if self.recurrent_cache is not None else PAGE_SIZE)
            # orig_max_rq_tokens is None only when output_chunking is off, in which case
            # prepare_for_queue set max_rq_tokens = max_new_tokens + 1. Clamping THAT makes a
            # non-chunking job start chunking — a behaviour change, but strictly better than
            # failing it.
            clamped = job.orig_max_rq_tokens or job.max_rq_tokens
            for _ in range(8):
                prev = clamped
                clamped = max(boundary, (clamped // 2 // boundary) * boundary)
                # 🔴 NEVER CLAMP UPWARD. boundary comes from the model
                # (default_recurrent_checkpoint_interval, 2048 here) and nothing stops a future
                # model declaring one LARGER than max_rq_tokens — in which case
                # max(boundary, ...) would ask for MORE headroom than the attempt that just
                # failed, and retry forever on a request that only grows.
                if clamped >= prev:
                    break
                job.max_rq_tokens = clamped
                try:
                    job.prepare_for_queue(self, job.serial_number, rq = True)
                except Exception:
                    if clamped <= boundary:
                        break
                    continue
                # Persist it. The sequence only ever grows, so pressure only increases; letting
                # the next yield start from orig_max_rq_tokens again would just repeat the
                # failed attempt every single time.
                job.orig_max_rq_tokens = clamped
                self.pending_jobs.insert(0, job)
                print(f" ## Note: job {getattr(job, 'serial_number', '?')} no longer fits its "
                      f"requeue headroom; max_rq_tokens clamped to {clamped} (yields more "
                      f"often) instead of failing the request")
                return True
            # Genuinely impossible: even one checkpoint interval of headroom does not fit, i.e.
            # the SEQUENCE ITSELF has outgrown the page table. No amount of yielding helps, so
            # the job dies here — but alone, and with the ORIGINAL exception so the client sees
            # the real reason.
            self.reap_failed_job(job, e, results)
            return False


    def recurrent_checkpoint(self):
        for job in self.active_jobs:
            # 🔴 FOURTH FAILURE SITE. iterate() calls this UNGUARDED on every iteration, and
            # maybe_stash_recurrent() holds two bare asserts (`seq.kv_position % PAGE_SIZE == 0`,
            # `page.kv_position == PAGE_SIZE`) plus a cache.put(). A raise here escapes iterate() into
            # AsyncGenerator._run_iteration, latches self.error and kills the generator for EVERY job --
            # the same failure class as patches 00/04/05.
            #
            # We SKIP rather than reap. The stash is a pure write into the shared recurrent cache keyed by
            # page hash; the running job generates from its own self.recurrent_state and never reads back
            # from the cache mid-stream. So a failed stash cannot corrupt this job's output -- it only means
            # a future job resuming this prefix has to recompute from an earlier checkpoint. Reaping a
            # healthy generation over a failed cache write would be strictly more destructive than the
            # degradation we are containing.
            if getattr(job, "_exl3_stash_broken", False):
                continue
            try:
                job.maybe_stash_recurrent(self.recurrent_cache)
            except Exception as e:
                # Latch per job: this is called every iteration, so an unlatched warning would spam the
                # log at decode rate for the life of the generation.
                job._exl3_stash_broken = True
                print(f" ## Warning: recurrent checkpoint failed for job "
                      f"{getattr(job, 'serial_number', '?')} ({repr(e)}); checkpointing "
                      f"disabled for this job, generation continues")


    def all_jobs_need_one_token(self) -> bool:
        """
        True when at least one job generates this round and every such job stops after its next token on
        max_new_tokens. Stop tokens and strings are not consulted: they only ever end a job earlier.
        A healing job (new_tokens == -1) needs two.
        """
        n = 0
        for job in self.active_jobs:
            if not job.is_prefill_done():
                continue
            if job.max_new_tokens is None or job.max_new_tokens - job.new_tokens > 1:
                return False
            n += 1
        return n > 0


    def _skip_draft_round(self, results: list):
        self._spec_round = None
        self._draft_conf_round = None
        pend = self._dkv_pending
        if pend is not None:
            gen_jobs = [j for j in self.active_jobs if j.is_prefill_done()]
            if len(gen_jobs) != 1 or id(gen_jobs[0]) != pend["job"]:
                self._flush_dkv()
        self.iterate_gen(results, None)


    def update_visualizer(self):
        chains = []
        for job in self.active_jobs:
            for seq in job.sequences:
                idx = job.serial_number
                chain = [page.page_index for page in seq.allocated_pages]
                chains.append((idx, chain))
        usage = [0] * self.pagetable.max_pages
        for page in self.pagetable.all_pages:
            usage[page.page_index] = page.kv_position / PAGE_SIZE
        self.visualizer.update(chains, usage)


    def iterate_draftmodel_gen(self, results: list, window: int):

        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + window + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        block_index, cache_seqlens = self._draft_batch_tables(batch_size, max_seq_len)

        # Indexed embeddings not supported when drafting
        # TODO: Allow multimodal draft model, perhaps with dummy embeddings?
        for job in self.active_jobs:
            assert not job.embeddings, \
                "MM embeddings not supported while using draft model."

        # Recurrent drafter: its states must stand exactly at the K/V position before the round (the
        # draft consumes every token it drafts, so this holds unless a hybrid round accepted past its
        # window); a lagging state first consumes the gap in a plain pass
        draft_states = None
        if self.draft_recurrent:
            draft_states = []
            for job in self.active_jobs:
                if not job.is_prefill_done(): continue
                seq = job.sequences[0]
                ds = job.recurrent_state.draft
                if ds.position < seq.kv_position:
                    gap = seq.sequence_ids.torch_slice(ds.position, seq.kv_position)
                    self.draft_model.prefill(gap, {
                        "attn_mode": "flash_attn",
                        "block_table": seq.block_index_tensor,
                        "cache": self.draft_cache,
                        "cache_seqlens": torch.tensor([ds.position], dtype = torch.int32),
                        "recurrent_states": [ds],
                    })
                assert ds.position == seq.kv_position, "recurrent draft state out of step with the K/V position"
                draft_states.append(ds)

        # Collect input IDs
        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        def draft_params(step):
            p = {
                "attn_mode": "flash_attn",
                "block_table": block_index,
                "cache": self.draft_cache,
                "cache_seqlens": cache_seqlens,
            }
            if draft_states is not None:
                # One speculative pass over the drafted tokens, fed one step at a time; the round's
                # verification settles it (rewind to the accepted prefix) in iterate_gen
                p.update({"recurrent_states": draft_states, "recurrent_history": True, "recurrent_spec_step": step})
            return p

        # Greedy sample batched draft tokens. With a confidence calibrator, drafting stops early
        cal = self.draft_calibrator
        conf_cols = []
        reach = None
        for idx in range(window):
            batch_logits = self.draft_model.forward(input_ids = batch_ids, params = draft_params(idx))
            if self.vocab_limit is not None:
                batch_logits = batch_logits[..., :self.vocab_limit]    # the draft head's padding columns have no piece
            if cal is not None:
                conf, new_ids = torch.max(batch_logits, dim = -1)
            else:
                new_ids = torch.argmax(batch_logits, dim = -1)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            # Feed the next step from the host copy just made rather than a
            # second device-to-host copy of the same value. Each one is a full
            # sync, and syncs are what leave the GPU idle while the host
            # re-dispatches the next draft step
            batch_ids.copy_(self.draft_ids_pinned[:batch_size, idx:idx+1])
            cache_seqlens += 1
            if cal is not None:
                c = conf.float().cpu()
                conf_cols.append(c)
                est = [cal.estimate(v) for v in c.view(-1).tolist()]
                reach = est if reach is None else [r * e for r, e in zip(reach, est)]
                if idx + 1 < window and max(reach) < cal.confidence:
                    window = idx + 1
                    break

        # The last drafted token goes into the draft cache too, so a fully accepted round leaves the
        # drafter at the next unprocessed token
        self.draft_model.prefill(input_ids = batch_ids, params = draft_params(window))

        if conf_cols:
            self._draft_conf_round = {
                "ids": self.draft_ids_pinned[:batch_size, :window],
                "conf": torch.cat(conf_cols, dim = 1),
                "window": window,
            }

        return self.draft_ids_pinned[:, :window]


    def clear_prompt_ckpts(self):
        """Drop every prompt-end recurrent checkpoint (host memory, ~142 MiB each)."""
        self.tail_ckpts.clear()


    def _flush_mtp_defer(self, job, with_tail: bool):
        """EXL3_PF_DEFER: run the draft prefill chunks that Job.prefill stashed instead of running them before the first
        token. with_tail also writes the draft K/V row of the last prompt token (token L-1 paired with the hidden state
        of L-2): the normal first draft round writes that row itself, a deferred first round (plain target step) does not."""
        chunks = job.mtp_defer
        job.mtp_defer = None
        if not chunks:
            return
        for ids, params in chunks:
            self.draft_model.prefill(ids, params)
        if with_tail and job.mtp_defer_carry is not None:
            seq = job.sequences[0]
            pos = job.mtp_defer_end
            self.draft_model.prefill(seq.sequence_ids.torch_slice(pos, pos + 1), {
                "attn_mode": "flash_attn",
                "block_table": seq.prefill_block_table(),
                "cache": self.draft_cache,
                "cache_seqlens": torch.tensor([pos], dtype = torch.int32),
                "target_hidden": job.mtp_defer_carry,
                "indexed_embeddings": job.embeddings,
                "kv_only": True,
            })


    def _mtp_defer_gate(self) -> bool:
        """True when the coming round must be a plain target step (first token without the draft prefill).
        Otherwise flushes whatever is pending and returns False. Called at the top of the MTP draft round."""
        pend = [j for j in self.active_jobs if j.is_prefill_done() and getattr(j, "mtp_defer", None)]
        if not pend:
            return False
        rows = sum(c[0].shape[-1] for j in pend for c in j.mtp_defer)
        if len(self.active_jobs) == 1 and not pend[0].mtp_defer_plain and \
                rows >= int(os.environ.get("EXL3_PF_DEFER_MIN", "1024")):
            pend[0].mtp_defer_plain = True
            return True
        for j in pend:
            self._flush_mtp_defer(j, with_tail = j.mtp_defer_plain)
        return False


    def _bound_draft_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Draft proposals come from a head as wide as the embedding table, whose last columns are padding (no piece in the
        tokenizer). Such a draft can never be accepted, because the sampler never returns a padding id, and fed back to
        the drafter it reads a meaningless embedding row. Replace it by id 0: the round verifies that position as a
        rejection exactly as it would have, and accepted tokens do not change. The drafters that know vocab_limit already
        search the bounded head, so this only guards the rest. No host sync.
        """
        if self.vocab_limit is None:
            return ids
        return torch.where(ids < self.vocab_limit, ids, torch.zeros_like(ids))


    def note_unknown_token(self, token_id: int, piece_count: int):
        """A sampled id without a piece reached Job.receive_sample. It decodes to empty text; the first one is logged."""
        if self._unknown_token_seen:
            return
        self._unknown_token_seen = True
        logger.error(
            "sampled token id %d has no piece (the tokenizer defines %d, the model head has %d columns); decoded as empty "
            "text. Logged once per generator",
            token_id, piece_count, self.model.config.vocab_size
        )


    def iterate_draftmodel_mtp_gen(self, results: list, window: int):

        self._spec_round = None
        if self._mtp_defer_gate():
            return None
        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + window + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        lean_mtp = _host_lean()
        block_index, cache_seqlens = self._draft_batch_tables(batch_size, max_seq_len)

        # Collect input IDs
        input_ids_list = []
        mtp_hidden_list = []
        # EXL3_MTP_FUSE_CATCHUP: a single job's deferred catch-up rows ride in the first draft
        # forward (rows K+1..K+A-1, then the new token); batched rounds flush them first
        fuse_job = None
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if getattr(job, "mtp_pending", None) is not None:
                if batch_size == 1 and job.mtp_last_hidden is not None:
                    fuse_job = job
                else:
                    self._flush_mtp_pending(job)
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            assert len(job.sequences) == 1, "Qwen3.5 MTP drafting does not currently support CFG/multi-sequence jobs"
            if job.mtp_last_hidden is None:
                # A one-token prompt has no token to prefill before the generation input.
                # Run one normal target step first; iterate_gen() will initialize MTP state.
                return None
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
            mtp_hidden_list.append(job.mtp_last_hidden)
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))
        # Lean: batch_size == 1 is the whole served case; skip a GPU alloc + copy per round.
        temp_hidden = mtp_hidden_list[0] if (lean_mtp and len(mtp_hidden_list) == 1) \
            else torch.cat(mtp_hidden_list, dim = 0)
        n_pend = 0
        first_ids = batch_ids
        if fuse_job is not None:
            pend_ids = fuse_job.mtp_pending[0]
            fuse_job.mtp_pending = None
            n_pend = pend_ids.shape[-1]
            first_ids = torch.cat((pend_ids, batch_ids), dim = 1)
            cache_seqlens -= n_pend

        # Greedy sample batched draft tokens. As in iterate_draftmodel_gen, drafting stops once
        # every row's running product of estimated conditional acceptance probabilities falls
        # below the confidence target, keeping the first low-confidence token as the label probe
        cal = self.draft_calibrator
        conf_cols = []
        reach = None
        # A sampled job (T > 0) draws its drafts from the drafter's law under its own sampler (spec_sample)
        spec_job = spec_prm = None
        if spec_sample.SPEC_SAMPLED and batch_size == 1:
            spec_job = next(j for j in self.active_jobs if j.is_prefill_done())
            spec_prm = spec_sample.eligible(spec_job)
        spec_q = []
        if spec_prm is not None:
            # Filled depth by depth: a draft policy that stops the loop early (spec_policy StopDraft) verifies
            # a prefix of draft_ids_pinned, whose laws are all recorded by then
            self._spec_round = (spec_job, spec_prm, spec_q)
        # The drafted ids stay on the device inside the loop: each step's argmax feeds the next step's
        # input directly (with the sparse head, together with the token's embedding row, so the embedding
        # table is never consulted), and the window comes back to pinned memory in one copy at the end
        draft_cols = []
        draft_emb = None
        embed_device = self.model.modules[0].device
        # A drafter without K/V of its own reads the target's cache (and window rings) at the K/V position
        # of the round; its query position stays at the last accepted token for every step, as the
        # reference implementation drafts
        target_kv = self.draft_model.caps.get("draft_uses_target_cache", False)
        if target_kv:
            kv_seqlens = cache_seqlens.clone()
            target_states = [job.recurrent_state.target for job in self.active_jobs if job.is_prefill_done()] \
                if self.target_recurrent else None
        try:
            for idx in range(window):
                params = {
                    "target_hidden": temp_hidden,
                    "attn_mode": "flash_attn",
                    "block_table": block_index,
                    "cache": self.cache if target_kv else self.draft_cache,
                    "cache_seqlens": kv_seqlens if target_kv else cache_seqlens,
                    "draft_step": idx,   # heads specialized per depth pick their head from this
                    "sparse_head": self._sparse_flag,
                }
                if target_kv:
                    params["positions"] = kv_seqlens
                    if target_states is not None:
                        params["source_recurrent_states"] = target_states
                if cal is not None:
                    params["export_draft_conf"] = True
                if spec_prm is not None:
                    params["export_draft_logits"] = True
                if draft_emb is not None:
                    params["draft_embedding"] = draft_emb
                if idx == 0 and n_pend and _mtp_fuse_catchup() == 2:
                    params["dflash_verify"] = True
                batch_state = self.draft_model.forward(first_ids if idx == 0 else batch_ids, params)
                if idx == 0 and n_pend:
                    batch_state = batch_state[:, -1:, :]
                lm_head = self.model.modules[self.model.logit_layer_idx]
                batch_state = lm_head.prepare_for_device(batch_state, params)
                new_ids = self.draft_model.sample_from_state(batch_state, params)
                if spec_prm is not None:
                    dl = params.pop("draft_logits", None)
                    if dl is None:
                        spec_prm = self._spec_round = None
                    else:
                        q = spec_sample.law(dl[0, -1, :], spec_prm)
                        spec_q.append(q)
                        new_ids = spec_sample.draw(q, spec_job.rng.random()).view(1, 1)
                new_ids = self._bound_draft_ids(new_ids)
                draft_cols.append(new_ids)
                draft_emb = params.get("draft_embedding")
                # Without a row from the sparse head, the next step looks the token up in the embedding table,
                # which wants the ids on its own device (a host round trip for a CPU-resident table, as before)
                batch_ids = new_ids if draft_emb is not None else to_device(new_ids, embed_device)
                cache_seqlens += 1 + (n_pend if idx == 0 else 0)
                # Drafters that re-project their state for the next step hand it back through params
                temp_hidden = params["next_target_hidden"] if "next_target_hidden" in params else batch_state
                draft_conf = params.get("draft_conf")
                if cal is not None and draft_conf is not None:
                    c = draft_conf.float().cpu()
                    conf_cols.append(c)
                    est = [cal.estimate(v) for v in c.view(-1).tolist()]
                    reach = est if reach is None else [r * e for r, e in zip(reach, est)]
                    if idx + 1 < window and max(reach) < cal.confidence:
                        window = idx + 1
                        break
        finally:
            # A draft policy that stops the loop (spec_policy StopDraft) verifies the prefix drafted so far
            if draft_cols:
                self.draft_ids_pinned[:batch_size, :len(draft_cols)].copy_(torch.cat(draft_cols, dim = 1))

        if conf_cols and len(conf_cols) == window:
            self._draft_conf_round = {
                "ids": self.draft_ids_pinned[:batch_size, :window],
                "conf": torch.cat(conf_cols, dim = 1),
                "window": window,
            }

        return self.draft_ids_pinned[:, :window]


    def _flush_mtp_pending(self, job):
        """Run a deferred MTP catch-up (EXL3_MTP_FUSE_CATCHUP) as its own draft prefill, for rounds
        where the next draft forward does not consume it. Restores the exact unfused state."""
        pend = getattr(job, "mtp_pending", None)
        if pend is None:
            return
        job.mtp_pending = None
        ids, bi, sl = pend
        n = ids.shape[-1]
        th = job.mtp_last_hidden
        if th is None:
            return
        self.draft_model.prefill(ids, {
            "attn_mode": "flash_attn",
            "block_table": bi,
            "cache": self.draft_cache,
            "cache_seqlens": sl,
            "target_hidden": th[:, :n, :],
        })
        job.mtp_last_hidden = th[:, n:n + 1, :].clone()


    def _try_defer_dkv(self, draft_tokens, batch_jobs, export_states, cache_seqlens, block_index,
                       accepted_lengths, completed_jobs, requeuing_jobs, rewound_jobs) -> bool:
        # (a recurrent drafter's state advances with each write, which the batched flush cannot replay)
        if not self.lazy_dkv or self.draft_recurrent or draft_tokens is not None or len(batch_jobs) != 1:
            return False
        job = batch_jobs[0]
        if len(job.sequences) != 1 or accepted_lengths != [1] or \
                job in completed_jobs or job in requeuing_jobs or id(job) in rewound_jobs:
            return False
        pend = self._dkv_pending
        if pend is not None and pend["job"] != id(job):
            self._flush_dkv()
            pend = None
        rows = [t[:, :1].clone() for t in export_states]
        if pend is None:
            pend = self._dkv_pending = {"job": id(job), "rows": [[r] for r in rows], "n": 0,
                                        "seqlens": cache_seqlens[:1].clone()}
        else:
            for acc, r in zip(pend["rows"], rows):
                acc.append(r)
        pend["n"] += 1
        pend["block_table"] = block_index[:1].clone()
        if pend["n"] >= 64:
            self._flush_dkv()
        return True


    def _flush_dkv(self):
        pend = self._dkv_pending
        if pend is None:
            return
        self._dkv_pending = None
        hidden = [torch.cat(acc, dim = 1) for acc in pend["rows"]]
        self.draft_model.update_kv_from_target(
            target_hidden = hidden,
            cache = self.draft_cache,
            lengths = [pend["n"]],
            params = {"block_table": pend["block_table"], "cache_seqlens": pend["seqlens"]},
        )


    # TODO: Refactor, share code with other draft fns
    def iterate_draftmodel_dflash_gen(self, results: list, window: int):

        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(
                max_seq_len,
                job.get_max_seq_len() + window + 1,
                job.get_max_seq_len() + self.draft_model.config.block_size + 1
            )
            batch_size += 1
        if batch_size == 0:
            return None

        # The diffusion drafter always runs at its fixed block size; the row ceiling and the dynamic
        # window only truncate the verified part of the drafted block

        block_index, cache_seqlens = self._draft_batch_tables(batch_size, max_seq_len)

        # Collect input IDs
        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        # Run draft model. A drafter with sliding-window states appends its block to each row's
        # window ring; the rows the verification accepts are written over it afterwards
        # (update_kv_from_target below), which also rolls the state back to the K/V position
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.draft_cache,
            "cache_seqlens": cache_seqlens,
        }
        if self.draft_recurrent:
            params["recurrent_states"] = [job.recurrent_state.draft for job in self.active_jobs if job.is_prefill_done()]
        if self.draft_calibrator is not None:
            params["export_draft_conf"] = True
        self._flush_dkv()
        _t0 = _pt()
        params["sparse_head"] = self._sparse_flag
        out_state = self.draft_model.forward(
            input_ids = batch_ids,
            params = params,
        )
        _t1 = _pt()
        new_ids = self.draft_model.sample_from_state(out_state, params)
        _t2 = _pt()
        if SPEC_PROF:
            _PROF_CUR["t_dfwd"] = _t1 - _t0
            _PROF_CUR["t_dsamp"] = _t2 - _t1

        # Draft models with a confidence head cap the usable draft length per round;
        # 0 means no draft position cleared the threshold, so skip drafting entirely
        conf_len = params.get("draft_confidence_len")
        if conf_len is not None:
            if conf_len == 0:
                return None
            window = min(window, conf_len)

        # Crop out the first token after sampling to keep batch contiguous for lm_head
        new_ids = new_ids[:, 1:]

        # Confidence-calibrated truncation: cut the batch window at the first position whose
        # drafter confidence falls below the calibrated threshold, taking the longest cut across
        # rows (jobs wanting less just reject earlier). The full block and
        # its scores are kept so iterate_gen can label the verified positions afterwards
        draft_conf = params.get("draft_conf")
        if self.draft_calibrator is not None and draft_conf is not None:
            conf = draft_conf[:batch_size, 1:].float().cpu()
            ids_full = new_ids[:batch_size, :].cpu()
            thr = self.draft_calibrator.threshold()
            below = conf[:, :window] < thr
            any_below = below.any(dim = 1)
            first_below = below.int().argmax(dim = 1)
            cuts = torch.where(any_below, first_below, torch.full_like(first_below, window))
            w_used = int(cuts.max().item())
            self._draft_conf_round = {"ids": ids_full, "conf": conf, "window": w_used}
            if w_used == 0:
                return None
            window = w_used

        self.draft_ids_pinned[:batch_size, :window].copy_(self._bound_draft_ids(new_ids[:batch_size, :window]))
        return self.draft_ids_pinned[:, :window]


    def _draft_batch_tables(self, batch_size: int, max_seq_len: int):
        """
        Block table and cache lengths for one draft round, in pinned staging.

        Same contract as iterate_gen's: pinned, so every upload of them is
        stream-ordered instead of a blocking pageable copy, and the width padded
        to 16 pages, so a growing context crosses a width boundary once every
        PAGE_SIZE * 16 tokens instead of every PAGE_SIZE -- which is what keeps
        one captured decode-graph signature alive across a generation.
        """
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1
        return block_index, cache_seqlens


    def iterate_ngram_gen(self, results: list):

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Generate draft
        window = self.num_draft_tokens
        draft_ids = []
        min_len = window
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            d = job.get_ngram_draft(window)
            min_len = min(min_len, d.shape[-1])
            draft_ids.append(d)

        if min_len == 0:
            return None

        # Trim to minimum length in batch
        draft_ids = torch.cat([d[:, :min_len] for d in draft_ids], dim = 0)
        return draft_ids


    def new_recurrent_state(self) -> PairedState:
        """A fresh (cleared) pair of recurrent states for a new sequence, one side per recurrent model"""
        target = draft = None
        try:
            target = self.cache.get_new_state() if self.target_recurrent else None
            draft = self.draft_cache.get_new_state() if self.draft_recurrent else None
            return PairedState(target, draft)
        except BaseException:
            if draft is not None:
                draft.free()
            if target is not None:
                target.free()
            raise


    def recurrent_state_from_stashed(self, stashed: dict, position: int) -> PairedState:
        """The pair restored from a checkpoint stash (PairedState.stash) at `position`"""
        target = draft = None
        try:
            target = self.cache.new_from_stashed(stashed["target"], position) if self.target_recurrent else None
            draft = self.draft_cache.new_from_stashed(stashed["draft"], position) if self.draft_recurrent else None
            return PairedState(target, draft)
        except BaseException:
            if draft is not None:
                draft.free()
            if target is not None:
                target.free()
            raise


    def row_limited(self, window: int) -> int:
        """
        Trim a draft window so that bsz * (1 + window) rows stay under draft_max_rows (see the constructor).
        """
        if not self.draft_max_rows or not window:
            return window
        rows = 0
        for job in self.active_jobs:
            if job.is_prefill_done():
                rows += len(job.sequences)
        if rows < 1:
            return window
        limited = min(window, max(0, self.draft_max_rows // rows - 1))
        if window == self.num_draft_tokens and limited != self._draft_window_last:
            self._draft_window_last = limited
            logger.debug("draft_max_rows %d, %d rows: draft window %d",
                         self.draft_max_rows, rows, limited)
        return limited

    def iterate_hybrid_gen(self, draft_tokens: torch.Tensor | None, H: int):
        """
        Replace the round's draft with hybrid_draft_tokens tokens of suffix-match continuation when some job's
        current suffix matches at least that many tokens of its earlier context and the continuation agrees
        with the regular draft (the model's own short-range guess). The verification pass is rectangular, so
        when any row takes the long draft every row does: rows without a qualifying match keep their regular
        draft and are padded with whatever continuation their longest match offers, then with repeats of
        their last token, which verification rejects at the first mismatch. Returns the (rows, width) draft
        for iterate_gen, or the regular draft unchanged. H is that length for this round, already trimmed
        to the row ceiling.
        """
        jobs = [job for job in self.active_jobs if job.is_prefill_done()]
        if not jobs or not H:
            return draft_tokens
        rows = []
        fire = False
        for i, job in enumerate(jobs):
            short = draft_tokens[i : i + 1] if draft_tokens is not None else None
            long, matched = job.get_hybrid_draft(H)
            # A qualifying row: a match of at least H tokens, H tokens following it, and no disagreement with
            # the regular drafter over the positions it covers
            ok = matched >= H and long.shape[-1] == H
            if ok and short is not None and short.shape[-1] and not torch.equal(long[:, :short.shape[-1]], short):
                ok = False
            fire = fire or ok
            rows.append((ok, short, long))
        if not fire:
            return draft_tokens

        out = torch.empty((len(jobs), H), dtype = torch.long)
        for i, (ok, short, long) in enumerate(rows):
            if ok:
                out[i] = long[0]
                continue
            parts = []
            if short is not None and short.shape[-1]:
                parts.append(short[0])
            if long.shape[-1]:
                parts.append(long[0])
            fill = torch.cat(parts) if parts else jobs[i].sequences[0].sequence_ids.torch()[0, -1:]
            fill = fill[:H]
            out[i, : fill.shape[-1]] = fill
            out[i, fill.shape[-1] :] = fill[-1]
        # The regular drafter's confidence record no longer describes the verified window
        self._draft_conf_round = None
        self.hybrid_rounds += 1
        return out


    def _sparse_head_update(self, jobs: list, accepted_lengths: list[int]):
        """
        Feed this step's target tokens to the sparse head. A job's first step contributes its whole context
        (prompt and anything generated so far) at prompt weight, since the identifiers a reply reuses are
        mostly already in its prompt; later steps contribute the tokens the target just accepted or sampled
        """
        sh = self.sparse_head
        for job, n in zip(jobs, accepted_lengths):
            seq = job.sequences[0]
            if not job.sparse_head_seeded:
                job.sparse_head_seeded = True
                sh.observe(seq.sequence_ids.torch(), sh.prompt_weight)
            elif n > 0:
                sh.observe(seq.sequence_ids.torch_slice(len(seq.sequence_ids) - n, None), 1.0)
        sh.maintain()


    def _staging(self, name, rows: int, width: int | None = None, dtype = torch.int32):
        """
        Reusable pinned staging buffer, keyed by name and row width, grown by rows on demand.
        """
        key = (name, width)
        buf = self.staging_buffers.get(key)
        if buf is None or buf.shape[0] < rows:
            alloc_rows = max(rows, 32)
            shape = (alloc_rows,) if width is None else (alloc_rows, width)
            buf = torch.zeros(shape, dtype = dtype, pin_memory = True)
            self.staging_buffers[key] = buf
        return buf[:rows]


    def _stage_arena(self, rows: int, width: int, with_positions: bool):
        """
        Pinned int32 arena holding one step's block index [rows, width], cache_seqlens [rows] and optionally
        positions [rows] back to back (the layout ext.generator_stage_decode fills), with its views and a device
        copy per attention device. Keyed by shape, so the views keep their identity from step to step.
        """
        key = (rows, width, with_positions)
        arena = self.stage_arenas.get(key)
        if arena is None:
            if len(self.stage_arenas) >= 32:
                self.stage_arenas.clear()
            n = rows * width + rows * (2 if with_positions else 1)
            host = torch.zeros((n,), dtype = torch.int32, pin_memory = True)
            arena = (host, self._arena_views(host, rows, width, with_positions), {})
            self.stage_arenas[key] = arena
        return arena


    @staticmethod
    def _arena_views(t: torch.Tensor, rows: int, width: int, with_positions: bool):
        nbi = rows * width
        return (
            t[:nbi].view(rows, width),
            t[nbi:nbi + rows],
            t[nbi + rows:nbi + 2 * rows] if with_positions else None,
        )


    def _upload_arena(self, arena, rows: int, width: int, with_positions: bool, params: dict):
        """
        Upload the staged tables to every attention device with one copy each, and seed the forward's per-device
        tensor cache (util.tensor.get_for_device) with the device views, so the layers find them there instead of
        uploading each table separately when first touched. The device arena is reused from step to step: its next
        upload is stream-ordered after the kernels reading this one.
        """
        if self.stage_devices is None:
            # TP forwards ship params to the workers themselves; there the tables stay host-side
            self.stage_devices = [] if self.model.loaded_tp else \
                sorted({torch.device(m.device) for m in self.model.get_cache_layers() if m.device is not None and
                        torch.device(m.device).type == "cuda"}, key = lambda d: d.index)
        if not self.stage_devices:
            return
        host, views, dev = arena
        dev_cache = {}
        for device in self.stage_devices:
            d = dev.get(device)
            if d is None:
                darena = torch.empty_like(host, device = device)
                d = (darena, self._arena_views(darena, rows, width, with_positions))
                dev[device] = d
            darena, dviews = d
            with torch.cuda.device(device):
                ext.cuda_memcpy_async(darena.data_ptr(), host.data_ptr(), host.numel() * 4, 1)
            for hv, dv in zip(views, dviews):
                if hv is not None:
                    dev_cache[(id(hv), device)] = (hv, dv)
        params["dev_cache"] = dev_cache


    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):

        # Get shape of active batch
        # Only jobs that have finished prefill can participate in token generation. The maximum sequence length
        # determines how many cache pages the temporary block table needs for this iteration.
        batch_size = 0
        max_seq_len = 0
        draft_len = draft_tokens.shape[-1] if draft_tokens is not None else 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + draft_len)
            batch_size += len(job.sequences)
        if batch_size == 0:
            return
        if draft_tokens is not None:
            max_seq_len += draft_tokens.shape[-1]

        # Create block index table for batch
        # The model sees a compact batch, so build per-row mappings from logical page positions to physical cache
        # page indices, along with current cache lengths and optional MRoPE position offsets.
        # Block-table width is padded to a multiple of 16 pages so the pinned staging buffers
        # cover a few distinct widths only; the extra (zeroed) columns are never dereferenced
        # since the kernels bound their reads by the cache lengths
        max_pages_batch = block_table_pages(max_seq_len)
        # Verify forwards: the block table width each sequence would have if its job ran alone (the MLA
        # decode kernel's KV split follows the table width, so a row must not see its neighbour's)
        own_pages = None
        if draft_tokens is not None:
            own_pages = []
            for job in self.active_jobs:
                if not job.is_prefill_done(): continue
                own = block_table_pages(job.get_max_seq_len() + self.num_draft_tokens + draft_tokens.shape[-1])
                own_pages += [own] * len(job.sequences)
        use_offsets = "mrope" in self.model.caps
        arena = self._stage_arena(batch_size, max_pages_batch, use_offsets)
        block_index, cache_seqlens, positions = arena[1]

        # Collect the rows' tables, and their input IDs and indexed embeddings. Jobs may contribute multiple
        # sequence rows. logit_mapping records the slice of the model output that belongs to each active job, while
        # batch_jobs keeps the corresponding Job objects in compact-batch order. Single-token decode rows (no draft)
        # have their input token gathered and recorded in its cache page by the staging call below, together with
        # the tables; any other row takes the general path through get_input_ids_list
        seq_bi, seq_kvp, seq_rope = [], [], []
        dec_ids, dec_pages, dec_offsets = [], [], []
        decode_rows = draft_tokens is None
        active_embeddings = []
        logit_mapping = []
        batch_jobs = []
        rows = 0
        for job in self.active_jobs:
            logit_mapping.append(rows)
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            for seq in job.sequences:
                kvp = seq.kv_position
                seq_bi.append(seq.block_index_tensor)
                seq_kvp.append(kvp)
                if use_offsets:
                    seq_rope.append(job.alt_rope_offset)
                if decode_rows:
                    decode_rows = len(seq.sequence_ids) - kvp == 1
            rows += len(job.sequences)
            batch_jobs.append(job)
            active_embeddings += job.embeddings
        logit_mapping.append(rows)

        if decode_rows:
            for job in batch_jobs:
                for seq in job.sequences:
                    page = seq.allocated_pages[seq.kv_position // PAGE_SIZE]
                    assert page.ref_count == 1
                    dec_ids.append(seq.sequence_ids.tensor)
                    dec_pages.append(page.sequence)
                    dec_offsets.append(page.kv_position)
                    page.kv_position += 1
                    page.can_revert = False
            batch_ids = self._staging("batch_ids", rows, 1, torch.long)
        else:
            batch_ids = None
        ext.generator_stage_decode(
            arena[0], max_pages_batch, seq_bi, seq_kvp, seq_rope, batch_ids, dec_ids, dec_pages, dec_offsets
        )
        if not decode_rows:
            input_ids_list = []
            for job in batch_jobs:
                input_ids_list += job.get_input_ids_list(draft_tokens, len(input_ids_list), add_to_cache = True)
            ids_width = input_ids_list[0].shape[-1]
            if all(ids.shape[-1] == ids_width for ids in input_ids_list):
                ids_staging = self._staging("batch_ids", len(input_ids_list), ids_width, torch.long)
                batch_ids = torch.cat(input_ids_list, dim = 0, out = ids_staging)
            else:
                batch_ids = torch.cat(input_ids_list, dim = 0)

        # Collect recurrent states for batch
        # Recurrent models carry mutable state beside the K/V cache; pass one state object per compact batch job so
        # the model can advance or rewind it consistently with accepted draft tokens.
        batch_states = None
        if self.target_recurrent:
            batch_states = [job.recurrent_state.target for job in batch_jobs]

        # GPU workload is scheduled here, so launch any sampling filters that can run in the background
        for job in batch_jobs:
            if job.new_tokens < 0: continue
            assert len(job.filter_futures) == 0
            for f in job.filters:
                if not f.is_active: continue
                if f.use_background_worker():
                    job.filter_futures.append(self.filter_pool.submit(f.get_next_logit_mask))
                else:
                    job.filter_futures.append(None)

        # Get logit batch from model. Forward writes new K/V entries into the cache for the supplied positions and
        # returns logits for either one target token or a target-plus-draft verification window.
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.cache,
            "cache_seqlens": cache_seqlens,
            "recurrent_states": batch_states,
            "indexed_embeddings": active_embeddings,
            "positions": positions,
            "recurrent_history": draft_tokens is not None,
            "pinned_staging": True,
            #  marks a real DFlash verify forward (R rows, target model), as opposed to
            # any other multi-row call with rows in the same 2..MAX_BSZN range -- prefill routes
            # 2-8 tokens to plenty of individual MoE experts, and the draft model's own forward
            # has draft_tokens is None too. The four R-row fast paths (block_sparse_mlp.py's MoE
            # union branch and dec_norm_route_r's R>1 branch, quant/exl3.py's dec_gemv_r branch,
            # sliding_attn.py's qkv gemv_r_multi branch) all require this marker in addition to
            # the row-count bound, so they only ever fire on the one call they were verified
            # bit-exact against.
            "dflash_verify": draft_tokens is not None,
        }
        if own_pages is not None:
            params["dflash_table_pages"] = own_pages
        self._upload_arena(arena, batch_size, max_pages_batch, use_offsets, params)
        if self.draft_model:
            params.update(self.draft_model.draft_verifier_params)
        _tv0 = _pt()
        batch_logits = self.model.forward(
            input_ids = batch_ids,
            params = params,
        )
        _tv1 = _pt()
        if SPEC_PROF:
            _PROF_CUR["t_verify"] = _tv1 - _tv0
            _PROF_CUR["R"] = batch_ids.shape[-1]

        # Keep only the fields needed below for draft-cache updates and drop the params dict so it cannot extend
        # references to recurrent state objects past this iteration.
        p_export_states = params.get("export_states")
        p_cache_seqlens = params.get("cache_seqlens")
        params = None

        # Background filters were launched before forward(); synchronous filters run now to overlap CPU work with
        # queued GPU execution as much as possible.
        for job in batch_jobs:
            if job.new_tokens < 0: continue
            assert len(job.logit_masks) == 0
            for f in job.filters:
                if not f.is_active: continue
                if f.use_background_worker():
                    job.logit_masks.append(None)
                else:
                    job.logit_masks.append(f.get_next_logit_mask())

        # Prepare past IDs (for sequences that need them for repetition penalty etc.) and logit masks where needed
        for job in batch_jobs:
            job.prepare_logit_mask()
            job.prepare_sampling_past_ids()

        # TODO: Batch sampling
        # Pass to jobs to sample
        # Sampling is still job-by-job. Each accepted token updates job state, may emit a stream result, may finish
        # the job, or may request requeueing once the configured per-job token budget is reached.
        completed_jobs = []
        requeuing_jobs = []
        accepted_lengths = []
        rewound_jobs = set()
        j = 0

        # Reject the trailing draft positions after the last accepted token at index i: count them, roll back the
        # job's recurrent state and return cache pages to the accepted position
        def reject_remainder(job_, j_, i_, batch_states_):
            num_rejected = batch_logits.shape[1] - 1 - i_
            if num_rejected == 0:
                return 0
            job_.rejected_draft_tokens += num_rejected

            # Rewind recurrent states
            if batch_states_ is not None:
                batch_states_[j_].rewind(num_rejected)

            # Rewind cache position (draft model cache layout is always the same as target)
            for seq_ in job_.sequences:
                r = num_rejected
                while r:
                    pos = seq_.kv_position + r
                    page = seq_.allocated_pages[(pos - 1) // PAGE_SIZE]
                    rp = min(page.kv_position, r)
                    page.kv_position -= rp
                    r -= rp
            return num_rejected

        # Without a draft, each job samples exactly one independent token, so all sampler chains
        # are launched first and the results collected in a second pass: the first collect
        # absorbs the whole GPU tail and the remaining reads are cheap, instead of one full
        # launch-to-readback round trip per job. Sampler launch order (and with it each job's
        # RNG draw) matches the serial loop exactly. Per-job constraints (filters, penalties)
        # are unaffected because no job's token depends on another job's result.
        if draft_tokens is None and batch_logits.shape[1] == 1:
            # Single-token results stage through one pinned buffer and one synchronize, so the
            # batch pays one launch-to-readback round trip instead of one per job
            pinned = self.sample_pinned
            if pinned is None or pinned.shape[0] < batch_size:
                self.sample_pinned = pinned = torch.empty(
                    (max(batch_size, 32), 1), dtype = torch.long, pin_memory = True
                )
            launched = []
            for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a == b: continue
                token_logits = batch_logits[a:b, :, :]
                sampled = job.receive_logits(token_logits)
                next_token = sampled[0]
                if next_token.is_cuda and next_token.numel() == 1:
                    k = len(launched)
                    pinned[k:k + 1].copy_(next_token.view(1, 1), non_blocking = True)
                    sampled = (pinned[k:k + 1],) + sampled[1:]
                launched.append((job, token_logits, sampled))
            if launched:
                torch.cuda.synchronize(batch_logits.device)
                # The sampled tokens are the next forward's inputs
                self.model.prefetch_tokens([t for _, _, s in launched for t in s[0].view(-1).tolist()])

            for job, token_logits, (next_token, next_k_tokens, next_k_probs, next_prob) in launched:
                eos, sampled_token, rq = job.receive_sample(
                    token_logits,
                    next_token,
                    next_k_tokens,
                    next_k_probs,
                    next_prob,
                    results,
                )
                # Single-token rewinds need no batch cleanup, but the post-batch draft carry update must still be
                # suppressed so it doesn't resurrect a stale MTP state
                if job.checkpoint_rewound:
                    job.checkpoint_rewound = False
                    rewound_jobs.add(id(job))

                # Requeue. Requeueing is only supported for single-sequence jobs because the replacement job
                # uses the full sequence generated so far as its next prompt.
                if len(job.sequences) == 1 and rq:
                    requeuing_jobs.append(job)
                elif eos:
                    completed_jobs.append(job)
                accepted_lengths.append(1)

        # With a draft, positions within a job are consumed strictly serially: acceptance of
        # position n gates position n+1, and constrained-decoding masks and sampling past IDs
        # must advance between positions.
        else:
            for idx, (job, a, b) in enumerate(zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:])):
                if a == b: continue
                job_logits = batch_logits[a:b, :, :]
                accepted_length = 1
                rejected = 0
                W = batch_logits.shape[1]

                if os.environ.get("DFLASH_DEBUG") and draft_tokens is not None:
                    print(f"[dflash_debug] job.new_tokens={job.new_tokens} j={j} "
                          f"draft_row={draft_tokens[j].tolist()}")
                # Sampled-draft round: the draft positions take the spec_sample keep / residual rule, only the last
                # (bonus) position samples from the target as usual. Applies only to drafts read from the drafter's own
                # buffer, for which this round recorded a law per position (a lookup round brings its own tensor of
                # copied tokens: a deterministic draft keeps the argmax rule)
                spec = None
                sr = self._spec_round
                if sr is not None and sr[0] is job and draft_tokens is not None and len(sr[2]) >= W - 1 and \
                        draft_tokens.data_ptr() == self.draft_ids_pinned.data_ptr():
                    spec = [(spec_sample.verify(job_logits[0, pi, :], sr[2][pi], int(draft_tokens[j, pi]), sr[1], job.rng),
                             None, None, None) for pi in range(W - 1)]
                # Positions within a job are independent when no per-position constraint advances between
                # them: no active filter (masks are rebuilt per accepted position), no sampling past IDs
                # (repetition penalty), no banned-string checkpoint, and outside the min_new_tokens mask
                # window (token healing, new_tokens < 0, is covered by the last test). Then all samplers
                # launch up front in serial-loop order (each position's RNG draw is unchanged) and the W
                # sampled tokens read back through one pinned buffer and one synchronize, as in the
                # no-draft path, leaving a host-only accept/reject walk. Constrained jobs keep the serial
                # loop below.
                # A dormant triggered filter can activate on a token inside the window and constrain the
                # positions after it, and a pending constrain_output_now injection is consumed one token per
                # sampled position; both keep the serial loop
                batchable = (
                    not _NO_BATCH_READBACK
                    and b - a == 1
                    and job.checkpoint is None
                    and job.forced_ids is None
                    and job.new_tokens >= job.min_new_tokens
                    and not any(f.is_active or f.trigger_token is not None for f in job.filters)
                    and not job.sampler.reqs_past_ids
                )
                staged = None
                if batchable:
                    rng_state = job.rng.getstate()
                    staged = []
                    any_cuda = False
                    for i in range(W):
                        token_logits = job_logits[:, i:i + 1, :]
                        sampled = spec[i] if spec is not None and i < W - 1 else job.receive_logits(token_logits)
                        is_cuda = sampled[0].is_cuda and sampled[0].numel() == 1
                        staged.append((sampled, is_cuda))
                        any_cuda |= is_cuda
                    if any_cuda:
                        pinned_w = getattr(self, "draft_sample_pinned", None)
                        if pinned_w is None or pinned_w.shape[1] < W:
                            pinned_w = torch.empty((1, max(W, 8)), dtype = torch.long, pin_memory = True)
                            self.draft_sample_pinned = pinned_w
                        for i, (sampled, is_cuda) in enumerate(staged):
                            if is_cuda:
                                pinned_w[0:1, i:i + 1].copy_(sampled[0].view(1, 1), non_blocking = True)
                                staged[i] = ((pinned_w[0:1, i:i + 1],) + sampled[1:], False)
                        torch.cuda.synchronize(batch_logits.device)

                for i in range(W):
                    positions_used = i + 1
                    token_logits = job_logits[:, i:i + 1, :]
                    if staged is not None:
                        (next_token, next_k_tokens, next_k_probs, next_prob), _ = staged[i]
                    elif spec is not None and i < W - 1:
                        next_token, next_k_tokens, next_k_probs, next_prob = spec[i]
                    else:
                        next_token, next_k_tokens, next_k_probs, next_prob = job.receive_logits(
                            token_logits,
                        )
                    eos, sampled_token, rq = job.receive_sample(
                        token_logits,
                        next_token,
                        next_k_tokens,
                        next_k_probs,
                        next_prob,
                        results,
                    )
                    if os.environ.get("DFLASH_DEBUG") and draft_tokens is not None:
                        dt = draft_tokens[j, i].item() if i < draft_tokens.shape[-1] else None
                        print(f"[dflash_debug]   i={i} sampled={sampled_token.item()} draft_tok={dt}")

                    # Requeue. Requeueing is only supported for single-sequence jobs because the replacement job
                    # uses the full sequence generated so far as its next prompt. Unconsumed draft positions must
                    # be rejected here so the recurrent state and page positions match the accepted sequence
                    # before the requeue stash.
                    if len(job.sequences) == 1 and rq:
                        if draft_tokens is not None:
                            rejected = reject_remainder(job, j, i, batch_states)
                        requeuing_jobs.append(job)
                        break

                    # EOS. Stop sampling this job immediately once a stop condition, filter condition or max
                    # token limit produces an EOS event. Draft positions from the EOS position onward were fed
                    # to verification but never resolved; count them as rejected so accepted + rejected equals
                    # the number of drafted positions (no cache/state rollback needed, the pages are released).
                    # The EOS stream event was already built inside receive_sample, so patch its counter too
                    if eos:
                        num_unresolved = batch_logits.shape[1] - 1 - i if draft_tokens is not None else 0
                        # EXL3_TAIL_REPLY: roll the recurrent state and page positions back to the accepted
                        # position (as a rejection would) so the finished job can stash an exact reply-end tail
                        if _TAIL_REPLY_GEN and draft_tokens is not None and batch_states is not None \
                                and len(job.sequences) == 1:
                            batch_states[j].rewind(num_unresolved)   # rewind(0) too: it normalises the per-token history
                            r_ = num_unresolved
                            seq_ = job.sequences[0]
                            while r_:
                                page = seq_.allocated_pages[(seq_.kv_position + r_ - 1) // PAGE_SIZE]
                                rp = min(page.kv_position, r_)
                                page.kv_position -= rp
                                r_ -= rp
                        if num_unresolved:
                            job.rejected_draft_tokens += num_unresolved
                            for r_ in reversed(results):
                                if r_.get("eos") and r_["job"] is job:
                                    if "rejected_draft_tokens" in r_:
                                        r_["rejected_draft_tokens"] = job.rejected_draft_tokens
                                    break
                        completed_jobs.append(job)
                        break

                    # A banned-string match inside receive_sample just rewound the job, resetting cache pages and
                    # rolling back or replacing any recurrent state. The remaining logit positions extend a
                    # sequence that no longer exists, so abandon them; everything is already consistent, and the
                    # usual rejection rollback and state normalization must not run on top of the rewind.
                    if job.checkpoint_rewound:
                        job.checkpoint_rewound = False
                        rewound_jobs.add(id(job))
                        if draft_tokens is not None:
                            job.rejected_draft_tokens += batch_logits.shape[1] - 1 - i
                            rejected = -1
                        break

                    # Continue sampling from logit batch as long as result matches draft, unless hitting
                    # checkpoint mark. For speculative decoding, consume additional logits only while the
                    # sampled target token matches the draft token. A recurrent checkpoint boundary also stops
                    # draft acceptance so state can be stashed at an exact page boundary.
                    if draft_tokens is not None and i < batch_logits.shape[1] - 1:
                        cp_boundary = batch_states is not None and job.is_checkpoint_boundary()
                        if draft_tokens[j, i].item() != sampled_token.item() or cp_boundary:
                            rejected = reject_remainder(job, j, i, batch_states)
                            break

                        # Accept draft token
                        else:
                            job.accepted_draft_tokens += 1
                            accepted_length += 1

                            # Advance filters
                            for f in job.filters:
                                if not f.is_active: continue
                                # Both lists take one entry per active filter (prepare_logit_mask indexes them
                                # in step)
                                if f.use_background_worker():
                                    job.filter_futures.append(self.filter_pool.submit(f.get_next_logit_mask))
                                    job.logit_masks.append(None)
                                else:
                                    job.logit_masks.append(f.get_next_logit_mask())
                                    job.filter_futures.append(None)

                            # Update masks and past IDs
                            job.prepare_logit_mask()
                            job.prepare_sampling_past_ids()

                if staged is not None and spec is None and positions_used < W:
                    # The serial loop would have drawn one sampler seed per position it reached; pass 1 drew
                    # W. Rewind the job's RNG and replay only the draws that were used, so a seeded stream
                    # continues exactly as it would have on the serial loop
                    job.rng.setstate(rng_state)
                    for _ in range(positions_used):
                        job.rng.randint(0, (1 << 32) - 1)

                # Make sure outgoing state is valid if entire draft was accepted
                if batch_states and draft_tokens is not None and rejected == 0:
                    batch_states[j].rewind(0)

                # Record per-round draft stats. Skip abandoned windows (banned-string rewind);
                # checkpoint-boundary truncations are rare enough to count as ordinary rejections.
                if draft_tokens is not None and rejected != -1:
                    if self.record_draft_stats:
                        job.draft_stats.append((
                            job.new_tokens,
                            draft_tokens.shape[-1],
                            accepted_length - 1,
                        ))

                # The last sampled token leads the next forward
                self.model.prefetch_tokens([sampled_token.item()])

                accepted_lengths.append(accepted_length)
                j += 1

        if self.sparse_head is not None:
            self._sparse_head_update(batch_jobs, accepted_lengths)

        # Update the draft-confidence calibration with this round's verification outcomes (any
        # draft-model mode)
        if self.draft_calibrator is not None and self._draft_conf_round is not None:
            st = self._draft_conf_round
            self._draft_conf_round = None
            cal = self.draft_calibrator
            row = 0
            for job, a_idx, b_idx in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx: continue
                accepted_length = accepted_lengths[row]
                conf = st["conf"][row]
                ids_full = st["ids"][row]
                row += 1
                if id(job) in rewound_jobs:
                    continue
                a = accepted_length - 1
                for i in range(a):
                    cal.add_label(conf[i].item(), True)
                if a < conf.shape[-1]:
                    seq = job.sequences[0]
                    n = len(seq.sequence_ids)
                    tail = seq.sequence_ids.torch_slice(n - 1, n).item()
                    cal.add_label(conf[a].item(), ids_full[a].item() == tail)
            cal.decay_step()

        # Settle a recurrent drafter's speculative pass: it consumed the round's input token and every
        # drafted token (last_history of them); the rejected drafts are undone, leaving its state at
        # the accepted prefix, i.e. at the new K/V position (or behind it after a long hybrid round)
        if self.draft_recurrent and draft_tokens is not None:
            accepted_idx = 0
            for job, a_idx, b_idx in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx:
                    continue
                accepted_drafts = accepted_lengths[accepted_idx] - 1
                accepted_idx += 1
                if id(job) in rewound_jobs or job.recurrent_state is None:
                    continue   # a checkpoint rewind replaced the pair
                ds = job.recurrent_state.draft
                ds.rewind(max(0, ds.last_history - accepted_drafts))

        # Accept new target_hidden if DFlash. DFlash draft models can update their cache from target-model hidden
        # states for the tokens accepted above, keeping draft and target cache layouts aligned.
        _tk0 = _pt()
        if self.dflash_draft and self._try_defer_dkv(
                draft_tokens, batch_jobs, p_export_states, p_cache_seqlens, block_index,
                accepted_lengths, completed_jobs, requeuing_jobs, rewound_jobs):
            pass
        elif self.dflash_draft:
            self._flush_dkv()
            update_params = {
                "block_table": block_index,
                "cache_seqlens": p_cache_seqlens,
            }
            lengths = accepted_lengths
            if self.draft_recurrent:
                # A row whose pair a checkpoint rewind replaced gets its window from the replay
                # prefill instead; its length 0 skips the write (and the rollback)
                drafting = [job for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]) if a != b]
                update_params["recurrent_states"] = [job.recurrent_state.draft for job in drafting]
                lengths = [0 if id(job) in rewound_jobs or job.recurrent_state is None else n
                           for job, n in zip(drafting, accepted_lengths)]
            self.draft_model.update_kv_from_target(
                target_hidden = p_export_states,
                cache = self.draft_cache,
                lengths = lengths,
                params = update_params,
            )
        if SPEC_PROF:
            _PROF_CUR["t_kvup"] = _pt() - _tk0
            _PROF_CUR["t_post"] = _tk0 - _tv1

        # Accept new target_hidden if MTP. MTP draft models can update their cache from target-model hidden
        # states for the tokens accepted above, keeping draft and target cache layouts aligned.
        if self.mtp_draft:
            target_hidden = p_export_states[-1]
            accepted_idx = 0
            for job in batch_jobs:
                self._flush_mtp_pending(job)
            for job, a_idx, b_idx in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx:
                    continue
                accepted_length = accepted_lengths[accepted_idx]
                accepted_idx += 1

                # A banned-string rewind invalidated this job's carry; leave it unset so drafting pauses until the
                # next target forward provides a fresh one, and don't propagate hidden states from the abandoned
                # window into the draft cache
                if id(job) in rewound_jobs:
                    continue

                # Position K was drafted from the last target state already. Replace accepted
                # speculative positions K+1..K+A-1 with the corresponding target-state inputs.
                # EXL3_MTP_FUSE_CATCHUP: defer these rows into the next draft forward (one R=A
                # draft-head pass instead of a catch-up prefill plus a 1-row draft pass). Only the
                # draft cache and draft ids depend on it; target ids are unaffected.
                if accepted_length > 1 and _mtp_fuse_catchup() and len(job.sequences) == 1 and \
                        job not in completed_jobs and job not in requeuing_jobs:
                    job.mtp_pending = (
                        batch_ids[a_idx:b_idx, 1:accepted_length].clone(),
                        block_index[a_idx:b_idx].clone(),
                        p_cache_seqlens[a_idx:b_idx] + 1,
                    )
                    job.mtp_last_hidden = target_hidden[a_idx:b_idx, :accepted_length, :].clone()
                    continue
                if accepted_length > 1:
                    self.draft_model.prefill(
                        batch_ids[a_idx:b_idx, 1:accepted_length],
                        {
                            "attn_mode": "flash_attn",
                            "block_table": block_index[a_idx:b_idx],
                            "cache": self.draft_cache,
                            "cache_seqlens": p_cache_seqlens[a_idx:b_idx] + 1,
                            "target_hidden": target_hidden[a_idx:b_idx, :accepted_length - 1, :],
                        },
                    )

                # The next unprocessed token is paired with the preceding target hidden state.
                job.mtp_last_hidden = target_hidden[
                    a_idx:b_idx, accepted_length - 1:accepted_length, :
                ].clone()

        # Release pages for completed jobs. Finished and requeued jobs no longer need their active page references.
        # Requeued recurrent jobs may stash the last checkpoint first so the next queued job can resume from cached
        # recurrent state.
        num_jobs = self.num_remaining_jobs()
        for job in completed_jobs + requeuing_jobs:
            if job in requeuing_jobs and self.recurrent_cache is not None:
                # 🔴 FIFTH FAILURE SITE, one line before the requeue guard added by
                # 05-contain-requeue-failure.patch and reachable on the same code path: a job that yields
                # at max_rq_tokens stashes its last checkpoint here so the replacement job can resume from
                # it. Same asserts, same blast radius (escapes iterate() -> latches self.error -> generator
                # dead for all). Skipping only costs the replacement job a re-prefill from an earlier
                # checkpoint, so containment here is free.
                try:
                    job.maybe_stash_recurrent(self.recurrent_cache, PAGE_SIZE)
                except Exception as e:
                    print(f" ## Warning: recurrent stash before requeue failed for job "
                          f"{getattr(job, 'serial_number', '?')} ({repr(e)}); the requeued "
                          f"job will re-prefill from an earlier checkpoint")
            if _REPLY_FIRST and job not in requeuing_jobs:
                self.deferred_jobs.append(job)   # released by flush_deferred() after the results are out
            else:
                job.deallocate_pages()
            self.active_jobs.remove(job)

        # Requeue jobs. Puts replacement jobs at the front so long generations continue promptly after they yield
        # their cache pages.
        for job in requeuing_jobs:
            # 🔴 THE REQUEUE IS A THIRD FAILURE SITE, and the one that actually killed
            # the service on 2026-09-05. prepare_for_requeue() builds the replacement
            # job and calls prepare_for_queue(..., rq = True), which ends in
            #     assert total_pages <= max_pages, "... cannot be enqueued"
            # So a long generation that yields at max_rq_tokens and whose REMAINING
            # max_new_tokens no longer fits the cache raises HERE, inside iterate(),
            # from a call the upstream #318 containment does not wrap (it guards
            # allocate_pages() and prefill() only). The exception escapes to
            # AsyncGenerator._run_iteration, which latches self.error, ends the single
            # iteration task serving EVERY job, and hands that exception to all of them.
            #
            # Measured: the failing request ran ~100 s (103.7 s and 106.0 s in two
            # independent runs) before the requeue, then the whole service went down —
            # which is why this looked like a mysterious ~100 s delay rather than a
            # rejection, and why it survived both the enqueue-path fix (patch 04) and
            # the TabbyAPI recreate fix (tabbyAPI patch 05).
            #
            # reap_failed_job is safe here even though the loop above already ran
            # deallocate_pages() and removed the job from active_jobs: deallocate_pages
            # skips sequences whose allocated_pages is None, and the removal is guarded
            # by `if job in self.active_jobs`.
            self.requeue_one(job, results)

        # Defrag. Physical page indices can only be compacted when no active block tables are using them.
        if num_jobs and not self.num_remaining_jobs() and not self.deferred_jobs:
            self.on_queue_drained()


    # ── Score-only jobs ─────────────────────────────────────────────────────────────────────────────────
    #
    # A score job wants the logits of the one position after its prompt. The regular path gets there with
    # a prefill of all but the last prompt token, a full generation round for the last one (with MTP: three
    # draft forwards and a four-wide verify whose drafts are discarded) and sampling. This path runs the
    # uncached prompt tail as ordinary forwards and reads the last position's logits from the final one.
    #
    # Scheduling: score jobs never enter pending_jobs or active_jobs. iterate() advances them first, FIFO,
    # within one prefill chunk's worth of tokens per round, so a score request waits only for the round in
    # flight when it arrives. They do not count against max_batch_size; they do need a free recurrent state
    # slot, and wait for one if every slot is taken.
    #
    # Containment: everything below runs inside iterate_score()'s try. A failure reaps that one job, as a
    # failed prefill would, and a failed checkpoint stash is skipped with a warning, as in patch 06.

    def iterate_score(self, results: list):
        num_jobs = self.num_remaining_jobs()
        budget = self.max_chunk_size
        while self.score_jobs and budget > 0:
            job = self.score_jobs[0]
            try:
                if not job.score_started:
                    if not self.score_can_start(job):
                        break
                    self.score_start(job)
                budget -= self.score_advance(job, budget, results)
            except Exception as e:
                if job in self.score_jobs:
                    self.score_jobs.remove(job)
                job.score_keep_pages = False
                self.score_retire(job)
                self.reap_failed_job(job, e, results)
                continue
            if not job.score_done:
                break
            self.score_jobs.remove(job)
            # Hand a finished request back now rather than after every request queued behind it: results only
            # reach their consumers when iterate() returns. A group is one request, so it finishes as a unit
            if self.score_jobs and (job.score_group is None or self.score_jobs[0].score_group is not job.score_group):
                break
        if num_jobs and not self.num_remaining_jobs():
            self.on_queue_drained()


    def score_can_start(self, job) -> bool:
        if self.recurrent_cache is not None and not self.cache.free_list:
            return False
        return job.current_new_pages_required() <= self.pagetable.num_unreferenced_pages()


    def _score_t(self):
        if self.score_trace:
            torch.cuda.synchronize()
        return time.perf_counter()


    def score_start(self, job):
        """
        Claim pages and a recurrent state, and plan the forwards. Pages come from the same prefix lookup as a
        regular job (host and durable tiers included); the resume point is the longest cached prefix with a
        checkpoint, or the job's group's shared stash when that reaches further.
        """
        t0 = self._score_t()
        job.score_started = True
        job.score_t0 = time.time()
        job.activate()
        seq = job.sequences[0]
        rc = self.recurrent_cache

        protected = set(job.all_unique_hashes)
        allocated, cached_pages, non_sequential, stashed = \
            seq.allocate_pages(self.pagetable, rc, protected)
        job.cached_pages += cached_pages
        job.total_pages += allocated
        job.non_sequential_pages += non_sequential
        # What the lookup found holds someone else's K/V; only the pages this job fills itself are its own
        job.score_hits = {id(p) for p in seq.allocated_pages if p.kv_position > 0}
        job.score_fresh = {}

        if rc is not None:
            group = job.score_group
            if group is not None and group.stash is not None and self.score_group_resume(job, seq, group):
                stashed = group.stash
            job.free_recurrent_state()
            job.recurrent_state = self.score_recurrent_state(stashed, seq.kv_position)
            job.last_recurrent_checkpoint_pos = job.recurrent_state.position

        L = len(seq.sequence_ids)
        start = seq.kv_position
        if start >= L:
            raise RuntimeError(f"score job resumed at {start}, past its {L}-token prompt")
        job.score_cached = start

        plan = []
        if rc is not None:
            split = job.score_split_at
            if split is None or not 0 < split < L:
                split = (L - 1) // PAGE_SIZE * PAGE_SIZE
            boundary = split // PAGE_SIZE * PAGE_SIZE
            if start < boundary and self.score_cache:
                plan.append((boundary, "checkpoint"))
            group = job.score_group
            if (group is not None and group.stash is None and group.pending > 1 and
                    boundary < group.split_at < L and start < group.split_at):
                plan.append((group.split_at, "group"))
        plan.append((L, "logits"))
        job.score_plan = plan
        job.score_steps = [("start", start, L, self._score_t() - t0)]


    def score_recurrent_state(self, stashed: dict | None, position: int) -> PairedState:
        """
        A score job's recurrent state: the target side only, fresh or from a stash. Score forwards never run the
        draft model, so a draft side (a recurrent drafter, e.g. DFlash2's sliding-window ring) would stay at its
        start while the target advanced, and the next stash or restore of the pair asserts on the mismatch
        """
        if not self.target_recurrent:
            return self.new_recurrent_state() if stashed is None else \
                self.recurrent_state_from_stashed(stashed, position = position)
        if stashed is None:
            return PairedState(self.cache.get_new_state(), None)
        return PairedState(self.cache.new_from_stashed(stashed["target"], position), None)


    def score_group_resume(self, job, seq, group) -> bool:
        """
        Resume a group member at the group's split point: copy the shared partial page from the anchor and take
        the group's stash. Only when every page below the split's page boundary is complete with this job's own
        content hash (found by hash, normally the anchor's pages: the shared recurrent cache need not hold a
        checkpoint there, since the stash replaces it) and the anchor still holds the same tokens on the same
        chain.
        """
        split = group.split_at
        boundary = split // PAGE_SIZE * PAGE_SIZE
        n = split - boundary
        if n <= 0 or seq.kv_position > boundary or split >= len(seq.sequence_ids):
            return False
        for li in range(boundary // PAGE_SIZE):
            p = seq.allocated_pages[li]
            if p.kv_position != PAGE_SIZE or li >= len(seq.page_hashes) or p.phash != seq.page_hashes[li]:
                return False
        anchor = group.anchor
        if anchor is None or anchor is job:
            return False
        aseq = anchor.sequences[0]
        pi = group.anchor_page
        if not aseq.allocated_pages or pi >= len(aseq.allocated_pages) or pi >= len(seq.allocated_pages):
            return False
        src = aseq.allocated_pages[pi]
        dst = seq.allocated_pages[pi]
        prev_hash = seq.allocated_pages[pi - 1].phash if pi > 0 else None
        if dst.ref_count != 1:
            return False
        tail = seq.sequence_ids.torch_slice(boundary, split)
        if dst.kv_position == 0:
            if src.kv_position < n or src.prev_hash != prev_hash or not torch.equal(src.sequence[:, :n], tail):
                return False
            self.cache.copy_page(self.cache, src.page_index, dst.page_index, n)
            self.score_own_page(job, dst)
            dst.prev_hash = prev_hash
            dst.sequence[:, :n].copy_(tail)
            dst.kv_position = n
        elif not (dst.kv_position >= n and dst.prev_hash == prev_hash and torch.equal(dst.sequence[:, :n], tail)):
            # A page this job found cached by hash (the same prompt scored before) already holds the shared
            # tokens on the same chain and needs no copy; anything else is not safe to resume into
            return False
        dst.can_revert = False
        seq.kv_position = split
        return True


    def score_advance(self, job, budget: int, results: list) -> int:
        """
        Run the job's planned forwards, at most one chunk at a time and within budget tokens (the first piece
        always runs). Returns the number of tokens processed.
        """
        seq = job.sequences[0]
        done = 0
        while job.score_plan:
            end, action = job.score_plan[0]
            a = seq.kv_position
            logits = None
            if a < end:
                b = end
                limit = (a + self.max_chunk_size) // PAGE_SIZE * PAGE_SIZE
                if limit > a:
                    b = min(b, limit)
                if done and done + (b - a) > budget:
                    break
                t0 = self._score_t()
                logits = self.score_forward(job, seq, a, b, action == "logits" and b == end)
                done += b - a
                job.score_steps.append(("forward", a, b, self._score_t() - t0))
                if b < end:
                    continue
            elif action == "logits":
                raise RuntimeError("score job has no prompt token left to evaluate")
            job.score_plan.pop(0)
            t0 = self._score_t()
            if action == "checkpoint":
                self.score_checkpoint(job, seq)
            elif action == "group":
                self.score_group_stash(job, seq)
            else:
                self.score_finish(job, seq, logits, results)
            job.score_steps.append((action, seq.kv_position, None, self._score_t() - t0))
        return done


    def score_forward(self, job, seq, a: int, b: int, want_logits: bool):
        """
        Forward prompt positions [a, b) into the job's pages and recurrent state, then update the page
        bookkeeping exactly as Job.prefill does. Returns the last position's logits if asked.
        """
        ids = seq.sequence_ids.torch_slice(a, b)
        params = {
            "attn_mode": "flash_attn",
            "block_table": seq.prefill_block_table(),
            "cache": self.cache,
            "cache_seqlens": torch.tensor([a], dtype = torch.int32),
            "recurrent_states": [job.recurrent_state.target] if job.recurrent_state is not None and
                                job.recurrent_state.target is not None else None,
            "indexed_embeddings": [],
            "inv_freq": None,
            "mm_span_prefix": 0,
        }
        if want_logits:
            params["last_tokens_only"] = 1
            logits = self.model.forward(input_ids = ids, params = params)
        else:
            self.model.prefill(input_ids = ids, params = params)
            logits = None

        seq.kv_position = b
        pages = seq.allocated_pages
        p0 = a // PAGE_SIZE
        p2 = min(b // PAGE_SIZE + 1, len(pages))
        for li in range(p0, p2):
            page = pages[li]
            page.kv_position = min(max(b - li * PAGE_SIZE, 0), PAGE_SIZE)
            page.prev_hash = None if li == 0 else pages[li - 1].phash
            pf_a = max(li * PAGE_SIZE, a)
            pf_b = min(li * PAGE_SIZE + PAGE_SIZE, b)
            if pf_b > pf_a:
                page.sequence[:, pf_a - li * PAGE_SIZE:pf_b - li * PAGE_SIZE].copy_(seq.sequence_ids.torch_slice(pf_a, pf_b))
            page.can_revert = False
            if pf_b > pf_a:
                self.score_own_page(job, page)
        for li in range(p0, p2):
            page = pages[li]
            if page.kv_position == PAGE_SIZE and li < len(seq.page_hashes) and page.phash != seq.page_hashes[li]:
                job.hash_completed_page(seq, li)
        return logits


    @staticmethod
    def score_state_ok(seq, state) -> bool:
        """A state that describes exactly this sequence's prefix: at its position, with no unsettled pass."""
        position = getattr(state, "position", None)
        return (position is None or position == seq.kv_position) and getattr(state, "last_history", 0) == 0


    def score_own_page(self, job, page):
        """Note a page this job wrote that no lookup gave it, so a private release can clear it."""
        if self.score_cache or id(page) in job.score_hits or id(page) in job.score_fresh:
            return
        job.score_fresh[id(page)] = page
        # Keeps the disk tier's write-behind and idle flush off it while it carries a content hash
        page.persistable = False


    def score_release_pages(self, job):
        """
        Drop a score job's pages. By default the pages it wrote itself go back as empty pages, first in line for
        eviction, so no later lookup can match them: a chat would otherwise reuse decision K/V written without
        its draft cache, and the host and durable tiers would fill with one-off decision state. A page another
        job holds by now is left to that job.
        """
        fresh = getattr(job, "score_fresh", None) or {}
        job.score_fresh = {}
        try:
            job.deallocate_pages()
        finally:
            pt = self.pagetable
            for page in fresh.values():
                if page.ref_count == 0 and pt.unreferenced_pages.get(page.phash) is page:
                    page.clear()


    def score_checkpoint(self, job, seq):
        """Stash the state at a page boundary in the shared recurrent cache, as prefill's last-page stash does."""
        try:
            pos = seq.kv_position
            state = job.recurrent_state
            if state is None or pos == 0 or pos % PAGE_SIZE or not self.score_state_ok(seq, state):
                return
            if self.draft_recurrent:
                # Score states carry no draft side (score_recurrent_state), and a chat resuming here needs one
                return
            page = seq.allocated_pages[pos // PAGE_SIZE - 1]
            if page.kv_position != PAGE_SIZE or not is_content_hash(page.phash):
                return
            self.recurrent_cache.put(page.phash, state)
            job.last_recurrent_checkpoint_pos = pos
        except Exception as e:
            print(f" ## Warning: score job {job.serial_number}: recurrent checkpoint failed ({e!r}); "
                  f"skipped, the score is unaffected")


    def score_group_stash(self, job, seq):
        """Leave the group's private stash at its split point; this job becomes the anchor and keeps its pages."""
        group = job.score_group
        try:
            state = job.recurrent_state
            if (group is None or group.stash is not None or group.pending <= 1 or state is None or
                    seq.kv_position != group.split_at or not self.score_state_ok(seq, state)):
                return
            group.stash = self.score_group_stash_state(state)
            group.anchor = job
            group.anchor_page = group.split_at // PAGE_SIZE
            job.score_keep_pages = True
        except Exception as e:
            print(f" ## Warning: score job {job.serial_number}: group stash failed ({e!r}); "
                  f"the other questions re-prefill from the last checkpoint")


    def score_group_stash_state(self, state: PairedState) -> dict:
        target = state.target
        if not self.score_group_device or state.draft is not None or not isinstance(target, GDNState):
            return state.stash()
        return {
            "position": state.position,
            "checkpoint_size": state.checkpoint_size,
            "target": target.stash(device = True),
        }


    def score_finish(self, job, seq, logits, results: list):
        row = logits[0, -1, :self.model.config.vocab_size]
        if job.score_token_ids is not None:
            idx = torch.tensor(job.score_token_ids, dtype = torch.long, device = row.device)
            out = row.index_select(0, idx).float().cpu()
        else:
            out = row.float().cpu()
        job.score_done = True
        now = time.time()
        self.score_retire(job)
        r = {
            "job": job,
            "stage": "score",
            "eos": True,
            "serial": job.serial_number,
            "logits": out,
            "prompt_tokens": len(seq.sequence_ids),
            "cached_tokens": job.score_cached,
            "time_enqueued": job.score_t0 - job.time_enqueue,
            "time_score": now - job.score_t0,
        }
        if job.identifier is not None:
            r["identifier"] = job.identifier
        results.append(r)
        if self.score_trace:
            steps = " ".join(f"{k}[{a}:{b}]={dt * 1000:.1f}ms" if b is not None else f"{k}@{a}={dt * 1000:.1f}ms"
                             for k, a, b, dt in job.score_steps)
            print(f" -- score #{job.serial_number}: {steps} total={(now - job.score_t0) * 1000:.1f}ms")


    def score_retire(self, job):
        """
        Release what a score job holds, once: its pages (unless it anchors a group still running, which keeps
        them until the group is done) and its state slot, and its share of the group.
        """
        if getattr(job, "score_retired", False):
            return
        job.score_retired = True
        if job.score_keep_pages:
            job.free_recurrent_state()
        else:
            self.score_release_pages(job)
        group = job.score_group
        if group is not None:
            group.pending -= 1
            if group.pending <= 0:
                self.score_group_release(group)


    def score_group_release(self, group):
        stashed, group.stash = group.stash, None
        anchor, group.anchor = group.anchor, None
        if stashed is not None:
            try:
                # A pair stash: host tensors and/or one TP handle per side
                host_pool.give(stashed)
                self.recurrent_cache._release_tp(stashed)
            except Exception as e:
                print(f" ## Warning: releasing a score group's stash failed ({e!r})")
        if anchor is not None:
            anchor.score_keep_pages = False
            self.score_release_pages(anchor)


    def reap_failed_job(self, job, error, results: list):
        """
        Contain a per-job failure so the generator stays usable for other jobs: release the
        failed job's pages and recurrent state, drop it from the active set, and append an
        error result carrying the standard serial/stage/eos fields. The async wrapper
        delivers the raw exception to that job's consumer only (AsyncJob.__aiter__ re-raises
        queued exceptions) and the sync generate() API raises it, instead of the failure
        escaping to _run_iteration, which latches AsyncGenerator.error permanently and kills
        the iteration task for every job.
        """
        try:
            job.deallocate_pages()
        except Exception:
            logger.error(
                "reap_failed_job: deallocate_pages() raised while cleaning up failed "
                "job %s; pages or recurrent state may be stranded",
                job,
                exc_info = True,
            )
        if job in self.active_jobs:
            self.active_jobs.remove(job)
        results.append({
            "job": job,
            "serial": job.serial_number,
            "stage": "error",
            "eos": True,
            "error": error,
        })


    def iterate_start_jobs(self, results: list):
        """
        Move pending jobs into the active set when batch and cache capacity allow.

        Jobs are considered in queue order, but a job can be skipped temporarily if it would exceed max_batch_size
        or needs more fresh pages than are currently unreferenced. Later jobs may start if they fit, which improves
        utilization, but each skipped job accumulates a skip count; once any skipped job reaches max_skips, startup
        stops for this iteration to preserve approximate queue fairness. Started jobs allocate their cache pages
        immediately and emit a "started" event.
        """

        # Get current max batch
        current_max_batch = 0
        for job in self.active_jobs:
            current_max_batch += len(job.sequences)

        # Start new jobs if possible
        if (self.pagetable.num_unreferenced_pages() and
            len(self.pending_jobs) and
            current_max_batch < self.max_batch_size):

            skipped_jobs = []
            for job in self.pending_jobs.copy():

                # A score job running across rounds holds a recurrent state slot outside max_batch_size,
                # so a full batch is no longer the only way to run out of slots
                if (len(job.sequences) + current_max_batch > self.max_batch_size or
                        job.current_new_pages_required() > self.pagetable.num_unreferenced_pages() or
                        (self.recurrent_cache is not None and len(self.cache.free_list) < len(job.sequences))):
                    skipped_jobs.append(job)
                    continue

                # Make sure the job we're about to add doesn't skip a job that's been skipped too many times
                for j in skipped_jobs:
                    if j.skips >= j.max_skips:
                        return
                for j in skipped_jobs:
                    j.skips += 1

                # Add job to active list
                self.pending_jobs.remove(job)
                self.active_jobs.append(job)
                job.activate()

                # Allocate pages for job
                try:
                    job.allocate_pages()
                except Exception as e:
                    self.reap_failed_job(job, e, results)
                    continue
                current_max_batch += len(job.sequences)

                r = {
                    "job": job,
                    "stage": "started",
                    "eos": False,
                    "serial": job.serial_number,
                }
                if job.identifier is not None:
                    r.update({ "identifier": job.identifier })
                results.append(r)


    def generate(
        self,
        prompt: list[tuple] | list[str] | tuple | str,
        max_new_tokens: int | None = None,
        min_new_tokens: int = 0,
        seed: int | None = None,
        sampler: Sampler | list[Sampler] | None = None,
        token_healing: bool = False,
        encode_special_tokens: bool = False,
        decode_special_tokens: bool = False,
        stop_conditions: list[int | str] | None = None,
        add_bos: bool = False,
        abort_event: threading.Event | None = None,
        completion_only: bool = False,
        filters: list[list[Filter]] | list[Filter] | None = None,
        return_last_results: bool = False,
        embeddings: list[MMEmbedding] | list[list[MMEmbedding]] | None = None,
        max_rq_tokens: int | None = None,
        stop_on_loop: tuple[int, int] = None,
        **kwargs
    ):
        """
        This is a utility function for easily generating one or more completions from one or more prompt strings. For
        more versatility and streaming functionality, use the async wrapper or create Job objects directly, enqueue()
        them and call iterate() to receive one token at a time

        :param prompt:
            If this argument is a list, its length determines the batch size, and the output will be a list of strings
            as well. Each prompt is either a string or a pair of prompts for CFG sampling. If CFG is used, sampler
            settings must contain cfg_scale.

        :param sampler:
            Sampler stack settings for all prompts in batch or list of samplers for each prompt.

        :param max_new_tokens:
            Max number of tokens to generate.

        :param min_new_tokens:
            Minimum number of tokens to generate before stop tokens become active. Until this number have been
            sampled, stop tokens are suppressed but stop strings will still end response.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param stop_conditions:
            List of strings and/or token IDs that will end generation. The stop condition is not included
            in the output.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :param filters:
            (List of) list of Filters to apply during generation. Each prompt in a batch needs
            its own filter list, or a value of None to disable filters for individual prompts.

        :param return_last_results:
            If True, returns the last results dict for each job

        :param embeddings:
            Optional list of MMEmbeddings to use for, or list of lists for batched generation

        :param max_rq_tokens:
            Maximum number of tokens before job is requeued. Rounded to nearest page boundary. This limits how
            many new pages are allocated in the cache for the job in any one round and allows a single job to use
            the full cache size without limiting concurrency for other jobs.

        :param stop_on_loop:
            Tuple of (window_size: int, min_reps: int), or None. If enabled, generation will end if the last
            window_size tokens sampled make up >= min_reps consecutive instances of a looping string.

        :return:
            Completion(s): (str or list[str] depending on the type of the input prompt argument)
            Optionally, last results: (dict or list[dict] depending on the type of the input prompt argument)
        """

        order = {}
        if isinstance(prompt, list):
            prompts = prompt
        else:
            prompts = [prompt]
            filters = [filters]
            embeddings = [embeddings]

        if not filters:
            filters = [None] * len(prompts)
        else:
            assert len(filters) == len(prompts) and \
                all((f is None or isinstance(f, list)) for f in filters), \
                "If using filters, must provide one filter list (or None-value) per prompt."

        if not embeddings:
            embeddings = [None] * len(prompts)
        else:
            assert len(embeddings) == len(prompts) and all((isinstance(f, list) or not f) for f in embeddings), \
                "Must provide one list of embeddings per prompt."

        prompts = prompt if isinstance(prompt, list) else [prompt]
        batch_size = len(prompts)
        for idx, p in enumerate(prompts):
            if isinstance(p, str):
                input_ids = self.tokenizer.encode(
                    p,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                )
            elif isinstance(p, tuple):
                input_ids = [self.tokenizer.encode(
                    p_,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                ) for p_ in p]
            else:
                assert False, "Unexpected type in prompt"

            if sampler is None or isinstance(sampler, Sampler):
                p_sampler = sampler
            elif isinstance(sampler, list):
                assert len(sampler) == len(prompts)
                p_sampler = sampler[idx]
            else:
                assert False, "Unexpected sampler type"

            job = Job(
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                min_new_tokens = min_new_tokens,
                seed = seed,
                stop_conditions = stop_conditions,
                sampler = p_sampler,
                filters = filters[idx] or [],
                token_healing = token_healing,
                decode_special_tokens = decode_special_tokens,
                embeddings = embeddings[idx] or [],
                max_rq_tokens = max_rq_tokens,
                stop_on_loop = stop_on_loop,
            )

            if seed is not None: seed += 1

            serial = self.enqueue(job)
            order[serial] = idx

        # Collect outputs until all jobs finish
        completions = [""] * batch_size
        last_results = [None] * batch_size

        while self.num_remaining_jobs():
            results = self.iterate()

            for r in results:
                idx = order[r["serial"]]
                if r["stage"] == "error":
                    # A per-job failure was contained by reap_failed_job; surface the
                    # original exception to this caller rather than returning a silently
                    # truncated completion. Jobs from the same batch that are still in
                    # flight are left to the generator's normal queue handling.
                    raise r["error"]
                if r["stage"] == "streaming":
                    text = r.get("text", "")
                    completions[idx] += text
                if r["eos"]:
                    last_results[idx] = r
            if abort_event is not None and abort_event.is_set():
                self.clear_queue()
                return None

        # Return results
        if not completion_only:
            completions = [(p if isinstance(p, str) else p[0]) + c for p, c in zip(prompts, completions)]

        if not isinstance(prompt, list):
            completions = completions[0]
            last_results = last_results[0]

        if return_last_results:
            return completions, last_results
        else:
            return completions


SPEC_PROF = os.environ.get("EXL3_SPEC_PROF") == "1"
SPEC_PROF_REC: list = []
_PROF_CUR: dict = {}


def _pt():
    if not SPEC_PROF:
        return 0.0
    cuda_sync_active()
    return time.perf_counter()


class SpecGate:
    """
    Cost-model gate for speculative decoding. Keeps EMAs of the plain step time, the
    speculative round time and the tokens a round yields, and speculates only while
    tokens_per_round / t_round >= 1 / t_plain. In plain mode it probes one speculative round after
    `probe` plain steps; a failed probe doubles the interval (up to probe_max), a successful one
    switches back to speculation. The first steps measure both arms once.
    """

    def __init__(self, alpha: float = None, probe_min: int = None, probe_max: int = None, margin: float = None):
        env = os.environ.get
        self.alpha = alpha or float(env("EXL3_SPEC_GATE_ALPHA", "0.3"))
        self.probe_min = probe_min or int(env("EXL3_SPEC_GATE_PROBE", "4"))
        self.probe_max = probe_max or int(env("EXL3_SPEC_GATE_PROBE_MAX", "32"))
        self.margin = margin or float(env("EXL3_SPEC_GATE_MARGIN", "1.0"))
        self.t_plain = None
        self.t_spec = None
        self.tok_round = None
        self.mode_spec = True
        self.probe = self.probe_min
        self.since_probe = 0
        self.n_spec = 0
        self.n_plain = 0

    def _ema(self, old, new):
        return new if old is None else old + self.alpha * (new - old)

    def spec_pays(self) -> bool:
        if self.t_plain is None or self.t_spec is None or self.tok_round is None:
            return True
        return self.tok_round / self.t_spec >= self.margin / self.t_plain

    def want_spec(self) -> bool:
        if self.t_spec is None:
            return True
        if self.t_plain is None:
            return False
        if self.mode_spec:
            return True
        return self.since_probe >= self.probe

    def record_spec(self, dt: float, produced: int):
        self.n_spec += 1
        if produced <= 0:
            return
        self.t_spec = self._ema(self.t_spec, dt)
        was_probe = not self.mode_spec
        self.tok_round = self._ema(self.tok_round, float(produced)) if not was_probe else \
            0.5 * (self.tok_round or produced) + 0.5 * produced
        pays = self.spec_pays()
        if was_probe:
            self.since_probe = 0
            if pays:
                self.mode_spec = True
                self.probe = self.probe_min
            else:
                self.probe = min(self.probe * 2, self.probe_max)
        else:
            self.mode_spec = pays

    def record_plain(self, dt: float, produced: int):
        self.n_plain += 1
        if produced != 1:
            return
        self.t_plain = self._ema(self.t_plain, dt)
        self.since_probe += 1
