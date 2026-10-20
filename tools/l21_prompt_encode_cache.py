# Incremental prompt encoding for chat servers (model-agnostic, Python only, portable HIP / CUDA / Metal).
# Clients (Hermes, OpenAI-style) re-send the whole conversation every turn; a full encode costs ~1.3 ms of CPU per 1K tokens
# (90 ms at 66K, 166 ms at 128K, Qwen3.8 tokenizer). PromptEncodeCache keeps the last n (prompt, ids) pairs; a prompt that shares a text
# prefix with one of them encodes only from the last turn-start special token (cut: one string or a list, e.g. '<|im_start|>' for
# ChatML, ['<|system|>', '<|user|>', '<|assistant|>', '<|observation|>'] for GLM) inside the shared part. The tokenizer splits its input
# at special tokens and encodes the pieces independently, so the ids equal the full encode's. A self-test at construction (split ==
# full on a rendered 2-turn conversation, or on a synthetic text without render) disables the cache where that fails; check=True also
# compares every request with the full encode (debug).
#
#   pec = PromptEncodeCache(tokenizer, n=8, cut="<|im_start|>", render=lambda msgs: render_prompt(template, msgs, None))
#   ids, hit = pec.encode(prompt)          # ids: 1-D LongTensor (CPU), same as tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
#
# Not for prompts with image embeddings (encode those in full). Thread-safe (one lock).
import re, threading
import torch


class PromptEncodeCache:

    def __init__(self, tokenizer, n: int = 8, cut="<|im_start|>", render=None, check: bool = False, log=print):
        cuts = [cut] if isinstance(cut, str) else list(cut)
        self.tokenizer, self.n, self.check, self.log = tokenizer, n, check, log
        self.e = []   # [text, ids, marker char starts, marker token positions], most recent first
        self.lock = threading.Lock()
        mids = [self.full(c) for c in cuts]
        if not cuts or any(m.numel() != 1 for m in mids):
            log(f"prompt encode cache off: {cuts!r} not one token each"); self.n = 0; return
        self.mids = torch.tensor([int(m[0]) for m in mids])
        self.cut_re = re.compile("|".join(re.escape(c) for c in sorted(cuts, key=len, reverse=True)))
        msgs = [{"role": "system", "content": "S x"}, {"role": "user", "content": "Hi there, how are you?"},
                {"role": "assistant", "content": "Fine. And you?"}, {"role": "user", "content": "Tell me about pages."}]
        try:
            if render is not None: p1, p2 = render(msgs[:2]), render(msgs)
            else:
                p1 = "Sys x " + cuts[-1] + " Hi there, how are you?\n" + cuts[0] + " ok"
                p2 = p1[:-3] + " Fine. And you?\n" + cuts[-1] + " Tell me about pages.\n" + cuts[0]
            self.encode(p1); ids, hit = self.encode(p2)
            ok = hit and torch.equal(ids, self.full(p2))
        except Exception as ex:
            ok = False; log(f"prompt encode cache self-test error: {ex}")
        if not ok:
            log("prompt encode cache off: self-test failed"); self.n = 0
        self.e = []

    def full(self, p: str) -> torch.Tensor:
        ids = self.tokenizer.encode(p, add_bos=False, encode_special_tokens=True)
        ids = ids[0] if isinstance(ids, tuple) else ids
        return ids.reshape(-1).cpu()

    def encode(self, p: str):
        if not self.n:
            return self.full(p), False
        with self.lock:
            best = None
            for i, (t, ids, cs, ts) in enumerate(self.e):
                lo, hi = -1, len(cs) - 1   # last marker whose start lies inside the shared text: binary search over the markers
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if cs[mid] <= len(p) and p[:cs[mid]] == t[:cs[mid]]: lo = mid
                    else: hi = mid - 1
                if lo >= 0 and cs[lo] > 0 and (best is None or cs[lo] > best[1]):
                    best = (i, cs[lo], ts[lo], lo == len(cs) - 1)   # last item: p extends this entry (same conversation)
            if best is not None:
                i, c, tpos, _ = best
                ids = torch.cat([self.e[i][1][:tpos], self.full(p[c:])])
                if self.check and not torch.equal(ids, self.full(p)):
                    self.log("prompt encode cache MISMATCH, full encode used"); ids, best = self.full(p), None
            else:
                ids = self.full(p)
            self._add(p, ids, best[0] if best is not None and best[3] else None)
            return ids, best is not None

    def _add(self, p, ids, drop=None):
        cs = [m.start() for m in self.cut_re.finditer(p)]
        ts = torch.isin(ids, self.mids).nonzero().reshape(-1).tolist()
        if len(cs) == len(ts):
            if drop is not None: self.e.pop(drop)
            self.e.insert(0, [p, ids, cs, ts]); del self.e[self.n:]
