"""Pure-Python suffix automaton, fallback for BC_SAM (exllamav3_ext/sam.cpp).

Drives n-gram speculative drafting (generator/job.py::get_ngram_draft). The C++ class is a plain
CPU data structure -- online suffix automaton with min_end per state -- so this is a literal
transcription with dict transitions in place of the intrusive edge lists (a SAM never holds two
edges on the same token from one state, so the two are equivalent).
"""
from __future__ import annotations

import torch


class SAM:
    __slots__ = ("_link", "_max_len", "_min_end", "_trans", "_last", "_match_state",
                 "_match_len", "_pos")

    def __init__(self):
        self.reset()

    def reset(self, reserve_tokens: int = 0) -> None:
        assert reserve_tokens >= 0, "reserve_tokens must be >= 0"
        self._link = [-1]
        self._max_len = [0]
        self._min_end = [0x7fffffff]
        self._trans = [{}]
        self._last = 0
        self._match_state = 0
        self._match_len = 0
        self._pos = 0

    def length(self) -> int:
        return self._pos

    def _new_state(self, max_len: int, link: int, min_end: int) -> int:
        idx = len(self._link)
        self._link.append(link)
        self._max_len.append(max_len)
        self._min_end.append(min_end)
        self._trans.append({})
        return idx

    def _advance_match(self, token: int):
        state = self._match_state
        length = self._match_len
        trans = self._trans
        link = self._link
        max_len = self._max_len

        nxt = trans[state].get(token)
        while state != 0 and nxt is None:
            state = link[state]
            if max_len[state] < length:
                length = max_len[state]
            nxt = trans[state].get(token)

        if nxt is not None:
            state = nxt
            length += 1
        else:
            state = 0
            length = 0

        self._match_state = state
        self._match_len = length
        return state, length

    def _extend(self, token: int) -> None:
        trans = self._trans
        link = self._link
        max_len = self._max_len

        cur = self._new_state(max_len[self._last] + 1, 0, self._pos)
        p = self._last

        while p != -1 and token not in trans[p]:
            trans[p][token] = cur
            p = link[p]

        if p == -1:
            link[cur] = 0
        else:
            q = trans[p][token]
            if max_len[p] + 1 == max_len[q]:
                link[cur] = q
            else:
                clone = self._new_state(max_len[p] + 1, link[q], self._min_end[q])
                trans[clone].update(trans[q])
                while p != -1 and trans[p].get(token) == q:
                    trans[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone

        self._last = cur
        self._pos += 1

    def _span(self, state: int, match_len: int):
        if match_len <= 0:
            return -1, -1
        source_end = self._min_end[state]
        return source_end - match_len + 1, source_end + 1

    def accept(self, token: int):
        state, match_len = self._advance_match(int(token))
        span = self._span(state, match_len)
        self._extend(int(token))
        return span

    def accept_tensor(self, tokens: torch.Tensor):
        assert tokens.dtype == torch.long, "tokens must be int64"
        assert tokens.is_contiguous(), "tokens must be contiguous"
        if tokens.dim() == 2:
            assert tokens.shape[0] == 1, "2D tokens must have bsz 1"

        total = tokens.shape[-1]
        # The sequence shrinks when the job rewinds (banned-string suppression); a suffix
        # automaton cannot un-accept tokens, so rebuild from the truncated sequence.
        if total < self._pos:
            self.reset(total)

        offset = self._pos
        if total - offset < 1:
            return -1, -1

        ids = tokens.reshape(-1)[offset:].tolist()
        for t in ids[:-1]:
            self._advance_match(t)
            self._extend(t)
        state, match_len = self._advance_match(ids[-1])
        self._extend(ids[-1])
        return self._span(state, match_len)


def BC_SAM() -> SAM:
    return SAM()
