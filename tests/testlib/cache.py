"""
Stand-ins for driving the CPU page tier (exllamav3/generator/cpu_cache.py) without a model or generator.

The tier only needs, per cache, a `layers` dict whose values expose get_tensors() (page-major tensors, None
entries skipped) and, per stored page, the page table's (phash, prev_hash, page_index, sequence) record.

    FakeCacheLayer([tensor, None])      one cache layer over the given tensors
    page(tag, page_index, prev = None)  a page record whose hash is phash(tag) and whose tokens are all `tag`
    phash(tag)                          the hash page(tag, ...) carries
    build_disk_tiers(slots, device, root)   a CPU tier over one NUM_PAGES-page tensor with a disk tier behind it
"""

from collections import namedtuple
from types import SimpleNamespace

import torch

PAGE_SIZE = 32
NUM_PAGES = 8


class FakeCacheLayer:

    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensors(self):
        return self.tensors


Page = namedtuple("Page", ["phash", "prev_hash", "page_index", "sequence", "persistable"], defaults = (True,))


def phash(tag: int) -> bytes:
    return tag.to_bytes(2, "big")


def page(tag: int, page_index: int, prev: bytes | None = None, page_size: int = PAGE_SIZE) -> Page:
    return Page(phash(tag), prev, page_index, torch.full((page_size,), tag, dtype = torch.long))


def build_disk_tiers(slots, device, root, disk_size = 1 << 30, fingerprint = "test"):
    from exllamav3.generator.cpu_cache import CPUPageCache
    from exllamav3.generator.disk_cache import DiskPageCache
    tensor = torch.zeros((NUM_PAGES, 64), dtype = torch.float16, device = device)
    cache_obj = SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})
    tier = CPUPageCache([cache_obj], slots * 4096)
    # Checkpoint spacing off unless a test sets it: the small positions used here would otherwise be refused
    disk = DiskPageCache(str(root), [cache_obj], tier, disk_size, fingerprint = fingerprint, min_state_distance = 0)
    tier.attach_disk(disk)
    return tier, disk, tensor
