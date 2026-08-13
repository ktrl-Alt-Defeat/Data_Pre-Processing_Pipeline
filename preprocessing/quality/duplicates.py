"""Exact and near-duplicate detection.

Exact duplicates are found by content hash (file bytes) and pixel hash (decoded
RGB), so the same photograph re-encoded as PNG is still caught. Near duplicates
are found by perceptual hash within a configured Hamming radius.

Nothing is deleted here. The detector groups images, nominates a representative
per group and records the relationship; the quality gate decides what that means.

Scaling: comparing every pair is quadratic and impossible at corpus scale, so
candidate pairs come from banded indexing — two hashes within distance *d* must
agree exactly on at least one of *d+1* disjoint bit bands (pigeonhole). Only
those candidates are compared bit for bit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..core.config import DuplicateConfig
from ..core.records import DuplicateStatus


@dataclass(frozen=True, slots=True)
class DuplicateEntry:
    """The identity of one image, as far as duplicate detection is concerned."""

    image_id: str
    label: str
    content_hash: str | None = None
    pixel_hash: str | None = None
    perceptual_hash: int | None = None
    quality_score: float | None = None
    megapixels: float | None = None


@dataclass(frozen=True, slots=True)
class DuplicateLink:
    """A non-representative image and its relationship to the one that is kept."""

    image_id: str
    duplicate_of: str
    group_id: str
    status: DuplicateStatus
    distance: int | None
    similarity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "duplicate_of": self.duplicate_of,
            "group_id": self.group_id,
            "status": self.status.value,
            "hamming_distance": self.distance,
            "similarity": self.similarity,
        }


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """A cluster of images that depict the same thing."""

    group_id: str
    representative: str
    members: tuple[str, ...]
    exact_members: int
    near_members: int

    @property
    def size(self) -> int:
        return len(self.members)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "representative": self.representative,
            "size": self.size,
            "exact_members": self.exact_members,
            "near_members": self.near_members,
            "members": list(self.members),
        }


@dataclass(slots=True)
class DuplicateResult:
    """Everything the detector found, indexed for the quality gate."""

    groups: list[DuplicateGroup] = field(default_factory=list)
    links: dict[str, DuplicateLink] = field(default_factory=dict)
    representatives: set[str] = field(default_factory=set)
    comparisons: int = 0

    @property
    def duplicate_count(self) -> int:
        return len(self.links)

    def statistics(self) -> dict[str, Any]:
        exact = sum(1 for link in self.links.values() if link.status is DuplicateStatus.EXACT_DUPLICATE)
        return {
            "groups": len(self.groups),
            "duplicates": len(self.links),
            "exact_duplicates": exact,
            "near_duplicates": len(self.links) - exact,
            "largest_group": max((group.size for group in self.groups), default=0),
            "candidate_comparisons": self.comparisons,
        }


class DuplicateDetector:
    """Groups images by exact and perceptual identity."""

    def __init__(self, config: DuplicateConfig) -> None:
        self._config = config
        self._bits = config.hash_size * config.hash_size

    def detect(self, entries: Sequence[DuplicateEntry]) -> DuplicateResult:
        """Cluster ``entries`` and nominate one representative per cluster."""
        if not self._config.enabled or len(entries) < 2:
            return DuplicateResult()

        union = _UnionFind(len(entries))
        if self._config.exact:
            self._link_exact(entries, union)
        comparisons = self._link_near(entries, union) if self._config.near else 0

        return self._build_result(entries, union, comparisons)

    # --- linking --------------------------------------------------------------- #

    def _link_exact(self, entries: Sequence[DuplicateEntry], union: "_UnionFind") -> None:
        for attribute in self._exact_attributes():
            buckets: dict[str, list[int]] = defaultdict(list)
            for index, entry in enumerate(entries):
                value = getattr(entry, attribute)
                if value:
                    buckets[value].append(index)
            for members in buckets.values():
                self._union_all(entries, union, members)

    def _exact_attributes(self) -> tuple[str, ...]:
        if self._config.exact_method == "content":
            return ("content_hash",)
        if self._config.exact_method == "pixel":
            return ("pixel_hash",)
        return ("content_hash", "pixel_hash")

    def _link_near(self, entries: Sequence[DuplicateEntry], union: "_UnionFind") -> int:
        radius = self._config.max_hamming_distance
        indexed = [(index, entry.perceptual_hash) for index, entry in enumerate(entries) if entry.perceptual_hash is not None]
        if len(indexed) < 2:
            return 0
        if radius == 0:
            buckets: dict[int, list[int]] = defaultdict(list)
            for index, value in indexed:
                buckets[value].append(index)
            for members in buckets.values():
                self._union_all(entries, union, members)
            return 0

        comparisons = 0
        for bucket in self._candidate_buckets(indexed, radius):
            for position, index in enumerate(bucket):
                for other in bucket[position + 1 :]:
                    if union.connected(index, other):
                        continue
                    comparisons += 1
                    left, right = entries[index].perceptual_hash, entries[other].perceptual_hash
                    if hamming_distance(left, right) <= radius and self._comparable(entries[index], entries[other]):
                        union.union(index, other)
        return comparisons

    def _candidate_buckets(self, indexed: Sequence[tuple[int, int]], radius: int) -> Iterable[list[int]]:
        """Band the hash into radius+1 slices and bucket by each slice's value."""
        bands = radius + 1
        width = max(1, self._bits // bands)
        for band in range(bands):
            shift = band * width
            mask = (1 << width) - 1 if band < bands - 1 else (1 << (self._bits - shift)) - 1
            buckets: dict[int, list[int]] = defaultdict(list)
            for index, value in indexed:
                buckets[(value >> shift) & mask].append(index)
            for members in buckets.values():
                if len(members) > 1:
                    yield members

    def _union_all(self, entries: Sequence[DuplicateEntry], union: "_UnionFind", members: Sequence[int]) -> None:
        for other in members[1:]:
            if self._comparable(entries[members[0]], entries[other]):
                union.union(members[0], other)

    def _comparable(self, left: DuplicateEntry, right: DuplicateEntry) -> bool:
        return self._config.across_classes or left.label == right.label

    # --- grouping -------------------------------------------------------------- #

    def _build_result(
        self,
        entries: Sequence[DuplicateEntry],
        union: "_UnionFind",
        comparisons: int,
    ) -> DuplicateResult:
        clusters: dict[int, list[int]] = defaultdict(list)
        for index in range(len(entries)):
            clusters[union.find(index)].append(index)

        result = DuplicateResult(comparisons=comparisons)
        for root in sorted(clusters):
            members = clusters[root]
            if len(members) < 2:
                continue
            representative = self._representative(entries, members)
            group_id = entries[representative].image_id
            links = [
                self._link(entries, representative, index, group_id)
                for index in members
                if index != representative
            ]
            exact = sum(1 for link in links if link.status is DuplicateStatus.EXACT_DUPLICATE)
            result.groups.append(
                DuplicateGroup(
                    group_id=group_id,
                    representative=group_id,
                    members=tuple(entries[index].image_id for index in members),
                    exact_members=exact,
                    near_members=len(links) - exact,
                )
            )
            result.representatives.add(group_id)
            result.links.update({link.image_id: link for link in links})
        return result

    def _representative(self, entries: Sequence[DuplicateEntry], members: Sequence[int]) -> int:
        """Ties always fall back to discovery order, so the choice is deterministic."""
        if self._config.keep == "highest_quality":
            return min(members, key=lambda index: (-(entries[index].quality_score or 0.0), index))
        if self._config.keep == "largest":
            return min(members, key=lambda index: (-(entries[index].megapixels or 0.0), index))
        return min(members)

    def _link(
        self,
        entries: Sequence[DuplicateEntry],
        representative: int,
        index: int,
        group_id: str,
    ) -> DuplicateLink:
        original, duplicate = entries[representative], entries[index]
        if _shares_exact_hash(original, duplicate, self._exact_attributes()):
            return DuplicateLink(duplicate.image_id, original.image_id, group_id,
                                 DuplicateStatus.EXACT_DUPLICATE, 0, 1.0)

        distance = hamming_distance(original.perceptual_hash, duplicate.perceptual_hash)
        similarity = 1.0 - (distance / self._bits) if distance is not None else 0.0
        return DuplicateLink(duplicate.image_id, original.image_id, group_id,
                             DuplicateStatus.NEAR_DUPLICATE, distance, round(similarity, 6))


# --------------------------------------------------------------------------- #
# Perceptual hashing
# --------------------------------------------------------------------------- #


def perceptual_hash(gray: np.ndarray, hash_size: int = 8, kind: str = "phash") -> int:
    """Perceptual hash of a grayscale array as an integer of ``hash_size**2`` bits."""
    if kind == "ahash":
        return average_hash(gray, hash_size)
    if kind == "dhash":
        return difference_hash(gray, hash_size)
    return dct_hash(gray, hash_size)


def average_hash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = _resize(gray, hash_size, hash_size).astype(np.float64)
    return _pack(resized > resized.mean())


def difference_hash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = _resize(gray, hash_size + 1, hash_size).astype(np.float64)
    return _pack(resized[:, 1:] > resized[:, :-1])


def dct_hash(gray: np.ndarray, hash_size: int = 8) -> int:
    """Classic pHash: low-frequency DCT coefficients thresholded at their median."""
    size = hash_size * 4
    resized = _resize(gray, size, size).astype(np.float32)
    coefficients = cv2.dct(resized)[:hash_size, :hash_size]
    # The DC term encodes overall brightness, not structure; excluding it from the
    # median keeps the hash stable under exposure changes.
    median = float(np.median(coefficients.flatten()[1:]))
    return _pack(coefficients > median)


def hamming_distance(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return (left ^ right).bit_count()


def to_hex(value: int | None, hash_size: int = 8) -> str | None:
    if value is None:
        return None
    return f"{value:0{max(1, (hash_size * hash_size) // 4)}x}"


def _resize(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def _pack(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _shares_exact_hash(left: DuplicateEntry, right: DuplicateEntry, attributes: Sequence[str]) -> bool:
    return any(
        getattr(left, attribute) is not None and getattr(left, attribute) == getattr(right, attribute)
        for attribute in attributes
    )


class _UnionFind:
    """Disjoint-set forest with path compression; turns pairs into clusters."""

    __slots__ = ("_parent",)

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        parent = self._parent
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Always attach the later root to the earlier one so clustering is
            # independent of the order pairs happen to be discovered in.
            low, high = sorted((left_root, right_root))
            self._parent[high] = low

    def connected(self, left: int, right: int) -> bool:
        return self.find(left) == self.find(right)


__all__ = [
    "DuplicateDetector",
    "DuplicateEntry",
    "DuplicateGroup",
    "DuplicateLink",
    "DuplicateResult",
    "average_hash",
    "dct_hash",
    "difference_hash",
    "hamming_distance",
    "perceptual_hash",
    "to_hex",
]
