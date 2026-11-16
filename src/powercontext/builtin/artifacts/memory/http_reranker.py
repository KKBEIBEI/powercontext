# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batched HTTP scoring with stable selection over the full Memory candidate pool."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

if TYPE_CHECKING:
    import httpx


class HTTPMemoryReranker:
    """Select original candidate ranks using scores from a caller-owned HTTP client.

    Each request sends ``query`` and ``candidates`` (global ``rank`` and ``text``).
    Responses must contain only ``scores``, covering that batch's ranks exactly
    once with finite numeric scores. All batches participate in one final sort;
    equal scores preserve the original candidate order. The client stays open
    for its caller to reuse and close.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
    ) -> MemoryRerankDecision:
        """Score every candidate and return the best ranks plus HTTP request usage.

        Invalid bounds or response data raise ``ValueError``. HTTP status and
        transport errors propagate from httpx without a fallback selection.
        """
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("rerank limit must be an integer between one and the candidate count")  # noqa: TRY003

        scores: dict[int, int | float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = [
                {"rank": rank, "text": candidate.text}
                for rank, candidate in enumerate(candidates[start : start + self._batch_size], start=start + 1)
            ]
            response = await self._client.post(
                self._endpoint,
                json={"query": query, "candidates": batch},
                follow_redirects=False,
            )
            requests += 1
            response.raise_for_status()
            scores.update(_batch_scores(response.json(), range(start + 1, start + len(batch) + 1)))

        selected = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(selected[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _batch_scores(payload: object, expected_ranks: range) -> dict[int, int | float]:
    """Validate the HTTP response without coercing wire types or losing integer precision."""
    if not isinstance(payload, dict) or payload.keys() != {"scores"} or not isinstance(payload["scores"], list):
        raise ValueError("rerank response must contain only a scores array")  # noqa: TRY003
    scores: dict[int, int | float] = {}
    for item in payload["scores"]:
        if not isinstance(item, dict) or item.keys() != {"rank", "score"}:
            raise ValueError("each rerank score must contain only rank and score")  # noqa: TRY003
        rank, score = item["rank"], item["score"]
        if type(rank) is not int or rank not in expected_ranks or rank in scores:
            raise ValueError("rerank response ranks must belong to this batch and appear once")  # noqa: TRY003
        if type(score) not in (int, float) or (isinstance(score, float) and not math.isfinite(score)):
            raise ValueError("rerank scores must be finite numbers")  # noqa: TRY003
        scores[rank] = score
    if len(scores) != len(expected_ranks):
        raise ValueError("rerank response must cover every rank in this batch")  # noqa: TRY003
    return scores


__all__ = ["HTTPMemoryReranker"]
