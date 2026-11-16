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

"""Public behavior of the batched HTTP Memory reranker."""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import HTTPMemoryReranker, MemoryHit


def _hits(count: int) -> tuple[MemoryHit, ...]:
    return tuple(
        MemoryHit(
            memory_ref=ArtifactRef(family="memory", artifact_id="memory", revision=2),
            entry_id=f"entry-{rank}",
            entry_version_id=f"version-{rank}",
            text=f"候选 {rank}",
            score=1 / rank,
            matched_by=("fts",),
        )
        for rank in range(1, count + 1)
    )


@pytest.mark.parametrize("count,batch_size,limit", [(1, 32, 1), (4, 2, 2), (5, 2, 5), (173, 7, 10)])
def test_global_selection_covers_all_candidates_and_counts_each_call(count: int, batch_size: int, limit: int) -> None:
    candidates = _hits(count)
    snapshots = tuple(hit.model_dump() for hit in candidates)
    # Equal pairs cross batch boundaries; the tail outranks the first batch.
    scores = {rank: rank // 2 for rank in range(1, count + 1)}
    seen: list[int] = []
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.method == "POST"
        assert str(request.url) == "https://reranker.invalid/score"
        body = json.loads(request.content)
        assert set(body) == {"query", "candidates"}
        assert body["query"] == "数据库连接超时怎么排查"
        batch = body["candidates"]
        assert 1 <= len(batch) <= batch_size
        for item in batch:
            assert set(item) == {"rank", "text"}
            assert type(item["rank"]) is int
            assert item["text"] == candidates[item["rank"] - 1].text
            seen.append(item["rank"])
        return httpx.Response(
            200, json={"scores": [{"rank": item["rank"], "score": scores[item["rank"]]} for item in reversed(batch)]}
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid/score", client=client, batch_size=batch_size)
            assert reranker.policy_id == "http-cross-encoder-v1"
            for _ in range(2):
                seen.clear()
                before = request_count
                decision = await reranker.rerank("数据库连接超时怎么排查", candidates, limit)
                assert decision.selected_ranks == tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
                assert decision.usage.requests == request_count - before == math.ceil(count / batch_size)
                assert sorted(seen) == list(range(1, count + 1))
                assert not decision.used_fallback
                assert decision.discarded_rank_count == 0
                assert not client.is_closed
            assert tuple(hit.model_dump() for hit in candidates) == snapshots

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([0.10, 0.90, 0.95, 0.90], (3, 2, 4, 1)),
        ([-2, 0, -0.0, -1.5], (2, 3, 4, 1)),
        ([2**53, 2**53 + 1, 0, -1], (2, 1, 3, 4)),
    ],
)
def test_numeric_scores_keep_exact_order(scores, expected) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        batch = json.loads(request.content)["candidates"]
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": scores[item["rank"] - 1]} for item in reversed(batch)]},
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=2).rerank(
                "query", _hits(4), 4
            )
            assert result.selected_ranks == expected

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"scores": None},
        {"scores": {}},
        {"scores": [], "extra": True},
        {"scores": []},
        {"scores": [None]},
        {"scores": [{"rank": 1}]},
        {"scores": [{"score": 0.5}]},
        {"scores": [{"rank": 1, "score": 0.5, "extra": True}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.4}]},
        *({"scores": [{"rank": rank, "score": 0.5}]} for rank in (0, -1, 2, True, 1.0, "1", None, [])),
        *(
            {"scores": [{"rank": 1, "score": score}]}
            for score in (True, "0.5", None, [], {}, float("nan"), float("inf"), -float("inf"))
        ),
    ],
)
def test_invalid_response_schema_raises_value_error(payload) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", _hits(1), 1)

    asyncio.run(scenario())


def test_second_batch_cannot_return_a_previous_batch_rank() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"scores": [{"rank": 1, "score": 1}]}))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=1).rerank(
                    "q", _hits(2), 1
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2", None])
def test_invalid_batch_size_raises_value_error(batch_size) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError):
                HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=batch_size)

    asyncio.run(scenario())


@pytest.mark.parametrize("count,limit", [(0, 1), (1, 0), (1, -1), (1, 2), (2, True), (2, 1.5), (2, "1"), (2, None)])
def test_invalid_bounds_fail_before_http(count, limit) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("invalid parameters must not send a request")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", _hits(count), limit)

    asyncio.run(scenario())


@pytest.mark.parametrize("content", [b"not json", b"", b'{"scores":', b"\xff"])
def test_invalid_json_raises_value_error(content: bytes) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
        ) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", _hits(1), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [302, 400, 429, 500])
def test_non_success_status_preserves_httpx_error(status: int) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(status, content=b"not json", headers={"Location": "/next"})
        )
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", _hits(1), 1)
            assert caught.value.response.status_code == status

    asyncio.run(scenario())


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_network_errors_propagate_unchanged(error_type) -> None:
    error = error_type("offline")

    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(error_type) as caught:
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", _hits(1), 1)
            assert caught.value is error

    asyncio.run(scenario())
