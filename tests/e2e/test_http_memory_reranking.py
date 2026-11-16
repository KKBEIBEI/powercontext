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

"""HTTP scoring through native Memory search and exact version expansion."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory import HTTPMemoryReranker, MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, RuntimeConfig, open_builtin_contexts


def test_http_reranking_preserves_memory_identity_and_reports_actual_requests() -> None:
    seen: list[dict[str, Any]] = []
    scores = {1: 0.10, 2: 0.90, 3: 0.95, 4: 0.90}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "scores": [
                    {"rank": item["rank"], "score": scores[item["rank"]]} for item in reversed(body["candidates"])
                ]
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(
                "https://reranker.invalid/score", client=client, batch_size=2, policy_id="test.http-rerank.v1"
            )
            config = BuiltinConfig(database=SQLiteConfig(), runtime=RuntimeConfig(memory_rerank_candidate_limit=4))
            async with open_builtin_contexts(config, memory_reranker=reranker) as contexts:
                service = (await contexts.get("http-rerank")).artifacts.memory
                initial = await service.remember(
                    memory=None,
                    entries=tuple(MemoryEntryInput(kind="fact", text=f"Project fact {i}.") for i in range(3)),
                    mode="append",
                )
                assert initial is not None
                memory = await service.remember(
                    memory=initial,
                    entries=(MemoryEntryInput(kind="fact", text="Project additional fact."),),
                    mode="append",
                )
                assert memory is not None
                assert memory.revision == initial.revision + 1
                result = await service.search("project", memories=(memory,), limit=2, mode="fts")
                trace = result.rerank
                assert trace is not None
                assert trace.policy_id == "test.http-rerank.v1"
                assert trace.selected_ranks == (3, 2)
                assert trace.usage.requests == len(seen) == 2
                assert len(trace.candidate_hits) == 4
                assert result.hits == (trace.candidate_hits[2], trace.candidate_hits[1])
                for body in seen:
                    assert body["query"] == "project"
                    for item in body["candidates"]:
                        assert item["text"] == trace.candidate_hits[item["rank"] - 1].text
                expanded = await service.expand(result.hits)
                for hit, version in zip(result.hits, expanded, strict=True):
                    assert hit.memory_ref == memory.as_ref()
                    assert version.entry_id == hit.entry_id
                    assert version.entry_version_id == hit.entry_version_id
                    assert version.text == hit.text
                empty = await service.search("unmatchedword", memories=(memory,), limit=2, mode="fts")
                assert empty.hits == ()
                assert empty.rerank is None
                assert len(seen) == 2
                assert not client.is_closed

    asyncio.run(scenario())
