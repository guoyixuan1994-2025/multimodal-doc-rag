from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import new_request_id
from app.generation.llm_client import create_chat_llm
from app.generation.prompt import build_rag_prompt
from app.generation.refusal import should_refuse
from app.indexing.bm25_store import tokenize
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_rewrite import safe_rewrite_query
from app.retrieval.reranker import Reranker, RerankHit
from app.schemas.chat import ChatResponse, SourceChunk
from app.schemas.document import ChunkRecord


class RagService:
    def __init__(self, chunks: list[ChunkRecord], collection_name: str | None = None) -> None:
        self.settings = get_settings()
        self.chunks = chunks
        self.collection_name = collection_name or self.settings.default_collection_name
        self.active_documents = sorted(
            {
                str(chunk.metadata.get("file_name"))
                for chunk in chunks
                if chunk.metadata.get("file_name")
            }
        )
        self.retriever = HybridRetriever(chunks, collection_name=self.collection_name)
        self.reranker = Reranker()

    def answer(self, question: str) -> ChatResponse:
        request_id = new_request_id()
        rewritten_query = safe_rewrite_query(question, self.active_documents)
        rewritten_query = self._expand_query_from_corpus(question, rewritten_query)
        scoped_chunks, requested_documents = self._scope_chunks(f"{question} {rewritten_query}")
        direct_image_caption = self._is_image_overview_question(question, scoped_chunks)
        document_overview = bool(requested_documents) and self._is_document_overview_question(question)
        if document_overview:
            rerank_hits = self._document_overview_hits(scoped_chunks)
        elif direct_image_caption:
            # 用户明确问当前图片整体内容时，VLM caption 本身就是一手证据。
            # 通用文本 reranker 不擅长判定这类 caption 是否相关，因此不让低分误触发拒答。
            candidates = [
                (chunk, 1.0)
                for chunk in scoped_chunks
                if chunk.chunk_type == "image_caption" or "<summary>natural_image</summary>" in chunk.content
            ]
        elif requested_documents:
            # 指定文件的问题必须限制在该文件内，避免同类图片或论文内容串入答案。
            candidates = [(chunk, 1.0) for chunk in scoped_chunks]
        else:
            hybrid_hits = self.retriever.retrieve(rewritten_query)
            candidates = [(hit.chunk, hit.score) for hit in hybrid_hits]
        if not document_overview:
            rerank_hits = self.reranker.rerank(
                rewritten_query,
                candidates,
                top_n=self._adaptive_top_n(question, rewritten_query),
            )
            rerank_hits = self._ensure_exact_evidence_coverage(question, rewritten_query, rerank_hits, scoped_chunks)

        if direct_image_caption and self._has_uninformative_image_caption(rerank_hits):
            return ChatResponse(
                request_id=request_id,
                question=question,
                rewritten_query=rewritten_query,
                answer=(
                    "MinerU 已识别到该文件中的图片区域，但当前解析结果没有识别到可读文字标签或图片说明。"
                    "因此系统只能确认它是一张图片/图形区域，不能基于现有证据进一步解释具体业务含义。"
                ),
                grounded=True,
                sources=[self._to_source(hit) for hit in rerank_hits],
            )

        if (
            not direct_image_caption
            and not document_overview
            and (
                (
                    should_refuse(rerank_hits)
                    and not self._has_exact_short_fact_support(rewritten_query, rerank_hits)
                )
                or self._has_missing_named_term_support(question, rerank_hits)
            )
        ):
            return ChatResponse(
                request_id=request_id,
                question=question,
                rewritten_query=rewritten_query,
                answer="资料中没有相关信息。",
                grounded=False,
                sources=[],
            )

        answer = self._generate_answer(question, rewritten_query, rerank_hits)
        if "资料中没有相关信息" in answer:
            return ChatResponse(
                request_id=request_id,
                question=question,
                rewritten_query=rewritten_query,
                answer="资料中没有相关信息。",
                grounded=False,
                sources=[],
            )

        return ChatResponse(
            request_id=request_id,
            question=question,
            rewritten_query=rewritten_query,
            answer=answer,
            grounded=True,
            sources=[self._to_source(hit) for hit in rerank_hits],
        )

    def stream_answer_events(self, question: str):
        """SSE 事件流：先返回检索状态，再流式输出 answer，最后返回 sources。"""
        import json

        request_id = new_request_id()
        rewritten_query = safe_rewrite_query(question, self.active_documents)
        rewritten_query = self._expand_query_from_corpus(question, rewritten_query)
        yield self._sse("status", {"request_id": request_id, "stage": "query_rewrite", "rewritten_query": rewritten_query})

        scoped_chunks, requested_documents = self._scope_chunks(f"{question} {rewritten_query}")
        direct_image_caption = self._is_image_overview_question(question, scoped_chunks)
        document_overview = bool(requested_documents) and self._is_document_overview_question(question)
        if document_overview:
            rerank_hits = self._document_overview_hits(scoped_chunks)
            candidates = [(hit.chunk, hit.hybrid_score) for hit in rerank_hits]
        elif direct_image_caption:
            candidates = [
                (chunk, 1.0)
                for chunk in scoped_chunks
                if chunk.chunk_type == "image_caption" or "<summary>natural_image</summary>" in chunk.content
            ]
        elif requested_documents:
            candidates = [(chunk, 1.0) for chunk in scoped_chunks]
        else:
            hybrid_hits = self.retriever.retrieve(rewritten_query)
            candidates = [(hit.chunk, hit.score) for hit in hybrid_hits]

        yield self._sse("status", {"request_id": request_id, "stage": "rerank", "candidate_count": len(candidates)})
        if not document_overview:
            rerank_hits = self.reranker.rerank(
                rewritten_query,
                candidates,
                top_n=self._adaptive_top_n(question, rewritten_query),
            )
            rerank_hits = self._ensure_exact_evidence_coverage(question, rewritten_query, rerank_hits, scoped_chunks)

        if direct_image_caption and self._has_uninformative_image_caption(rerank_hits):
            answer = (
                "MinerU 已识别到该文件中的图片区域，但当前解析结果没有识别到可读文字标签或图片说明。"
                "因此系统只能确认它是一张图片/图形区域，不能基于现有证据进一步解释具体业务含义。"
            )
            yield self._sse("token", {"request_id": request_id, "text": answer})
            yield self._sse("done", {
                "request_id": request_id,
                "question": question,
                "rewritten_query": rewritten_query,
                "answer": answer,
                "grounded": True,
                "sources": [self._to_source(hit).model_dump() for hit in rerank_hits],
            })
            return

        if (
            not direct_image_caption
            and not document_overview
            and (
                (
                    should_refuse(rerank_hits)
                    and not self._has_exact_short_fact_support(rewritten_query, rerank_hits)
                )
                or self._has_missing_named_term_support(question, rerank_hits)
            )
        ):
            answer = "资料中没有相关信息。"
            yield self._sse("token", {"request_id": request_id, "text": answer})
            yield self._sse("done", {
                "request_id": request_id,
                "question": question,
                "rewritten_query": rewritten_query,
                "answer": answer,
                "grounded": False,
                "sources": [],
            })
            return

        if document_overview:
            # Some OpenAI-compatible providers produce a refusal only in
            # token-streaming mode for long multi-chunk summaries, while the
            # same prompt succeeds through invoke(). Generate the summary via
            # the stable path, then deliver the completed answer as an SSE
            # event so Streamlit and POST /chat stay behaviorally consistent.
            answer = self._generate_answer(question, rewritten_query, rerank_hits)
            grounded = "资料中没有相关信息" not in answer
            sources = [self._to_source(hit).model_dump() for hit in rerank_hits] if grounded else []
            yield self._sse("token", {"request_id": request_id, "text": answer})
            yield self._sse("done", {
                "request_id": request_id,
                "question": question,
                "rewritten_query": rewritten_query,
                "answer": answer if grounded else "资料中没有相关信息。",
                "grounded": grounded,
                "sources": sources,
            })
            return

        answer_parts: list[str] = []
        llm = create_chat_llm()
        prompt = build_rag_prompt(question, rewritten_query, rerank_hits)
        if llm is None:
            answer = self._fallback_answer(question, rerank_hits)
            answer_parts.append(answer)
            yield self._sse("token", {"request_id": request_id, "text": answer})
        else:
            try:
                for chunk in llm.stream(prompt):
                    text = getattr(chunk, "content", "") or ""
                    if not text:
                        continue
                    answer_parts.append(text)
                    yield self._sse("token", {"request_id": request_id, "text": text})
            except Exception:
                answer = self._fallback_answer(question, rerank_hits)
                answer_parts = [answer]
                yield self._sse("token", {"request_id": request_id, "text": answer})

        answer = "".join(answer_parts).strip()
        grounded = "资料中没有相关信息" not in answer
        sources = [self._to_source(hit).model_dump() for hit in rerank_hits] if grounded else []
        yield self._sse("done", {
            "request_id": request_id,
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer if grounded else "资料中没有相关信息。",
            "grounded": grounded,
            "sources": sources,
        })

    @staticmethod
    def _sse(event: str, data: dict) -> str:
        import json

        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _is_image_overview_question(self, question: str, scoped_chunks: list[ChunkRecord]) -> bool:
        """Route to direct image evidence only for actual image-focused questions.

        A PDF may contain a logo or portrait alongside many text chunks. Generic
        document questions such as "主要内容是什么" must still search the text
        instead of being captured by the single image block.
        """

        image_chunks = [chunk for chunk in scoped_chunks if self._is_image_chunk(chunk)]
        if not image_chunks:
            return False

        only_image_evidence = bool(scoped_chunks) and len(image_chunks) == len(scoped_chunks)
        lowered_question = question.casefold()
        explicit_image_phrases = (
            "描述了什么",
            "图中有什么",
            "图片中有什么",
            "图像中有什么",
            "图里有什么",
            "图表",
            "内嵌图",
            "画面是什么",
            "图片内容",
            "这是什么图",
            "解释这张图",
            "解释图片",
            "caption",
            "image",
        )
        return only_image_evidence or any(phrase in lowered_question for phrase in explicit_image_phrases)

    @staticmethod
    def _is_image_chunk(chunk: ChunkRecord) -> bool:
        return chunk.chunk_type == "image_caption" or "<summary>natural_image</summary>" in chunk.content

    @staticmethod
    def _is_document_overview_question(question: str) -> bool:
        return any(
            phrase in question
            for phrase in (
                "主要内容",
                "内容是什么",
                "讲了什么",
                "总结一下",
                "概括一下",
                "概述一下",
                "介绍一下这篇",
                "介绍一下这份",
            )
        )

    def _document_overview_hits(self, scoped_chunks: list[ChunkRecord], limit: int = 12) -> list[RerankHit]:
        """Select representative text evidence across a named document."""

        text_chunks = [
            chunk
            for chunk in scoped_chunks
            if not self._is_image_chunk(chunk) and len(chunk.content.strip()) >= 12
        ]
        if len(text_chunks) <= limit:
            selected = text_chunks
        else:
            indexes = {
                round(index * (len(text_chunks) - 1) / (limit - 1))
                for index in range(limit)
            }
            selected = [text_chunks[index] for index in sorted(indexes)]
        return [RerankHit(chunk=chunk, score=1.0, hybrid_score=1.0) for chunk in selected]

    def _scope_chunks(self, question: str) -> tuple[list[ChunkRecord], list[str]]:
        """Limit retrieval to files explicitly referenced in the user's question."""

        lowered_question = question.casefold()
        requested_documents = [
            file_name
            for file_name in self.active_documents
            if file_name.casefold() in lowered_question or Path(file_name).stem.casefold() in lowered_question
        ]
        if not requested_documents and self._references_current_document(question):
            document_like = self._document_like_names(self.active_documents)
            if len(document_like) == 1:
                requested_documents = document_like
        if not requested_documents:
            return self.chunks, []
        requested_names = {name.casefold() for name in requested_documents}
        scoped_chunks = [
            chunk
            for chunk in self.chunks
            if str(chunk.metadata.get("file_name") or "").casefold() in requested_names
        ]
        return scoped_chunks, requested_documents

    def _references_current_document(self, question: str) -> bool:
        return any(
            phrase in question
            for phrase in (
                "这篇论文",
                "本文",
                "这份文档",
                "这个文档",
                "该文档",
                "这份报告",
                "这个报告",
            )
        )

    def _document_like_names(self, file_names: list[str]) -> list[str]:
        document_suffixes = {".pdf", ".doc", ".docx", ".txt", ".md", ".markdown"}
        document_like = [
            name
            for name in file_names
            if Path(name).suffix.casefold() in document_suffixes
        ]
        return document_like or file_names

    def _ensure_exact_evidence_coverage(
        self,
        question: str,
        rewritten_query: str,
        hits: list[RerankHit],
        source_chunks: list[ChunkRecord],
    ) -> list[RerankHit]:
        """Make multi-point questions carry evidence for each explicit concept.

        Cross-encoder rerankers are good at global relevance, but they can still
        rank one concept very high and push the second concept below top_n. For
        questions like "分别介绍 A 和 B", the prompt must include evidence for
        both A and B, otherwise the LLM will correctly refuse.
        """

        phrase_groups = self._required_phrase_groups(f"{question} {rewritten_query}")
        if not phrase_groups or not source_chunks:
            return hits

        hit_by_id = {hit.chunk.chunk_id: hit for hit in hits}
        coverage_hits: list[RerankHit] = []
        coverage_score = max(
            [hit.score for hit in hits] + [self.settings.min_rerank_score + 0.1]
        )
        for phrases in phrase_groups:
            best_chunk = self._best_exact_phrase_chunk(phrases, source_chunks)
            if best_chunk is None:
                continue
            if best_chunk.chunk_id in hit_by_id:
                coverage_hits.append(hit_by_id[best_chunk.chunk_id])
            else:
                coverage_hits.append(RerankHit(chunk=best_chunk, score=coverage_score, hybrid_score=1.0))

        if not coverage_hits:
            return hits
        coverage_ids = {hit.chunk.chunk_id for hit in coverage_hits}
        rest = [hit for hit in hits if hit.chunk.chunk_id not in coverage_ids]
        return coverage_hits[:3] + rest

    def _required_phrase_groups(self, text: str) -> list[tuple[str, ...]]:
        lowered = text.casefold()
        groups: list[tuple[str, ...]] = []
        if any(marker in lowered for marker in ("掩码语言模型", "遮盖语言模型", "masked language model", "masked lm", "mlm")):
            groups.append(("masked language model", "masked lm", "mlm"))
        if any(marker in lowered for marker in ("下一句预测", "句子预测", "next sentence prediction", "nsp", "nep")):
            groups.append(("next sentence prediction", "nsp"))
        if any(marker in lowered for marker in ("输入表示", "输入表征", "input representation", "embedding", "embeddings")):
            groups.append(("input representation", "token embeddings", "segment embeddings", "position embeddings"))
        return groups

    def _best_exact_phrase_chunk(
        self,
        phrases: tuple[str, ...],
        source_chunks: list[ChunkRecord],
    ) -> ChunkRecord | None:
        best: tuple[int, ChunkRecord] | None = None
        lowered_phrases = tuple(phrase.casefold() for phrase in phrases)
        for chunk in source_chunks:
            content = chunk.content.casefold()
            title = str(chunk.metadata.get("title") or "").casefold()
            score = 0
            for phrase in lowered_phrases:
                score += content.count(phrase) * 3
                score += title.count(phrase) * 2
            if "task #" in content:
                score += 20
            if "pre-training" in title:
                score += 5
            if title.startswith("3.1"):
                score += 30
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, chunk)
        return best[1] if best else None

    def _expand_query_from_corpus(self, question: str, rewritten_query: str) -> str:
        """Append likely canonical terms found in the current knowledge base.

        This handles a common RAG failure mode: the user remembers a keyword
        slightly wrong, while the corpus contains a near match. We keep the
        original term and append corpus terms instead of silently replacing it.
        """

        query_terms = self._extract_candidate_terms(question)
        if not query_terms:
            return rewritten_query

        corpus_terms = self._collect_corpus_terms()
        additions: list[str] = []
        lowered_query = rewritten_query.casefold()
        for term in query_terms:
            for candidate in self._nearest_terms(term, corpus_terms):
                if candidate.casefold() not in lowered_query:
                    additions.append(candidate)

        if not additions:
            return rewritten_query

        seen: set[str] = set()
        deduped = []
        for term in additions:
            key = term.casefold()
            if key in seen:
                continue
            deduped.append(term)
            seen.add(key)
        return f"{rewritten_query} {' '.join(deduped[:8])}"

    def _extract_candidate_terms(self, text: str) -> list[str]:
        terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_.+-]{1,}\b", text)
        return [term for term in terms if len(term) >= 2]

    def _adaptive_top_n(self, question: str, query: str) -> int:
        """Use more evidence for multi-point questions."""

        acronym_count = len(set(re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", query)))
        asks_multiple_points = any(phrase in question for phrase in ("分别", "两大", "多个", "哪些", "对比", "区别"))
        if asks_multiple_points or acronym_count >= 2:
            return max(self.settings.rerank_top_n, 6)
        return self.settings.rerank_top_n

    def _collect_corpus_terms(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            for term in re.findall(r"\b[A-Za-z][A-Za-z0-9_.+-]{1,}\b", chunk.content):
                normalized = term.strip(".,;:()[]{}")
                if len(normalized) < 2:
                    continue
                # Acronyms and technical identifiers are the most typo-sensitive.
                if normalized.isupper() or any(char.isdigit() for char in normalized) or len(normalized) >= 4:
                    counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    def _nearest_terms(self, term: str, corpus_terms: dict[str, int]) -> list[str]:
        matches: list[tuple[str, int, float]] = []
        for candidate, count in corpus_terms.items():
            if candidate.casefold() == term.casefold():
                continue
            distance = self._edit_distance(term.casefold(), candidate.casefold(), max_distance=2)
            ratio = SequenceMatcher(None, term.casefold(), candidate.casefold()).ratio()
            if len(term) <= 4:
                similar = distance <= 1 and ratio >= 0.55
            else:
                similar = distance <= 2 or ratio >= 0.82
            if similar:
                matches.append((candidate, count, ratio))
        matches.sort(key=lambda item: (-item[1], -item[2], item[0]))
        return [candidate for candidate, _, _ in matches[:3]]

    def _edit_distance(self, left: str, right: str, max_distance: int = 2) -> int:
        if abs(len(left) - len(right)) > max_distance:
            return max_distance + 1
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            row_min = i
            for j, right_char in enumerate(right, start=1):
                cost = 0 if left_char == right_char else 1
                value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
                current.append(value)
                row_min = min(row_min, value)
            if row_min > max_distance:
                return max_distance + 1
            previous = current
        return previous[-1]

    def _has_uninformative_image_caption(self, hits) -> bool:
        evidence = "\n".join(hit.chunk.content for hit in hits)
        return "未识别到可读文字标签" in evidence or "No readable labels" in evidence

    def _has_exact_short_fact_support(self, query: str, hits) -> bool:
        """
        保护短结构化事实块。

        表格行、指标行经常只有 “SLA 99.9%” 这种短文本，cross-encoder reranker
        对这类短块可能打负分；如果混合检索已经强命中且 query token 与内容重合，
        不应该因为 rerank 负分直接拒答。
        """
        if not hits:
            return False
        top = hits[0]
        if top.hybrid_score < 0.5 or len(top.chunk.content) > 180:
            return False
        query_tokens = set(tokenize(query))
        content_tokens = set(tokenize(top.chunk.content))
        strong_tokens = {token for token in query_tokens if len(token) >= 2}
        return bool(strong_tokens & content_tokens)

    def _has_missing_named_term_support(self, question: str, hits) -> bool:
        """Reject answers about an explicit named technology absent from all retrieved evidence."""
        if not hits:
            return False
        stopwords = {"what", "where", "when", "which", "who", "why", "how", "does", "is", "are", "can"}
        named_terms = {
            term
            for term in re.findall(r"\b[A-Za-z][A-Za-z0-9_.+-]{2,}\b", question)
            if (
                term.lower() not in stopwords
                and len(term) > 3
                and (term[0].isupper() or term.isupper())
            )
        }
        if not named_terms:
            return False
        evidence = "\n".join(
            f"{hit.chunk.metadata.get('file_name', '')}\n{hit.chunk.content}"
            for hit in hits
        ).lower()
        evidence_terms = self._collect_terms_from_text(evidence)
        return any(not self._term_supported_by_evidence(term, evidence, evidence_terms) for term in named_terms)

    def _collect_terms_from_text(self, text: str) -> set[str]:
        return {
            term.casefold()
            for term in re.findall(r"\b[A-Za-z][A-Za-z0-9_.+-]{1,}\b", text)
            if len(term) >= 2
        }

    def _term_supported_by_evidence(self, term: str, evidence: str, evidence_terms: set[str]) -> bool:
        lowered = term.casefold()
        if lowered in evidence:
            return True
        return any(
            self._edit_distance(lowered, candidate, max_distance=2) <= (1 if len(term) <= 4 else 2)
            or SequenceMatcher(None, lowered, candidate).ratio() >= 0.82
            for candidate in evidence_terms
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=2))
    def _generate_answer(self, question: str, rewritten_query: str, hits) -> str:
        llm = create_chat_llm()
        if llm is None:
            return self._fallback_answer(question, hits)
        prompt = build_rag_prompt(question, rewritten_query, hits)
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return self._fallback_answer(question, hits)

    def _fallback_answer(self, question: str, hits) -> str:
        # 本地兜底只用于接口失败时保证链路可演示，真实答案仍应由 LLM 生成。
        if not hits:
            return "资料中没有相关信息。"
        source_ids = "、".join(hit.chunk.chunk_id for hit in hits)
        preview = hits[0].chunk.content[:160]
        return f"根据资料 {source_ids}，相关内容为：{preview}"

    def _to_source(self, hit) -> SourceChunk:
        metadata = hit.chunk.metadata
        return SourceChunk(
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            file_name=metadata.get("file_name"),
            page=metadata.get("page"),
            title=metadata.get("title"),
            chunk_type=metadata.get("chunk_type"),
            bbox=metadata.get("bbox"),
            score=hit.hybrid_score,
            rerank_score=hit.score,
            text=hit.chunk.content,
        )
