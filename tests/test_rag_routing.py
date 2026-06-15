from __future__ import annotations

import unittest

from app.schemas.document import ChunkRecord
from app.services.rag_service import RagService


def chunk(chunk_id: str, chunk_type: str, content: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="doc-resume",
        chunk_type=chunk_type,
        content=content,
        metadata={"file_name": "resume.pdf"},
    )


class ImageRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RagService.__new__(RagService)
        self.text_chunk = chunk("text-1", "ocr_text", "熟悉 RAG、深度学习与大模型应用开发。")
        self.image_chunk = chunk(
            "image-1",
            "image_caption",
            "MinerU 识别到图片区域，但未识别到可读文字标签或图片说明。",
        )

    def test_document_summary_is_not_captured_by_embedded_image(self) -> None:
        scoped = [self.text_chunk, self.image_chunk]
        self.assertFalse(self.service._is_image_overview_question("resume.pdf 主要内容是什么", scoped))

    def test_explicit_image_question_uses_image_route(self) -> None:
        scoped = [self.text_chunk, self.image_chunk]
        self.assertTrue(self.service._is_image_overview_question("解释这张图片内容", scoped))

    def test_image_only_scope_uses_image_route(self) -> None:
        self.assertTrue(self.service._is_image_overview_question("这是什么", [self.image_chunk]))

    def test_named_document_summary_uses_overview_route(self) -> None:
        self.assertTrue(self.service._is_document_overview_question("resume.pdf 这篇文档主要内容是什么"))

    def test_document_overview_evidence_excludes_images(self) -> None:
        hits = self.service._document_overview_hits([self.text_chunk, self.image_chunk])
        self.assertEqual([hit.chunk.chunk_id for hit in hits], ["text-1"])


if __name__ == "__main__":
    unittest.main()
