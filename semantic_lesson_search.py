import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from data_types import SemanticSearchResult
from lesson_repository import Lesson

PROJECT_ROOT = Path(__file__).resolve().parent
STORAGE_PATH = PROJECT_ROOT / "storage" / "semantic_index"
METADATA_FILE_NAME = "index_metadata.json"
EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class IndexSettings:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


class SemanticLessonSearch:
    def __init__(
        self,
        lessons: list[Lesson],
        storage_dir: Path = STORAGE_PATH,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        force_rebuild_index: bool = False,
    ) -> None:

        self.storage_dir = storage_dir
        self.index_settings = IndexSettings(
            embedding_model=EMBEDDING_MODEL,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

        if force_rebuild_index or not self._can_load_existing_index():
            self.index = self._build_and_persist_index(
                lessons=lessons,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self.index = self._load_index()

        self.nodes = list(self.index.docstore.docs.values())
        self.retriever = self._create_hybrid_retriever()

    def _build_and_persist_index(
        self,
        lessons: list[Lesson],
        chunk_size: int,
        chunk_overlap: int,
    ) -> VectorStoreIndex:
        self._clear_storage_dir()

        documents = self._create_documents(lessons)
        nodes = self._create_nodes(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        index = self._create_index(nodes)
        index.storage_context.persist(persist_dir=str(self.storage_dir))
        self._write_index_metadata()

        return index

    @staticmethod
    def _create_documents(lessons: list[Lesson]) -> list[Document]:
        documents: list[Document] = []
        for lesson in lessons:
            documents.append(
                Document(
                    text=lesson.text,
                    metadata={
                        "section": lesson.section,
                        "title": lesson.title,
                        "path": str(lesson.path),
                    },
                )
            )
        return documents

    @staticmethod
    def _create_nodes(
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[BaseNode]:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.get_nodes_from_documents(documents)

    def _create_index(self, nodes: list[BaseNode]) -> VectorStoreIndex:
        return VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embedding_model,
        )

    def _load_index(self) -> BaseIndex:
        storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
        return load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.embedding_model,
        )

    def _index_exists(self) -> bool:
        return (self.storage_dir / "docstore.json").exists()

    def _can_load_existing_index(self) -> bool:
        return self._index_exists() and self._stored_settings_match_current_settings()

    @property
    def _metadata_path(self) -> Path:
        return self.storage_dir / METADATA_FILE_NAME

    def _stored_settings_match_current_settings(self) -> bool:
        if not self._metadata_path.exists():
            return False

        with self._metadata_path.open("r", encoding="utf-8") as metadata_file:
            stored_metadata = json.load(metadata_file)

        return stored_metadata == asdict(self.index_settings)

    def _write_index_metadata(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        with self._metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(asdict(self.index_settings), metadata_file, indent=2)

    def _clear_storage_dir(self) -> None:
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)

    @staticmethod
    def _to_search_results(nodes: list[NodeWithScore]) -> list[SemanticSearchResult]:
        results: list[SemanticSearchResult] = []

        for node_with_score in nodes:
            metadata = node_with_score.node.metadata

            results.append(
                SemanticSearchResult(
                    section=metadata["section"],
                    title=metadata["title"],
                    path=metadata["path"],
                    text=node_with_score.node.get_content(),
                    score=node_with_score.score or 0.0,
                )
            )

        return results

    def retrieve(self, question: str, top_k: int = 20) -> list[SemanticSearchResult]:
        nodes = self.retriever.retrieve(question)
        return self._to_search_results(nodes[:top_k])

    def _create_hybrid_retriever(self) -> QueryFusionRetriever:
        vector_retriever = self.index.as_retriever(similarity_top_k=20)

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=20,
        )

        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            llm=MockLLM(),
            mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=20,
            num_queries=1,
            use_async=False,
        )
