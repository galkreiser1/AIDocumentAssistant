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
EMBEDDING_MODEL = "text-embedding-3-small"


class SemanticLessonSearch:
    def __init__(
        self,
        lessons: list[Lesson],
        storage_dir: Path = STORAGE_PATH,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:

        self.storage_dir = storage_dir
        self.embedding_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

        if self._index_exists():
            self.index = self._load_index()
        else:
            documents = self._create_documents(lessons)
            nodes = self._create_nodes(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.index = self._create_index(nodes)
            self.index.storage_context.persist(persist_dir=str(storage_dir))

        self.nodes = list(self.index.docstore.docs.values())
        self.retriever = self._create_hybrid_retriever()

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
