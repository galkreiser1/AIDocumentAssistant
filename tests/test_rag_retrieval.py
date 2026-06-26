from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from data_types import SemanticSearchResult
from lesson_repository import LessonRepository
from semantic_lesson_search import SemanticLessonSearch


COURSE_ROOT = Path(r"C:\Users\galkr\Desktop\Gal\projects\hyperskill_ai_engineer")


@dataclass(frozen=True)
class RetrievalRegressionCase:
    name: str
    question: str
    expected_text: str
    expected_title: str
    max_expected_rank: int = 3


@pytest.fixture(scope="session")
def semantic_search() -> SemanticLessonSearch:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for embedding-based retrieval tests.")

    lessons = LessonRepository(COURSE_ROOT).load_lessons()
    return SemanticLessonSearch(lessons)


def _find_text_rank(
    results: list[SemanticSearchResult],
    expected_text: str,
) -> int | None:
    expected_text = expected_text.lower()

    for rank, result in enumerate(results, start=1):
        if expected_text in result.text.lower():
            return rank

    return None


def _format_top_results(results: list[SemanticSearchResult], limit: int = 5) -> str:
    lines: list[str] = []

    for rank, result in enumerate(results[:limit], start=1):
        preview = result.text.replace("\n", " ")[:240]
        lines.append(
            f"{rank}. score={result.score:.4f} | "
            f"{result.section} / {result.title} | {preview}"
        )

    return "\n".join(lines)


@pytest.mark.parametrize(
    "test_case",
    [
        RetrievalRegressionCase(
            name="find rag lesson",
            question="What is retrieval-augmented generation?",
            expected_text="retrieval-augmented generation",
            expected_title="Introduction to retrieval-augmented generation",
        ),
        RetrievalRegressionCase(
            name="find agent memory lesson",
            question="How does agent memory work?",
            expected_text="short-term memory",
            expected_title="Memory in agents",
        ),
        RetrievalRegressionCase(
            name="find fox bingo regression sentence",
            question="why did the fox jump over the fence?",
            expected_text="one two three bingo",
            expected_title="LLM regression testing",
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_retrieval_ranks_expected_course_text_high_enough(
    semantic_search: SemanticLessonSearch,
    test_case: RetrievalRegressionCase,
) -> None:
    results = semantic_search.retrieve(test_case.question)

    found_rank = _find_text_rank(results, test_case.expected_text)

    assert found_rank is not None, (
        f"Expected text was not found for question: {test_case.question}\n"
        f"Expected text: {test_case.expected_text}\n\n"
        f"Top retrieved results:\n{_format_top_results(results)}"
    )
    assert found_rank <= test_case.max_expected_rank, (
        f"Expected text was found at rank {found_rank}, "
        f"but expected rank <= {test_case.max_expected_rank}.\n\n"
        f"Top retrieved results:\n{_format_top_results(results)}"
    )


@pytest.mark.parametrize(
    "test_case",
    [
        RetrievalRegressionCase(
            name="rank rag lesson title",
            question="What is retrieval-augmented generation?",
            expected_text="retrieval-augmented generation",
            expected_title="Introduction to retrieval-augmented generation",
        ),
        RetrievalRegressionCase(
            name="rank memory lesson title",
            question="How does agent memory work?",
            expected_text="short-term memory",
            expected_title="Memory in agents",
        ),
        RetrievalRegressionCase(
            name="rank regression lesson title",
            question="why did the fox jump over the fence?",
            expected_text="one two three bingo",
            expected_title="LLM regression testing",
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_retrieval_ranks_expected_lesson_title_high_enough(
    semantic_search: SemanticLessonSearch,
    test_case: RetrievalRegressionCase,
) -> None:
    results = semantic_search.retrieve(test_case.question)
    top_titles = [result.title for result in results[: test_case.max_expected_rank]]

    assert test_case.expected_title in top_titles, (
        f"Expected lesson title was not in the top {test_case.max_expected_rank} "
        f"results for question: {test_case.question}\n"
        f"Expected title: {test_case.expected_title}\n"
        f"Top titles: {top_titles}\n\n"
        f"Top retrieved results:\n{_format_top_results(results)}"
    )
