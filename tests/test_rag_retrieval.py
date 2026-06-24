from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from lesson_repository import LessonRepository
from semantic_lesson_search import SemanticLessonSearch


COURSE_ROOT = Path(r"C:\Users\galkr\Desktop\Gal\projects\hyperskill_ai_engineer")


@dataclass(frozen=True)
class RetrievalRegressionCase:
    name: str
    question: str
    expected_text: str


@pytest.fixture(scope="session")
def semantic_search() -> SemanticLessonSearch:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for embedding-based retrieval tests.")

    lessons = LessonRepository(COURSE_ROOT).load_lessons()
    return SemanticLessonSearch(lessons)


@pytest.mark.parametrize(
    "test_case",
    [
        RetrievalRegressionCase(
            name="find rag lesson",
            question="What is retrieval-augmented generation?",
            expected_text="retrieval-augmented generation",
        ),
        RetrievalRegressionCase(
            name="find agent memory lesson",
            question="How does agent memory work?",
            expected_text="short-term memory",
        ),
        RetrievalRegressionCase(
            name="find fox bingo regression sentence",
            question="why did the fox jump over the fence?",
            expected_text="one two three bingo",
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_retrieval_finds_expected_course_text(
    semantic_search: SemanticLessonSearch,
    test_case: RetrievalRegressionCase,
) -> None:
    results = semantic_search.retrieve(test_case.question)

    combined_text = "\n".join(result.text for result in results).lower()

    assert test_case.expected_text.lower() in combined_text
