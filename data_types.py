from __future__ import annotations

from dataclasses import dataclass

from lesson_repository import Lesson


@dataclass
class SemanticSearchResult:
    section: str
    title: str
    path: str
    text: str
    score: float


@dataclass
class SearchResult:
    lesson: Lesson
    score: int
