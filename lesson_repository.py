from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Lesson:
    section: str
    title: str
    path: Path
    text: str


class LessonRepository:
    def __init__(self, course_root: Path) -> None:
        self.course_root = course_root
        self.course_index_path = course_root / "course_index.json"

    def load_lessons(self) -> list[Lesson]:
        index = self._load_course_index()
        lessons: list[Lesson] = []

        for section in index["sections"]:
            section_name = section["name"]

            for title in section["lessons"]:
                path = self._lesson_path(section_name, title)
                lessons.append(
                    Lesson(
                        section=section_name,
                        title=title,
                        path=path,
                        text=self._read_lesson_text(path),
                    )
                )

        return lessons

    def _load_course_index(self) -> dict[str, Any]:
        return self._load_json(self.course_index_path)

    def _lesson_path(self, section: str, title: str) -> Path:
        return self.course_root / section / f"{title}.txt"

    def _read_lesson_text(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Lesson file not found: {path}")

        return path.read_text(encoding="utf-8")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
