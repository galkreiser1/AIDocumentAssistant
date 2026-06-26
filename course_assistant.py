from dataclasses import dataclass
from pathlib import Path

from ai_utilities import AiUtilities
from data_types import SemanticSearchResult
from lesson_repository import LessonRepository
from openai_answer_generator import OpenAIAnswerGenerator
from semantic_lesson_search import SemanticLessonSearch


@dataclass
class AssistantResponse:
    question: str
    answer: str
    sources: list[SemanticSearchResult]


class CourseAssistant:
    def __init__(
        self,
        course_root: Path,
        force_rebuild_index: bool = False,
    ) -> None:
        self.lesson_repository = LessonRepository(course_root)
        self.semantic_lesson_search = SemanticLessonSearch(
            lessons=self.lesson_repository.load_lessons(),
            force_rebuild_index=force_rebuild_index,
        )

        self.answer_generator = OpenAIAnswerGenerator()

    def answer_question(self, question: str) -> AssistantResponse:
        search_results = self.semantic_lesson_search.retrieve(question)
        context = AiUtilities.build_context(search_results)
        prompt = AiUtilities.build_prompt(question, context)
        answer = self.answer_generator.generate_answer(prompt)

        return AssistantResponse(
            question=question,
            answer=answer,
            sources=search_results,
        )





