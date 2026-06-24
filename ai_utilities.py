from __future__ import annotations

from textwrap import dedent

from data_types import SemanticSearchResult


class AiUtilities:
    @staticmethod
    def build_context(
        search_results: list[SemanticSearchResult],
    ) -> str:
        context_parts: list[str] = []
        for i, result in enumerate(search_results, start=1):
            text = result.text
            context_parts.append(f"SOURCE {i}\n"
                                 f"Section: {result.section}\n"
                                 f"Title: {result.title}\n"
                                 f"Score: {result.score:.4f}\n"
                                 f"Text:\n{text}\n")
        return "\n".join(context_parts)

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        return dedent(
            f"""\
            You are a course assistant for the Hyperskill AI Engineer course.

            Answer the user's question using only the provided course sources.
            If the sources do not contain enough information, say that the course material shown is not enough.
            Cite the lesson titles you used.

            COURSE SOURCES:
            {context}

            USER QUESTION:
            {question}

            ANSWER:
            """
        )
