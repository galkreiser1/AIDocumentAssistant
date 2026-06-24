from __future__ import annotations

from pathlib import Path

from course_assistant import CourseAssistant

COURSE_ROOT = Path(r"C:\Users\galkr\Desktop\Gal\projects\hyperskill_ai_engineer")


def main() -> None:
    assistant = CourseAssistant(COURSE_ROOT)

    question = input("Ask a course question: ")
    response = assistant.answer_question(question)

    print("\nAnswer:\n")
    print(response.answer)
    print("\nSources:")
    for source in response.sources:
        print(f"- [{source.score:.4f}] {source.section} / {source.title}")


if __name__ == "__main__":
    main()
