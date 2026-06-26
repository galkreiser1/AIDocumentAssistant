from __future__ import annotations

import argparse
from pathlib import Path

from course_assistant import AssistantResponse, CourseAssistant

COURSE_ROOT = Path(r"C:\Users\galkr\Desktop\Gal\projects\hyperskill_ai_engineer")
EXIT_COMMANDS = {"exit", "quit", "q"}
SOURCE_DISPLAY_LIMIT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about the local Hyperskill AI Engineer course.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the persisted semantic search index before answering.",
    )
    parser.add_argument(
        "--show-info",
        action="store_true",
        help="Display information about retrieved sources."
    )
    return parser.parse_args()


def should_exit(user_input: str) -> bool:
    return user_input.strip().lower() in EXIT_COMMANDS


def print_response(response: AssistantResponse, print_top_sources: bool = False) -> None:
    print("\nAnswer:\n")
    print(response.answer)

    if print_top_sources:
        print("\nTop sources:")
        for index, source in enumerate(response.sources[:SOURCE_DISPLAY_LIMIT], start=1):
            print(f"{index}. [{source.score:.4f}] {source.section} / {source.title}")


def main() -> None:
    args = parse_args()
    assistant = CourseAssistant(
        course_root=COURSE_ROOT,
        force_rebuild_index=args.rebuild_index,
    )

    print("Ask a course question. Type 'exit' to quit.")

    while True:
        question = input("\n> ").strip()

        if should_exit(question):
            print("Bye.")
            break

        if not question:
            continue

        print("Generating answer...")
        response = assistant.answer_question(question)
        print_response(response, args.show_info)


if __name__ == "__main__":
    main()
