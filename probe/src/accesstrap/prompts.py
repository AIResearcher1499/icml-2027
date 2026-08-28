"""Frozen prompt templates. Do not retune after seeing numbers."""

MATH_SYSTEM = (
    "You are a careful math solver. Show step-by-step reasoning. "
    "Put the final numeric answer on its own line as `#### <number>`."
)

QA_SYSTEM = (
    "You are a careful question answering system. Reason step by step. "
    "End with a short final answer on its own line as `#### <answer>`."
)


def math_user(question: str, access_block: str | None) -> str:
    if not access_block:
        return question
    return (
        "The following identities were verified by a calculator (treat them as evidence, "
        "not as the final answer unless they already solve the question):\n"
        f"{access_block}\n\n"
        f"Question: {question}"
    )


def qa_user(question: str, access_block: str | None) -> str:
    if not access_block:
        return question
    return (
        "Evidence paragraphs (may or may not be sufficient; reason over them):\n"
        f"{access_block}\n\n"
        f"Question: {question}"
    )


def format_identities(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {expr} = {val}" for expr, val in pairs)
