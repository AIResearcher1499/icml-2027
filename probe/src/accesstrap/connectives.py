"""Frozen lexical forks. Do not edit after seeing probe numbers."""

CONNECTIVES = frozenset(
    {
        "therefore",
        "thus",
        "hence",
        "since",
        "because",
        "so",
        "wait",
        "however",
        "instead",
        "alternatively",
        "actually",
        "but",
        "first",
        "then",
        "let's",
        "hmm",
    }
)

MIN_LEXICAL_HITS = 20
TOP_ENTROPY_FRACTION = 0.20


def normalize_token(token: str) -> str:
    return token.lower().strip().strip(".,:;!?\"'`()[]{}")


def is_connective_token(token: str) -> bool:
    return normalize_token(token) in CONNECTIVES
