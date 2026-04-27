"""halluguard — reverse-RAG hallucination detector."""
from halluguard.guard import Guard
from halluguard.report import Claim, ClaimStatus, SupportReport

__all__ = ["Guard", "Claim", "ClaimStatus", "SupportReport"]
__version__ = "0.3.1"
