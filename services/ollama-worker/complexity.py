import re


def score_complexity(query: str) -> float:
    """
    Simple heuristic complexity scorer — returns 0.0 to 1.0.
    High score → route to large model.
    """
    score = 0.0
    q = query.strip().lower()

    # Length signal
    word_count = len(q.split())
    if word_count > 50:
        score += 0.4
    elif word_count > 20:
        score += 0.2

    # Technical keywords
    technical_terms = [
        "synthesize", "optimize", "architecture", "algorithm", "pipeline",
        "timing", "constraint", "netlist", "floorplan", "verification",
        "simulation", "power", "analyze", "compare", "difference between",
        "how does", "explain", "why does",
    ]
    matches = sum(1 for term in technical_terms if term in q)
    score += min(matches * 0.15, 0.45)

    # Question complexity markers
    if re.search(r"\b(versus|vs|compared to|trade.?off)\b", q):
        score += 0.2
    if re.search(r"\b(multi.step|step by step|walk me through)\b", q):
        score += 0.2

    return min(score, 1.0)
