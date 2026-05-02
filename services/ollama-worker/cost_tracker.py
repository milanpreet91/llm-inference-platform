import os


COST_PER_1K = {
    os.getenv("SMALL_MODEL", "llama3.2:1b"): float(os.getenv("SMALL_MODEL_COST", 0.0002)),
    os.getenv("LARGE_MODEL", "llama3.2:3b"): float(os.getenv("LARGE_MODEL_COST", 0.0008)),
}


class CostTracker:
    @staticmethod
    def calculate(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rate = COST_PER_1K.get(model, 0.0005)
        total_tokens = prompt_tokens + completion_tokens
        return round((total_tokens / 1000) * rate, 6)
