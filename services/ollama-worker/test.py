from complexity import score_complexity
from cost_tracker import CostTracker

# Test complexity scorer
print("=== Complexity Scorer ===")
tests = [
    'what is timing constraint',
    'how does synthesis pipeline work compared to simulation',
    'hi',
    'explain the difference between netlist optimization and floorplan architecture step by step',
]
for t in tests:
    print(f'{score_complexity(t):.2f}  |  {t[:60]}')

# Test cost tracker
print("\n=== Cost Tracker ===")
cost = CostTracker.calculate(model="llama3.2:1b", prompt_tokens=200, completion_tokens=150)
print(f"Small model cost: ${cost}")

cost = CostTracker.calculate(model="llama3.2:3b", prompt_tokens=200, completion_tokens=150)
print(f"Large model cost: ${cost}")
