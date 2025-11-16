import numpy as np

class TaxOptimizer:
    def optimize(self, income: float, expenses: float, deductions: float):
        plans = []
        for shift in [-1000, 0, 1000]:
            tax = max(0, income - deductions - shift) * 0.2
            plans.append({"shift": shift, "estimated_tax": tax})
        best_plan = min(plans, key=lambda x: x["estimated_tax"])
        return best_plan