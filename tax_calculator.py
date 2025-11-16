# tax_optimization/tax_calculator.py

import numpy as np
from typing import Dict, List, Tuple

class TaxCalculator:
    def __init__(self):
        self.tax_brackets = [
            (36000, 0.03),
            (144000, 0.10),
            (300000, 0.20),
            (420000, 0.25),
            (660000, 0.30),
            (960000, 0.35),
            (float('inf'), 0.45)
        ]
        
        self.quick_deductions = [0, 2520, 16920, 31920, 52920, 85920, 181920]
        
        self.standard_deduction = 60000
    
    def calculate_personal_income_tax(self, annual_income: float, 
                                     deductions: float,
                                     special_deductions: float = 0) -> Dict:
        taxable_income = max(0, annual_income - self.standard_deduction - 
                            deductions - special_deductions)
        
        tax_amount = 0
        applicable_rate = 0
        quick_deduction = 0
        
        for idx, (bracket, rate) in enumerate(self.tax_brackets):
            if taxable_income <= bracket:
                applicable_rate = rate
                quick_deduction = self.quick_deductions[idx]
                break
        
        tax_amount = taxable_income * applicable_rate - quick_deduction
        
        effective_rate = tax_amount / annual_income if annual_income > 0 else 0
        
        return {
            'taxable_income': taxable_income,
            'tax_amount': tax_amount,
            'applicable_rate': applicable_rate,
            'effective_rate': effective_rate,
            'after_tax_income': annual_income - tax_amount,
            'quick_deduction': quick_deduction
        }
    
    def calculate_progressive_tax(self, taxable_income: float) -> float:
        if taxable_income <= 0:
            return 0
        
        tax = 0
        remaining = taxable_income
        prev_bracket = 0
        
        for bracket, rate in self.tax_brackets:
            if remaining <= 0:
                break
            
            taxable_in_bracket = min(remaining, bracket - prev_bracket)
            tax += taxable_in_bracket * rate
            remaining -= taxable_in_bracket
            prev_bracket = bracket
        
        return tax
    
    def calculate_tax_for_strategy(self, income_vector: np.ndarray,
                                   expense_vector: np.ndarray,
                                   deduction_vector: np.ndarray,
                                   strategy: Dict) -> float:
        total_income = np.sum(income_vector)
        
        adjusted_deductions = deduction_vector.copy()
        if 'deduction_adjustments' in strategy:
            for idx, adjustment in strategy['deduction_adjustments'].items():
                adjusted_deductions[idx] = adjustment
        
        total_deductions = np.sum(adjusted_deductions)
        
        income_adjustments = strategy.get('income_split', {})
        adjusted_income = total_income
        for adjustment_type, amount in income_adjustments.items():
            if adjustment_type == 'year_end_bonus_separate':
                adjusted_income -= amount
        
        result = self.calculate_personal_income_tax(
            adjusted_income,
            total_deductions
        )
        
        if 'year_end_bonus_separate' in income_adjustments:
            bonus_amount = income_adjustments['year_end_bonus_separate']
            bonus_tax = self._calculate_year_end_bonus_tax(bonus_amount)
            result['tax_amount'] += bonus_tax
            result['after_tax_income'] -= bonus_tax
        
        return result['tax_amount']
    
    def _calculate_year_end_bonus_tax(self, bonus: float) -> float:
        monthly_average = bonus / 12
        
        for idx, (bracket, rate) in enumerate(self.tax_brackets):
            if monthly_average <= bracket:
                quick_deduction = self.quick_deductions[idx]
                return bonus * rate - quick_deduction
        
        return bonus * 0.45 - 181920
    
    def estimate_tax_savings(self, current_tax: float, 
                            optimized_tax: float) -> Dict:
        savings = current_tax - optimized_tax
        savings_rate = (savings / current_tax * 100) if current_tax > 0 else 0
        
        return {
            'current_tax': current_tax,
            'optimized_tax': optimized_tax,
            'absolute_savings': savings,
            'savings_rate': savings_rate
        }