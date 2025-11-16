# tax_optimization/feature_engineering.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class TaxFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.income_categories = [
            'salary', 'bonus', 'investment', 'rental', 
            'business', 'freelance', 'capital_gains', 'other'
        ]
        self.expense_categories = [
            'education', 'medical', 'housing_loan', 'housing_rent',
            'elderly_care', 'child_education', 'continuing_education', 
            'serious_illness', 'infant_care', 'other'
        ]
        
    def build_income_vector(self, user_data: Dict) -> np.ndarray:
        income_vector = np.zeros(len(self.income_categories))
        
        for idx, category in enumerate(self.income_categories):
            if category in user_data.get('income', {}):
                income_vector[idx] = user_data['income'][category]
        
        return income_vector
    
    def build_expense_vector(self, user_data: Dict) -> np.ndarray:
        expense_vector = np.zeros(len(self.expense_categories))
        
        for idx, category in enumerate(self.expense_categories):
            if category in user_data.get('expenses', {}):
                expense_vector[idx] = user_data['expenses'][category]
        
        return expense_vector
    
    def build_deduction_vector(self, user_data: Dict) -> np.ndarray:
        deductions = user_data.get('deductions', {})
        
        deduction_features = [
            deductions.get('children_education', 0),
            deductions.get('continuing_education', 0),
            deductions.get('serious_illness', 0),
            deductions.get('housing_loan_interest', 0),
            deductions.get('housing_rent', 0),
            deductions.get('elderly_support', 0),
            deductions.get('infant_care', 0)
        ]
        
        return np.array(deduction_features)
    
    def compute_tax_contribution_features(self, income: np.ndarray, 
                                         expenses: np.ndarray,
                                         deductions: np.ndarray) -> Dict:
        total_income = np.sum(income)
        total_expenses = np.sum(expenses)
        total_deductions = np.sum(deductions)
        
        taxable_income = max(0, total_income - 60000 - total_deductions)
        
        features = {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'total_deductions': total_deductions,
            'taxable_income': taxable_income,
            'income_diversity': np.count_nonzero(income),
            'expense_diversity': np.count_nonzero(expenses),
            'deduction_utilization_rate': total_deductions / max(total_income, 1),
            'income_concentration': np.max(income) / max(total_income, 1),
            'expense_concentration': np.max(expenses) / max(total_expenses, 1)
        }
        
        for idx, category in enumerate(self.income_categories):
            features[f'income_{category}'] = income[idx]
            features[f'income_{category}_ratio'] = income[idx] / max(total_income, 1)
        
        for idx, category in enumerate(self.expense_categories):
            features[f'expense_{category}'] = expenses[idx]
            features[f'expense_{category}_ratio'] = expenses[idx] / max(total_expenses, 1)
        
        deduction_names = [
            'children_education', 'continuing_education', 'serious_illness',
            'housing_loan', 'housing_rent', 'elderly_support', 'infant_care'
        ]
        for idx, name in enumerate(deduction_names):
            features[f'deduction_{name}'] = deductions[idx]
        
        return features
    
    def generate_comprehensive_features(self, user_data: Dict) -> pd.DataFrame:
        income_vector = self.build_income_vector(user_data)
        expense_vector = self.build_expense_vector(user_data)
        deduction_vector = self.build_deduction_vector(user_data)
        
        basic_features = self.compute_tax_contribution_features(
            income_vector, expense_vector, deduction_vector
        )
        
        temporal_features = self._generate_temporal_features(user_data)
        interaction_features = self._generate_interaction_features(
            income_vector, expense_vector, deduction_vector
        )
        
        all_features = {**basic_features, **temporal_features, **interaction_features}
        
        return pd.DataFrame([all_features])
    
    def _generate_temporal_features(self, user_data: Dict) -> Dict:
        current_month = datetime.now().month
        current_quarter = (current_month - 1) // 3 + 1
        
        return {
            'month': current_month,
            'quarter': current_quarter,
            'is_year_end': int(current_month == 12),
            'months_remaining': 12 - current_month
        }
    
    def _generate_interaction_features(self, income: np.ndarray, 
                                       expenses: np.ndarray,
                                       deductions: np.ndarray) -> Dict:
        total_income = np.sum(income)
        total_deductions = np.sum(deductions)
        
        salary_income = income[0]
        business_income = income[4]
        
        housing_deduction = deductions[3] + deductions[4]
        family_deduction = deductions[0] + deductions[5] + deductions[6]
        
        return {
            'salary_to_total_ratio': salary_income / max(total_income, 1),
            'business_to_total_ratio': business_income / max(total_income, 1),
            'housing_deduction_total': housing_deduction,
            'family_deduction_total': family_deduction,
            'deduction_efficiency': total_deductions / max(total_income, 1),
            'income_deduction_gap': total_income - total_deductions
        }