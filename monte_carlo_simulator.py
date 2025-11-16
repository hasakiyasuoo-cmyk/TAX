# tax_optimization/monte_carlo_simulator.py

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

class MonteCarloTaxSimulator:
    def __init__(self, tax_calculator):
        self.tax_calculator = tax_calculator
        self.n_simulations = 10000
    
    def simulate_tax_strategies(self, 
                               income_vector: np.ndarray,
                               expense_vector: np.ndarray,
                               deduction_vector: np.ndarray,
                               strategies: List[Dict],
                               uncertainty_params: Dict) -> Dict:
        results = {}
        
        for strategy_idx, strategy in enumerate(strategies):
            strategy_name = strategy.get('name', f'Strategy_{strategy_idx}')
            
            tax_samples = self._run_simulations(
                income_vector, 
                expense_vector, 
                deduction_vector,
                strategy,
                uncertainty_params
            )
            
            results[strategy_name] = self._compute_statistics(tax_samples)
        
        best_strategy = min(results.items(), 
                           key=lambda x: x[1]['expected_tax'])
        
        return {
            'strategies': results,
            'best_strategy': best_strategy[0],
            'best_strategy_stats': best_strategy[1]
        }
    
    def _run_simulations(self,
                        income_vector: np.ndarray,
                        expense_vector: np.ndarray,
                        deduction_vector: np.ndarray,
                        strategy: Dict,
                        uncertainty_params: Dict) -> np.ndarray:
        tax_samples = np.zeros(self.n_simulations)
        
        for i in range(self.n_simulations):
            simulated_income = self._simulate_income(
                income_vector, 
                uncertainty_params.get('income_std', 0.05)
            )
            
            simulated_deductions = self._simulate_deductions(
                deduction_vector,
                uncertainty_params.get('deduction_std', 0.02)
            )
            
            tax = self.tax_calculator.calculate_tax_for_strategy(
                simulated_income,
                expense_vector,
                simulated_deductions,
                strategy
            )
            
            tax_samples[i] = tax
        
        return tax_samples
    
    def _simulate_income(self, income_vector: np.ndarray, 
                        std_ratio: float) -> np.ndarray:
        simulated = np.zeros_like(income_vector)
        
        for i, income in enumerate(income_vector):
            if income > 0:
                std = income * std_ratio
                simulated[i] = max(0, np.random.normal(income, std))
            else:
                simulated[i] = 0
        
        return simulated
    
    def _simulate_deductions(self, deduction_vector: np.ndarray,
                            std_ratio: float) -> np.ndarray:
        simulated = np.zeros_like(deduction_vector)
        
        for i, deduction in enumerate(deduction_vector):
            if deduction > 0:
                std = deduction * std_ratio
                simulated[i] = max(0, np.random.normal(deduction, std))
            else:
                simulated[i] = 0
        
        return simulated
    
    def _compute_statistics(self, tax_samples: np.ndarray) -> Dict:
        expected_tax = np.mean(tax_samples)
        std_tax = np.std(tax_samples)
        
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        ci_lower = expected_tax - z_score * std_tax
        ci_upper = expected_tax + z_score * std_tax
        
        percentiles = np.percentile(tax_samples, [5, 25, 50, 75, 95])
        
        return {
            'expected_tax': expected_tax,
            'std_tax': std_tax,
            'min_tax': np.min(tax_samples),
            'max_tax': np.max(tax_samples),
            'confidence_interval': (ci_lower, ci_upper),
            'percentile_5': percentiles[0],
            'percentile_25': percentiles[1],
            'median': percentiles[2],
            'percentile_75': percentiles[3],
            'percentile_95': percentiles[4],
            'var_95': percentiles[4]
        }
    
    def compare_strategies_statistical(self, 
                                      results: Dict,
                                      significance_level: float = 0.05) -> Dict:
        strategy_names = list(results['strategies'].keys())
        comparisons = {}
        
        for i in range(len(strategy_names)):
            for j in range(i + 1, len(strategy_names)):
                strategy_a = strategy_names[i]
                strategy_b = strategy_names[j]
                
                mean_a = results['strategies'][strategy_a]['expected_tax']
                mean_b = results['strategies'][strategy_b]['expected_tax']
                std_a = results['strategies'][strategy_a]['std_tax']
                std_b = results['strategies'][strategy_b]['std_tax']
                
                t_stat = (mean_a - mean_b) / np.sqrt(std_a**2 + std_b**2)
                
                comparison_key = f"{strategy_a}_vs_{strategy_b}"
                comparisons[comparison_key] = {
                    'mean_difference': mean_a - mean_b,
                    't_statistic': t_stat,
                    'better_strategy': strategy_a if mean_a < mean_b else strategy_b,
                    'savings_if_switch': abs(mean_a - mean_b)
                }
        
        return comparisons