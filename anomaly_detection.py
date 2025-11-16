# tax_optimization/anomaly_detection.py

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple

class TaxAnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.is_fitted = True
    
    def compute_anomaly_score(self, x: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        path_length = self._compute_path_length(x)
        n_samples = 100
        
        c_n = self._compute_average_path_length(n_samples)
        
        score = np.power(2, -path_length / c_n)
        
        return score
    
    def _compute_path_length(self, x: np.ndarray) -> float:
        decision_path = self.model.decision_function(x.reshape(1, -1))
        return -decision_path[0]
    
    def _compute_average_path_length(self, n: int) -> float:
        if n <= 1:
            return 0
        
        H_n_minus_1 = np.sum(1 / np.arange(1, n))
        
        c_n = 2 * H_n_minus_1 - (2 * (n - 1) / n)
        
        return c_n
    
    def detect_anomalies(self, X: np.ndarray, 
                        threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        predictions = self.model.predict(X)
        scores = np.array([self.compute_anomaly_score(x) for x in X])
        
        anomalies = scores >= threshold
        
        return anomalies, scores
    
    def explain_anomaly(self, x: np.ndarray, 
                       feature_names: List[str]) -> Dict:
        score = self.compute_anomaly_score(x)
        
        if score < 0.6:
            return {'is_anomaly': False, 'score': score}
        
        feature_contributions = self._compute_feature_contributions(x)
        
        top_features = np.argsort(np.abs(feature_contributions))[-5:][::-1]
        
        explanation = {
            'is_anomaly': True,
            'anomaly_score': score,
            'top_anomalous_features': [
                {
                    'feature': feature_names[idx],
                    'value': x[idx],
                    'contribution': feature_contributions[idx]
                }
                for idx in top_features
            ]
        }
        
        return explanation
    
    def _compute_feature_contributions(self, x: np.ndarray) -> np.ndarray:
        contributions = np.zeros(len(x))
        
        baseline_score = self.compute_anomaly_score(x)
        
        for i in range(len(x)):
            x_perturbed = x.copy()
            x_perturbed[i] = 0
            perturbed_score = self.compute_anomaly_score(x_perturbed)
            contributions[i] = baseline_score - perturbed_score
        
        return contributions

class TaxFraudDetector:
    def __init__(self):
        self.anomaly_detector = TaxAnomalyDetector(contamination=0.05)
        self.fraud_patterns = self._initialize_fraud_patterns()
    
    def _initialize_fraud_patterns(self) -> List[Dict]:
        return [
            {
                'name': 'excessive_deductions',
                'condition': lambda x: x['deduction_utilization_rate'] > 0.8
            },
            {
                'name': 'income_expense_mismatch',
                'condition': lambda x: x['total_expenses'] > x['total_income'] * 1.2
            },
            {
                'name': 'suspicious_income_diversity',
                'condition': lambda x: x['income_diversity'] > 6
            },
            {
                'name': 'unrealistic_deductions',
                'condition': lambda x: x['total_deductions'] > x['total_income'] * 0.5
            }
        ]
    
    def detect_fraud(self, features: Dict) -> Dict:
        rule_based_flags = []
        
        for pattern in self.fraud_patterns:
            if pattern['condition'](features):
                rule_based_flags.append(pattern['name'])
        
        feature_vector = np.array(list(features.values()))
        
        if self.anomaly_detector.is_fitted:
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomalies(
                feature_vector.reshape(1, -1)
            )
            is_anomaly = is_anomaly[0]
            anomaly_score = anomaly_score[0]
        else:
            is_anomaly = False
            anomaly_score = 0.0
        
        risk_score = len(rule_based_flags) * 20 + anomaly_score * 40
        risk_score = min(100, risk_score)
        
        return {
            'fraud_detected': len(rule_based_flags) > 0 or is_anomaly,
            'risk_score': risk_score,
            'rule_based_flags': rule_based_flags,
            'anomaly_detected': is_anomaly,
            'anomaly_score': anomaly_score,
            'recommendation': self._generate_recommendation(risk_score, rule_based_flags)
        }
    
    def _generate_recommendation(self, risk_score: float, 
                                flags: List[str]) -> str:
        if risk_score < 30:
            return "Low risk. Tax declaration appears normal."
        elif risk_score < 60:
            return f"Medium risk. Please review: {', '.join(flags)}"
        else:
            return f"High risk. Potential fraud detected: {', '.join(flags)}. Manual review required."