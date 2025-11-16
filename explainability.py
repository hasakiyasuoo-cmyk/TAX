# tax_optimization/explainability.py

import numpy as np
import pandas as pd
from typing import Dict, List
import shap

class TaxSHAPExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.base_value = None
    
    def fit(self, X: pd.DataFrame):
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X)
        self.base_value = self.explainer.expected_value
    
    def explain_prediction(self, x: np.ndarray, 
                          prediction: float) -> Dict:
        if self.explainer is None:
            raise ValueError("Explainer not fitted")
        
        shap_values = self.explainer.shap_values(x.reshape(1, -1))[0]