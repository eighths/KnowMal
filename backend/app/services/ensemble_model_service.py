import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from app.models.ensemble_model import EnsembleMalwareClassifier

class EnsembleModelService:
    def __init__(self):
        self.classifier = None
        self.model_loaded = False
        self.models_dir = Path(__file__).parent.parent / "models"
        self.ensemble_model_path = self.models_dir / "ensemble_model.pkl"
    
    def load_models(self) -> bool:
        try:
            if not self.models_dir.exists():
                return False
            
            if not self.ensemble_model_path.exists():
                return False
            
            self.classifier = EnsembleMalwareClassifier()
            success = self.classifier.load_model(str(self.ensemble_model_path))
            
            if success:
                self.model_loaded = True
                model_info = self.classifier.get_model_info()
                return True
            else:
                return False
                
        except Exception as e:
            self.model_loaded = False
            return False
    
    def predict_malware_type(self, analysis_report: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            predicted_classes, class_probabilities = self.classifier.predict(analysis_report)
            
            if not predicted_classes:
                return None
            
            result = {
                "ai_analysis": {
                    "predicted_types": predicted_classes,
                    "class_probabilities": class_probabilities,
                    "model_type": "ensemble",
                    "model_info": self.classifier.get_model_info()
                }
            }
            
            return result
            
        except Exception as e:
            return None
    
    def predict_static_only(self, static_features: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            analysis_report = {'features': static_features}
            
            hard_labels, soft_labels, network_probs = self.classifier.predict_static_only(static_features)
            
            result = {
                "ai_analysis": {
                    "predicted_types": hard_labels,
                    "class_probabilities": soft_labels,
                    "network_feature_probabilities": network_probs,
                    "model_type": "ensemble"
                }
            }
            
            return result
            
        except Exception as e:
            return None
    
    def _extract_static_features_from_report(self, analysis_report: Dict) -> Optional[Dict]:
        try:
            if 'report' in analysis_report:
                actual_report = analysis_report['report']
                features = actual_report.get('features', {})
            else:
                features = analysis_report.get('features', {})
            
            if not features:
                return None
            
            return features
            
        except Exception as e:
            return None
    
    def get_model_status(self) -> Dict[str, any]:
        return {
            "model_loaded": self.model_loaded,
            "model_path": str(self.ensemble_model_path),
            "model_exists": self.ensemble_model_path.exists(),
            "model_info": self.classifier.get_model_info() if self.classifier else None
        }
    
    def reload_model(self) -> bool:
        self.model_loaded = False
        self.classifier = None
        return self.load_models()

_ensemble_model_service = None

def get_ensemble_model_service() -> EnsembleModelService:
    global _ensemble_model_service
    if _ensemble_model_service is None:
        _ensemble_model_service = EnsembleModelService()
        _ensemble_model_service.load_models()
    return _ensemble_model_service

