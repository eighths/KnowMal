import os
import pickle
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from app.models.kd_model import FinalEnhancedMalwareDocumentClassifier

class AIModelService:
    
    def __init__(self):
        self.classifier = None
        self.model_loaded = False
        self.models_dir = Path(__file__).parent.parent / "models"
    
    def load_models(self) -> bool:
        try:
            if not self.models_dir.exists():
                return False
            
            required_files = [
                'individual_models.pkl',
                'metadata.pkl', 
                'network_estimator.pkl',
                'soft_label_model.pkl'
            ]
            
            for file_name in required_files:
                file_path = self.models_dir / file_name
                if not file_path.exists():
                    return False
            
            self.classifier = FinalEnhancedMalwareDocumentClassifier()
            self.classifier.load_models(str(self.models_dir))
            
            if not hasattr(self.classifier, 'individual_models'):
                self.model_loaded = False
                return False
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            self.model_loaded = False
            return False
    
    def predict_malware_type(self, analysis_report: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            static_features = self._extract_static_features_from_report(analysis_report)
            
            if not static_features:
                return None
            
            hard_labels, soft_labels, network_probs = self.classifier.predict_static_only(static_features)
            
            result = {
                "ai_analysis": {
                    "predicted_types": hard_labels,
                    "network_feature_probabilities": network_probs,
                    "model_classes": self.classifier.classes
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
            
            structure = features.get('structure', {})
            macros = features.get('macros', {})
            strings = features.get('strings', {})
            apis = features.get('apis', {})
            obfuscation = features.get('obfuscation', {})
            network_indicators = features.get('network_indicators', {})
            security_indicators = features.get('security_indicators', {})
            
            static_features = {
                'structure': structure,
                'macros': macros,
                'strings': strings,
                'apis': apis,
                'obfuscation': obfuscation,
                'network_indicators': network_indicators,
                'security_indicators': security_indicators
            }
            
            return static_features
            
        except Exception as e:
            return None

_ai_model_service = None

def get_ai_model_service() -> AIModelService:
    global _ai_model_service
    if _ai_model_service is None:
        _ai_model_service = AIModelService()
        _ai_model_service.load_models()
    return _ai_model_service