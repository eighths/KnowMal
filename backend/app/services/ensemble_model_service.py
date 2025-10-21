import os
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from app.services.enhanced_ensemble_model_service import EnhancedEnsembleModelService

class EnsembleModelService:
    def __init__(self):
        self.enhanced_service = EnhancedEnsembleModelService()
        self.model_loaded = False
        self.models_dir = Path(__file__).parent.parent / "models"
        self.ensemble_model_path = self.models_dir / "ensemble_model.pkl"
    
    def load_models(self) -> bool:
        try:
            success = self.enhanced_service.load_models()
            self.model_loaded = success
            return success
        except Exception as e:
            print(f"Model loading error: {e}")
            self.model_loaded = False
            return False
    
    def predict_malware_type(self, analysis_report: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            return self.enhanced_service.predict_malware_type(analysis_report)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def predict_static_only(self, static_features: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            return self.enhanced_service.predict_static_only(static_features)
        except Exception as e:
            print(f"Static prediction error: {e}")
            return None
    
    def get_model_status(self) -> Dict[str, Any]:
        try:
            enhanced_status = self.enhanced_service.get_model_status()
            enhanced_status.update({
                "model_loaded": self.model_loaded,
                "model_path": str(self.ensemble_model_path),
                "model_exists": self.ensemble_model_path.exists()
            })
            return enhanced_status
        except Exception as e:
            print(f"Status error: {e}")
            return {
                "model_loaded": self.model_loaded,
                "model_path": str(self.ensemble_model_path),
                "model_exists": self.ensemble_model_path.exists(),
                "error": str(e)
            }
    
    def reload_model(self) -> bool:
        try:
            success = self.enhanced_service.reload_models()
            self.model_loaded = success
            return success
        except Exception as e:
            print(f"Reload error: {e}")
            self.model_loaded = False
            return False

_ensemble_model_service = None

def get_ensemble_model_service() -> EnsembleModelService:
    global _ensemble_model_service
    if _ensemble_model_service is None:
        _ensemble_model_service = EnsembleModelService()
        _ensemble_model_service.load_models()
    return _ensemble_model_service

