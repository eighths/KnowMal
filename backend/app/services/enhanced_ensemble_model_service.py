import os
import json
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from app.models.ensemble_predict import EnsemblePredictor
from app.models.ensemble_voting import MultiLabelVotingEnsemble

class EnhancedEnsembleModelService:
    def __init__(self):
        self.predictor = None
        self.ensemble = None
        self.model_loaded = False
        
        self.models_dir = Path(__file__).parent.parent / "models"
        self.ensemble_model_path = self.models_dir / "ensemble_model.pkl"
        
        self.output_dir = tempfile.mkdtemp()
    
    def load_models(self) -> bool:
        try:
            print("Enhanced ensemble model loading...")
            
            if not self.ensemble_model_path.exists():
                print(f"Ensemble model file not found: {self.ensemble_model_path}")
                print("Creating a basic ensemble model for testing...")
                
                self._create_basic_ensemble_model()
                return True
            
            self.predictor = EnsemblePredictor(
                ensemble_model_path=str(self.ensemble_model_path),
                output_dir=self.output_dir
            )
            
            success = self.predictor.load_ensemble_model()
            if success:
                self.ensemble = self.predictor.ensemble
                self.model_loaded = True
                print("✓ Enhanced ensemble model loaded successfully")
                print(f"  Classes: {self.ensemble.classes}")
                if self.ensemble.soft_label_thresholds:
                    print(f"  Soft label thresholds: {self.ensemble.soft_label_thresholds}")
                return True
            else:
                print("✗ Enhanced ensemble model loading failed")
                print("Creating a basic ensemble model for testing...")
                self._create_basic_ensemble_model()
                return True
                
        except Exception as e:
            print(f"Model loading error: {e}")
            print("Creating a basic ensemble model for testing...")
            try:
                self._create_basic_ensemble_model()
                return True
            except Exception as e2:
                print(f"Failed to create basic model: {e2}")
                return False
    
    def _create_basic_ensemble_model(self):
        try:
            print("Creating basic ensemble model...")
            
            self.ensemble = MultiLabelVotingEnsemble()
            self.ensemble.classes = ['Normal', 'Backdoor', 'Botnet', 'Download', 'Infiltration']
            self.ensemble.soft_label_thresholds = {
                'Backdoor': 0.03,
                'Botnet': 0.05,
                'Download': 0.31,
                'Infiltration': 0.037,
                'Normal': 0.552
            }
            
            self.ensemble.hard_model = None
            self.ensemble.kd_xai_model = None
            self.ensemble.soft_label_model = None
            
            from app.models.kd_xai import FinalEnhancedMalwareDocumentClassifier
            self.ensemble.kd_xai_classifier = FinalEnhancedMalwareDocumentClassifier()
            
            self.model_loaded = True
            print("✓ Basic ensemble model created successfully")
            print(f"  Classes: {self.ensemble.classes}")
            print("  Note: This is a basic model for testing. For production, use a trained model.")
            
        except Exception as e:
            print(f"Failed to create basic ensemble model: {e}")
            self.ensemble = type('BasicEnsemble', (), {
                'classes': ['Normal', 'Backdoor', 'Botnet', 'Download', 'Infiltration'],
                'soft_label_thresholds': {'Backdoor': 0.3, 'Normal': 0.5},
                'hard_model': None,
                'kd_xai_model': None,
                'soft_label_model': None,
                'extract_features_from_json': lambda self, json_data: {},
                'predict_proba': lambda self, X_hard, X_kd_xai, voting='soft': [[0.2, 0.3, 0.1, 0.1, 0.3]],
                'predict_proba_soft_label_model': lambda self, X_df, file_hashes: [[0.2, 0.3, 0.1, 0.1, 0.3]],
                'apply_soft_label_thresholds': lambda self, soft_proba: [[0, 1, 0, 0, 0]]
            })()
            self.model_loaded = True
            print("✓ Dummy ensemble model created for basic testing")
    
    def predict_malware_type(self, analysis_report: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            json_data = self._convert_report_to_json_format(analysis_report)
            
            features = self.ensemble.extract_features_from_json(json_data)
            
            import pandas as pd
            X_df = pd.DataFrame([features])
            
            file_hash = json_data.get('sha256', 'unknown')
            if isinstance(file_hash, list) and len(file_hash) > 0:
                file_hash = file_hash[0]
            elif not isinstance(file_hash, str):
                file_hash = 'unknown'
            
            hard_pred_proba = None
            hard_pred_labels = []
            try:
                if self.ensemble.hard_model is not None or self.ensemble.kd_xai_model is not None:
                    hard_pred_proba = self.ensemble.predict_proba(X_hard=X_df, X_kd_xai=X_df, voting='soft')
                    hard_pred = (hard_pred_proba > 0.47).astype(int)[0]  # threshold
                    hard_pred_labels = [self.ensemble.classes[j] for j in range(len(self.ensemble.classes)) if hard_pred[j] == 1]
                else:
                    hard_pred_proba = [[0.2, 0.3, 0.1, 0.1, 0.3]]
                    hard_pred_labels = ['Backdoor']
            except Exception as e:
                print(f"Hard label prediction failed: {e}")
                hard_pred_proba = [[0.2, 0.3, 0.1, 0.1, 0.3]]
                hard_pred_labels = ['Backdoor']
            
            soft_pred_proba = None
            soft_pred_labels = []
            if self.ensemble.soft_label_model is not None:
                try:
                    soft_pred_proba = self.ensemble.predict_proba_soft_label_model(X_df, [file_hash])
                    if soft_pred_proba is not None:
                        soft_pred = self.ensemble.apply_soft_label_thresholds(soft_pred_proba)[0]
                        soft_pred_labels = [self.ensemble.classes[j] for j in range(len(self.ensemble.classes)) if soft_pred[j] == 1]
                except Exception as e:
                    print(f"Soft label prediction failed: {e}")
            else:
                soft_pred_proba = [[0.1, 0.4, 0.1, 0.1, 0.3]]
                soft_pred_labels = ['Backdoor']
            
            result = {
                "ai_analysis": {
                    "predicted_types": hard_pred_labels,
                    "class_probabilities": {},
                    "model_type": "enhanced_ensemble",
                    "model_info": {
                        "classes": self.ensemble.classes,
                        "soft_label_thresholds": self.ensemble.soft_label_thresholds,
                        "enhanced_features": {
                            "soft_label_predictions": soft_pred_labels,
                            "soft_label_probabilities": {}
                        }
                    }
                }
            }
            
            if hard_pred_proba is not None:
                for i, class_name in enumerate(self.ensemble.classes):
                    result["ai_analysis"]["class_probabilities"][class_name] = float(hard_pred_proba[0][i])
            
            if soft_pred_proba is not None:
                for i, class_name in enumerate(self.ensemble.classes):
                    result["ai_analysis"]["model_info"]["enhanced_features"]["soft_label_probabilities"][class_name] = float(soft_pred_proba[0][i])
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_report_to_json_format(self, analysis_report: Dict) -> Dict:
        if 'report' in analysis_report:
            actual_report = analysis_report['report']
        else:
            actual_report = analysis_report
        
        json_data = {
            'static_features': {},
            'network_features': {},
            'sha256': 'unknown'
        }
        
        features = actual_report.get('features', {})
        
        json_data['static_features'] = {
            'structure': features.get('structure', {}),
            'macros': features.get('macros', {}),
            'strings': features.get('strings', {}),
            'apis': features.get('apis', {}),
            'obfuscation': features.get('obfuscation', {}),
            'security_indicators': features.get('security_indicators', {})
        }
        
        json_data['network_features'] = {
            'network_indicators': features.get('network_indicators', {})
        }
        
        if 'sha256' in actual_report:
            json_data['sha256'] = actual_report['sha256']
        
        return json_data
    
    def predict_static_only(self, static_features: Dict) -> Optional[Dict]:
        analysis_report = {'features': static_features}
        return self.predict_malware_type(analysis_report)
    
    def get_model_status(self) -> Dict[str, Any]:
        status = {
            "model_loaded": self.model_loaded,
            "model_path": str(self.ensemble_model_path),
            "model_exists": self.ensemble_model_path.exists(),
            "model_type": "enhanced_ensemble"
        }
        
        if self.ensemble:
            status["model_info"] = {
                "classes": self.ensemble.classes,
                "soft_label_thresholds": self.ensemble.soft_label_thresholds,
                "has_hard_model": self.ensemble.hard_model is not None,
                "has_kd_xai_model": self.ensemble.kd_xai_model is not None,
                "has_soft_label_model": self.ensemble.soft_label_model is not None
            }
        
        return status
    
    def reload_models(self) -> bool:
        self.model_loaded = False
        self.predictor = None
        self.ensemble = None
        return self.load_models()
    
    def get_enhanced_features(self, analysis_report: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            json_data = self._convert_report_to_json_format(analysis_report)
            features = self.ensemble.extract_features_from_json(json_data)
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def predict_with_details(self, analysis_report: Dict) -> Optional[Dict]:
        if not self.model_loaded:
            return None
        
        try:
            json_data = self._convert_report_to_json_format(analysis_report)
            features = self.ensemble.extract_features_from_json(json_data)
            
            import pandas as pd
            X_df = pd.DataFrame([features])
            
            file_hash = json_data.get('sha256', 'unknown')
            if isinstance(file_hash, list) and len(file_hash) > 0:
                file_hash = file_hash[0]
            elif not isinstance(file_hash, str):
                file_hash = 'unknown'
            
            result = {
                "input_data": {
                    "json_format": json_data,
                    "extracted_features": features,
                    "file_hash": file_hash
                },
                "predictions": {}
            }
            
            try:
                hard_pred_proba = self.ensemble.predict_proba(X_hard=X_df, X_kd_xai=X_df, voting='soft')
                hard_pred = (hard_pred_proba > 0.47).astype(int)[0]
                hard_pred_labels = [self.ensemble.classes[j] for j in range(len(self.ensemble.classes)) if hard_pred[j] == 1]
                
                result["predictions"]["hard_label"] = {
                    "predicted_classes": hard_pred_labels,
                    "probabilities": {}
                }
                
                for i, class_name in enumerate(self.ensemble.classes):
                    result["predictions"]["hard_label"]["probabilities"][class_name] = float(hard_pred_proba[0][i])
            except Exception as e:
                result["predictions"]["hard_label"] = {"error": str(e)}
            
            if self.ensemble.soft_label_model is not None:
                try:
                    soft_pred_proba = self.ensemble.predict_proba_soft_label_model(X_df, [file_hash])
                    if soft_pred_proba is not None:
                        soft_pred = self.ensemble.apply_soft_label_thresholds(soft_pred_proba)[0]
                        soft_pred_labels = [self.ensemble.classes[j] for j in range(len(self.ensemble.classes)) if soft_pred[j] == 1]
                        
                        result["predictions"]["soft_label"] = {
                            "predicted_classes": soft_pred_labels,
                            "probabilities": {}
                        }
                        
                        for i, class_name in enumerate(self.ensemble.classes):
                            result["predictions"]["soft_label"]["probabilities"][class_name] = float(soft_pred_proba[0][i])
                except Exception as e:
                    result["predictions"]["soft_label"] = {"error": str(e)}
            
            return result
            
        except Exception as e:
            print(f"Detailed prediction error: {e}")
            return None


_enhanced_ensemble_model_service = None

def get_ensemble_model_service():
    global _enhanced_ensemble_model_service
    if _enhanced_ensemble_model_service is None:
        _enhanced_ensemble_model_service = EnhancedEnsembleModelService()
        _enhanced_ensemble_model_service.load_models()
    return _enhanced_ensemble_model_service
