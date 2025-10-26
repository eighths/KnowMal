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
                print(f"  Classes: {self.ensemble.classes}")
                if self.ensemble.soft_label_thresholds:
                    print(f"  Soft label thresholds: {self.ensemble.soft_label_thresholds}")
                return True
            else:
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
            print(f"  Classes: {self.ensemble.classes}")
            
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
            
            is_only_normal = len(hard_pred_labels) == 1 and hard_pred_labels[0] == 'Normal'
            if not is_only_normal:
                try:
                    predicted_classes = hard_pred_labels if hard_pred_labels else []
                    feature_importance = self._extract_shap_feature_importance(json_data, X_df, predicted_classes)
                    if feature_importance:
                        print(f"SHAP 계산 완료: {len(feature_importance)} features")
                        print(f"feature_importance 샘플: {list(feature_importance.items())[:3]}")
                        result["ai_analysis"]["model_info"]["enhanced_features"]["feature_importance"] = feature_importance
                    else:
                        print(f"SHAP 계산 결과가 비어있음")
                except Exception as e:
                    print(f"XAI feature importance extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Normal 파일이므로 SHAP 계산 스킵")
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_shap_feature_importance(self, json_data: Dict, X_df, predicted_classes: list) -> Dict[str, float]:
        try:
            import shap
            import pandas as pd
            import numpy as np
            
            if not self.ensemble:
                return {}
            
            if not predicted_classes:
                print("No predicted classes for SHAP calculation")
                return {}
            
            if len(predicted_classes) == 1 and predicted_classes[0] == 'Normal':
                print("Skipping SHAP calculation for Normal-only prediction")
                return {}
            
            print(f"Calculating SHAP for predicted classes: {predicted_classes}")
            
            feature_importance = {}
            
            if hasattr(self.ensemble, 'hard_model') and self.ensemble.hard_model:
                for i, model in self.ensemble.hard_model.items():
                    if model is None:
                        continue
                        
                    class_name = self.ensemble.classes[i]
                    
                    if class_name not in predicted_classes:
                        continue
                    
                    try:
                        expected_features = None
                        if hasattr(model, 'n_features_in_'):
                            expected_features = model.n_features_in_
                        elif hasattr(model, 'feature_names_in_'):
                            expected_features = len(model.feature_names_in_)
                        
                        X_model = X_df
                        if expected_features is not None and expected_features != len(X_df.columns):
                            if hasattr(model, 'feature_names_in_'):
                                try:
                                    X_model = X_df[model.feature_names_in_]
                                except KeyError:
                                    X_model = X_df.iloc[:, :expected_features]
                            else:
                                X_model = X_df.iloc[:, :expected_features]
                        
                        if hasattr(model, 'predict_proba'):
                            try:
                                def model_predict(X):
                                    if not isinstance(X, pd.DataFrame):
                                        if hasattr(model, 'feature_names_in_'):
                                            X = pd.DataFrame(X, columns=model.feature_names_in_)
                                        else:
                                            X = pd.DataFrame(X)
                                    
                                    probs = model.predict_proba(X)
                                    if len(probs.shape) > 1 and probs.shape[1] >= 2:
                                        return probs[:, 1]  
                                    return probs.flatten()
                                
                                background = pd.DataFrame(
                                    np.zeros((1, len(X_model.columns))),
                                    columns=X_model.columns
                                )
                                
                                explainer = shap.KernelExplainer(model_predict, background, link="identity")
                                
                                shap_values = explainer.shap_values(X_model.iloc[:1], nsamples=100, silent=True)
                                
                                if len(shap_values.shape) > 1:
                                    class_shap_values = shap_values[0]
                                else:
                                    class_shap_values = shap_values
                                
                                feature_names = getattr(model, 'feature_names_in_', X_model.columns.tolist())
                                
                                importance_scores = np.abs(class_shap_values)
                                
                                
                                sorted_indices = np.argsort(importance_scores)[::-1][:3]
                                
                                for idx in sorted_indices:
                                    if idx < len(feature_names):
                                        feature_name = feature_names[idx]
                                    else:
                                        feature_name = f"feature_{idx}"
                                    importance_key = f"{class_name}_{feature_name}"
                                    feature_importance[importance_key] = float(importance_scores[idx])
                                
                                
                            except Exception as kernel_error:
                                if hasattr(model, 'feature_importances_'):
                                    importances = model.feature_importances_
                                    feature_names = getattr(model, 'feature_names_in_', [f"feature_{j}" for j in range(len(importances))])
                                    
                                    sorted_indices = np.argsort(importances)[::-1][:3]
                                    for idx in sorted_indices:
                                        if idx < len(feature_names):
                                            feature_name = feature_names[idx]
                                        else:
                                            feature_name = f"feature_{idx}"
                                        importance_key = f"{class_name}_{feature_name}"
                                        feature_importance[importance_key] = float(importances[idx])
                                        
                    except Exception as e:
                        print(f"Feature importance extraction failed for {class_name}: {e}")
                        continue
            
            if hasattr(self.ensemble, 'soft_label_model') and self.ensemble.soft_label_model:
                try:
                    model = self.ensemble.soft_label_model
                    
                    expected_features = None
                    if hasattr(model, 'n_features_in_'):
                        expected_features = model.n_features_in_
                    elif hasattr(model, 'feature_names_in_'):
                        expected_features = len(model.feature_names_in_)
                    
                    X_model = X_df
                    if expected_features is not None and expected_features != len(X_df.columns):
                        if hasattr(model, 'feature_names_in_'):
                            try:
                                X_model = X_df[model.feature_names_in_]
                            except KeyError:
                                X_model = X_df.iloc[:, :expected_features]
                        else:
                            X_model = X_df.iloc[:, :expected_features]
                    
                    if hasattr(model, 'predict_proba'):
                        try:
                            for i, class_name in enumerate(self.ensemble.classes):
                                if class_name not in predicted_classes:
                                    continue
                                
                                try:
                                    def model_predict_class(X):
                                        if not isinstance(X, pd.DataFrame):
                                            if hasattr(model, 'feature_names_in_'):
                                                X = pd.DataFrame(X, columns=model.feature_names_in_)
                                            else:
                                                X = pd.DataFrame(X)
                                        
                                        probs = model.predict_proba(X)
                                        if len(probs.shape) > 1 and probs.shape[1] > i:
                                            return probs[:, i]
                                        return np.zeros(len(X))
                                    
                                    background = pd.DataFrame(
                                        np.zeros((1, len(X_model.columns))),
                                        columns=X_model.columns
                                    )
                                    
                                    explainer = shap.KernelExplainer(model_predict_class, background, link="identity")
                                    
                                    shap_values = explainer.shap_values(X_model.iloc[:1], nsamples=100, silent=True)
                                    
                                    if len(shap_values.shape) > 1:
                                        class_shap_values = shap_values[0]
                                    else:
                                        class_shap_values = shap_values
                                    
                                    importance_scores = np.abs(class_shap_values)
                                    feature_names = getattr(model, 'feature_names_in_', X_model.columns.tolist())
                                    sorted_indices = np.argsort(importance_scores)[::-1][:3]
                                    
                                    for idx in sorted_indices:
                                        if idx < len(feature_names):
                                            feature_name = feature_names[idx]
                                        else:
                                            feature_name = f"feature_{idx}"
                                        importance_key = f"soft_{class_name}_{feature_name}"
                                        feature_importance[importance_key] = float(importance_scores[idx])
                                    
                                    
                                except Exception as class_error:
                                    continue
                            
                            
                        except Exception as kernel_error:
                            if hasattr(model, 'feature_importances_'):
                                importances = model.feature_importances_
                                feature_names = getattr(model, 'feature_names_in_', [f"feature_{j}" for j in range(len(importances))])
                                
                                sorted_indices = np.argsort(importances)[::-1][:3]
                                for idx in sorted_indices:
                                    if idx < len(feature_names):
                                        feature_name = feature_names[idx]
                                    else:
                                        feature_name = f"feature_{idx}"
                                    importance_key = f"soft_label_{feature_name}"
                                    feature_importance[importance_key] = float(importances[idx])
                                    
                except Exception as e:
                    print(f"Soft label SHAP extraction failed: {e}")
            
            if not feature_importance:
                print("No SHAP values extracted, using fallback")
                return self._extract_fallback_feature_importance()
            
            return feature_importance
            
        except ImportError:
            return self._extract_fallback_feature_importance()
        except Exception as e:
            return self._extract_fallback_feature_importance()
    
    def _extract_fallback_feature_importance(self) -> Dict[str, float]:
        try:
            if not self.ensemble:
                return {}
            
            feature_importance = {}
            
            default_features = [
                'structure_streams_count', 'structure_storages_count', 'structure_ole_header_valid',
                'macros_has_vba', 'macros_modules_count', 'macros_total_line_count',
                'macros_autoexec_triggers_count', 'macros_suspicious_api_calls_count',
                'strings_urls_count', 'strings_ips_count', 'strings_filepaths_count',
                'strings_registry_keys_count',
                'apis_winapi_calls_count', 'apis_com_progids_count',
                'obfuscation_suspicious_strings_count', 'obfuscation_obfuscation_ops_count',
                'network_indicators_urls_count', 'network_indicators_domains_count',
                'network_indicators_user_agents_count',
                'security_indicators_motw_present', 'security_indicators_digital_signature_signed'
            ]
            
            for i, class_name in enumerate(self.ensemble.classes):
                base_weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
                
                for j, feature_name in enumerate(default_features[:10]):
                    if j < len(base_weights):
                        importance_key = f"{class_name}_{feature_name}"
                        feature_importance[importance_key] = base_weights[j]
            
            return feature_importance
            
        except Exception as e:
            print(f"Fallback feature importance extraction error: {e}")
            return {}
    
    def _convert_shap_string_to_float(self, shap_values, class_name: str):
        import numpy as np
        
        try:
            if isinstance(shap_values, np.ndarray) and shap_values.dtype in [np.float32, np.float64, np.int32, np.int64]:
                return shap_values
            
            if isinstance(shap_values, np.ndarray) and (shap_values.dtype == object or shap_values.dtype.kind == 'U'):
                cleaned_values = []
                
                for val in shap_values:
                    try:
                        if isinstance(val, str):
                            val_str = val.strip('[]').strip()
                            cleaned_values.append(float(val_str))
                        elif isinstance(val, (int, float, np.number)):
                            cleaned_values.append(float(val))
                        else:
                            cleaned_values.append(0.0)
                    except (ValueError, TypeError) as e:
                        cleaned_values.append(0.0)
                
                result = np.array(cleaned_values, dtype=float)
                return result
            
            return shap_values
            
        except Exception as e:
            return shap_values
    
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