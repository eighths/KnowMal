import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class EnsembleMalwareClassifier:
    def __init__(self):
        self.ensemble_model = None
        self.feature_names = None
        self.class_names = None
        self.is_loaded = False
        
    def load_model(self, model_path: str) -> bool:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                possible_model_keys = ['model', 'ensemble_model', 'classifier', 'estimator', 'best_model']
                self.ensemble_model = None
                
                for key in possible_model_keys:
                    if key in model_data and model_data[key] is not None:
                        candidate_model = model_data[key]
                        if hasattr(candidate_model, 'predict'):
                            self.ensemble_model = candidate_model
                            break
                
                if self.ensemble_model is None:
                    for key, value in model_data.items():
                        
                        if hasattr(value, 'predict') and value is not None:
                            self.ensemble_model = value
                            break
                        
                        elif isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                if hasattr(nested_value, 'predict') and nested_value is not None:
                                    self.ensemble_model = nested_value
                                    break
                            if self.ensemble_model is not None:
                                break
                        
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if hasattr(item, 'predict') and item is not None:
                                    self.ensemble_model = item
                                    break
                            if self.ensemble_model is not None:
                                break
                
                self.feature_names = model_data.get('feature_names')
                self.class_names = model_data.get('class_names') or model_data.get('classes')
                
                if self.ensemble_model is not None and hasattr(self.ensemble_model, 'feature_names_in_'):
                    actual_feature_names = self.ensemble_model.feature_names_in_
                    self.feature_names = actual_feature_names
                elif self.ensemble_model is not None and hasattr(self.ensemble_model, 'get_booster'):
                    try:
                        booster = self.ensemble_model.get_booster()
                        actual_feature_names = booster.feature_names
                        self.feature_names = actual_feature_names
                    except Exception as e:
                        print(f"Could not extract XGBoost feature names: {e}")
                
                if self.feature_names is None:
                    self.feature_names = self._get_default_feature_names()
                
                if self.class_names is None:
                    self.class_names = ['Normal', 'Backdoor', 'Botnet', 'Download', 'Infiltration']
                    
            elif hasattr(model_data, 'predict'):
                self.ensemble_model = model_data
                self.feature_names = self._get_default_feature_names()
                self.class_names = ['Normal', 'Backdoor', 'Botnet', 'Download', 'Infiltration']
            else:
                return False
            
            if self.ensemble_model is None:
                return False
                
            if not hasattr(self.ensemble_model, 'predict'):
                return False
            
            self.is_loaded = True
            
            if self.feature_names is not None:
                try:
                    feature_count = len(self.feature_names)
                except:
                    print(f"Feature names: {self.feature_names}")
            else:
                print("Feature names: Unknown")
                
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False
    
    def _get_default_feature_names(self) -> List[str]:
        return [
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
    
    def _extract_features_from_report(self, analysis_report: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        report_features = analysis_report.get('features', {})
        
        if self.feature_names is not None:
            # numpy 배열을 리스트로 변환
            if hasattr(self.feature_names, 'tolist'):
                feature_names_list = self.feature_names.tolist()
            else:
                feature_names_list = list(self.feature_names)
            
            # 실제 모델 특징 이름을 사용
            for feature_name in feature_names_list:
                # numpy 문자열을 일반 문자열로 변환
                if hasattr(feature_name, 'item'):
                    feature_name = feature_name.item()
                elif str(type(feature_name)).startswith("<class 'numpy."):
                    feature_name = str(feature_name)
                
                features[feature_name] = self._extract_single_feature(report_features, feature_name)
        else:
            # 기본 특징 이름 사용 (fallback)
            features = self._extract_default_features(report_features)
        
        return features
    
    def _extract_single_feature(self, report_features: Dict[str, Any], feature_name: str) -> float:
        if feature_name == 'obfuscated_strings_count':
            obfuscation = report_features.get('obfuscation', {})
            return float(len(obfuscation.get('suspicious_strings', [])))
        
        elif feature_name in ['entropy_overall', 'entropy_max']:
            obfuscation = report_features.get('obfuscation', {})
            entropy = obfuscation.get('entropy', {})
            if feature_name == 'entropy_overall':
                return float(entropy.get('overall', 0.0))
            elif feature_name == 'entropy_max':
                return float(entropy.get('max', 0.0))
        
        elif feature_name == 'total_line_count':
            macros = report_features.get('macros', {})
            return float(macros.get('total_line_count', 0))
        
        else:
            return 0.0
    
    def _extract_default_features(self, report_features: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        
        structure = report_features.get('structure', {})
        features['structure_streams_count'] = float(structure.get('streams_count', 0))
        features['structure_storages_count'] = float(structure.get('storages_count', 0))
        features['structure_ole_header_valid'] = float(structure.get('ole_header_valid', False))
        
        macros = report_features.get('macros', {})
        features['macros_has_vba'] = float(macros.get('has_vba', False) or macros.get('vba_present', False))
        features['macros_modules_count'] = float(len(macros.get('modules', [])))
        features['macros_total_line_count'] = float(macros.get('total_line_count', 0))
        features['macros_autoexec_triggers_count'] = float(len(macros.get('autoexec_triggers', [])))
        features['macros_suspicious_api_calls_count'] = float(macros.get('suspicious_api_calls_count', 0))
        
        strings = report_features.get('strings', {})
        features['strings_urls_count'] = float(len(strings.get('urls', [])))
        features['strings_ips_count'] = float(len(strings.get('ips', [])))
        features['strings_filepaths_count'] = float(len(strings.get('filepaths', [])))
        features['strings_registry_keys_count'] = float(len(strings.get('registry_keys', [])))
        
        apis = report_features.get('apis', {})
        features['apis_winapi_calls_count'] = float(len(apis.get('winapi_calls', [])))
        features['apis_com_progids_count'] = float(len(apis.get('com_progids', [])))
        
        obfuscation = report_features.get('obfuscation', {})
        features['obfuscation_suspicious_strings_count'] = float(len(obfuscation.get('suspicious_strings', [])))
        features['obfuscation_obfuscation_ops_count'] = float(len(obfuscation.get('obfuscation_ops', [])))
        
        network_indicators = report_features.get('network_indicators', {})
        features['network_indicators_urls_count'] = float(len(network_indicators.get('urls', [])))
        features['network_indicators_domains_count'] = float(len(network_indicators.get('domains', [])))
        features['network_indicators_user_agents_count'] = float(len(network_indicators.get('user_agents', [])))
        
        security_indicators = report_features.get('security_indicators', {})
        features['security_indicators_motw_present'] = float(security_indicators.get('motw_present', False))
        features['security_indicators_digital_signature_signed'] = float(
            security_indicators.get('digital_signature', {}).get('signed', False)
        )
        
        return features
    
    def _prepare_features_for_prediction(self, features: Dict[str, float]) -> pd.DataFrame:
        if self.feature_names is not None:
            # numpy 배열을 리스트로 변환
            if hasattr(self.feature_names, 'tolist'):
                feature_names_list = self.feature_names.tolist()
            else:
                feature_names_list = list(self.feature_names)
            
            feature_values = []
            for feature_name in feature_names_list:
                value = features.get(feature_name, 0.0)
                if value is None:
                    value = 0.0
                feature_values.append(float(value))
            
            return pd.DataFrame([feature_values], columns=feature_names_list)
        else:
            return pd.DataFrame([list(features.values())], columns=list(features.keys()))
    
    def predict(self, analysis_report: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
        if not self.is_loaded:
            return [], {}
        
        try:
            features = self._extract_features_from_report(analysis_report)
            
            X = self._prepare_features_for_prediction(features)
            
            if hasattr(self.ensemble_model, 'predict_proba'):
                probabilities = self.ensemble_model.predict_proba(X)
                
                if len(probabilities.shape) == 2 and probabilities.shape[1] > 1:
                    class_probs = {}
                    for i, class_name in enumerate(self.class_names):
                        if i < probabilities.shape[1]:
                            class_probs[class_name] = float(probabilities[0, i])
                    
                    threshold = 0.3
                    predicted_classes = [cls for cls, prob in class_probs.items() if prob >= threshold]
                    
                    if not predicted_classes:
                        max_idx = np.argmax(probabilities[0])
                        predicted_classes = [self.class_names[max_idx]]
                    
                else:
                    prob = float(probabilities[0, 1] if probabilities.shape[1] > 1 else probabilities[0])
                    predicted_classes = ['Malicious'] if prob >= 0.5 else ['Normal']
                    class_probs = {'Normal': 1.0 - prob, 'Malicious': prob}
                
            else:
                predictions = self.ensemble_model.predict(X)
                
                if isinstance(predictions[0], (list, np.ndarray)):
                    predicted_classes = [self.class_names[i] for i, pred in enumerate(predictions[0]) if pred]
                    class_probs = {cls: 1.0 for cls in predicted_classes}
                else:
                    predicted_classes = [str(predictions[0])]
                    class_probs = {predicted_classes[0]: 1.0}
            
            return predicted_classes, class_probs
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return [], {}
    
    def predict_static_only(self, static_features: Dict[str, Any]) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
        analysis_report = {'features': static_features}
        
        predicted_classes, class_probs = self.predict(analysis_report)
        
        hard_labels = predicted_classes
        soft_labels = class_probs
        network_probs = {}
        
        return hard_labels, soft_labels, network_probs
    
    def get_model_info(self) -> Dict[str, Any]:
        feature_count = 0
        feature_names_preview = []
        
        if self.feature_names is not None:
            try:
                feature_count = len(self.feature_names)
                if hasattr(self.feature_names, 'tolist'):
                    feature_names_list = self.feature_names.tolist()
                else:
                    feature_names_list = list(self.feature_names)
                feature_names_preview = feature_names_list[:10]
            except:
                feature_count = 0
                feature_names_preview = []
        
        return {
            'is_loaded': self.is_loaded,
            'model_type': type(self.ensemble_model).__name__ if self.ensemble_model else None,
            'feature_count': feature_count,
            'class_names': self.class_names,
            'feature_names': feature_names_preview
        }
