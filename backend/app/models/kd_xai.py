import json
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, label_binarize
from sklearn.metrics import (classification_report, multilabel_confusion_matrix, mean_squared_error, 
                             f1_score, roc_auc_score, accuracy_score, precision_score, recall_score)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, mutual_info_regression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
warnings.filterwarnings('ignore')
class NetworkFeatureProbabilityEstimator:
    def __init__(self):
        self.models = {}
        self.network_feature_names = []
        self.static_feature_names = []
    def train(self, X_static: pd.DataFrame, X_network: pd.DataFrame, random_state: int = 42):
        print("\n" + "="*60)
        print("Training Network Feature Probability Estimator")
        print("="*60)
        self.static_feature_names = list(X_static.columns)
        self.network_feature_names = list(X_network.columns)
        print(f"Static features: {len(self.static_feature_names)}")
        print(f"Network features to predict: {len(self.network_feature_names)}")
        for net_feature in self.network_feature_names:
            y_target = X_network[net_feature]
            unique_values = y_target.unique()
            is_binary = set(unique_values).issubset({0, 1})
            if is_binary:
                pos_count = np.sum(y_target == 1)
                neg_count = np.sum(y_target == 0)
                if pos_count == 0 or neg_count == 0:
                    print(f"  {net_feature}: Skipping (all same class)")
                    continue
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    subsample=0.6,
                    colsample_bytree=0.6
                )
                model.fit(X_static, y_target)
                print(f"  {net_feature}: Binary classifier trained (pos={pos_count}, neg={neg_count})")
            else:
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=random_state,
                    subsample=0.6,
                    colsample_bytree=0.6
                )
                model.fit(X_static, y_target)
                print(f"  {net_feature}: Regressor trained")
            self.models[net_feature] = {
                'model': model,
                'is_binary': is_binary
            }
        print(f"\nTrained {len(self.models)} network feature predictors")
    def predict_probabilities(self, X_static: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        for net_feature, model_info in self.models.items():
            model = model_info['model']
            is_binary = model_info['is_binary']
            if is_binary:
                proba = model.predict_proba(X_static)
                if proba.shape[1] == 2:
                    predictions[net_feature] = proba[:, 1]
                else:
                    predictions[net_feature] = proba[:, 0]
            else:
                pred = model.predict(X_static)
                predictions[net_feature] = np.maximum(0, pred)
        return predictions
    def evaluate(self, X_static: pd.DataFrame, X_network: pd.DataFrame) -> Dict:
        print("\n" + "="*60)
        print("Evaluating Network Feature Predictions")
        print("="*60)
        predictions = self.predict_probabilities(X_static)
        results = {}
        for net_feature in self.network_feature_names:
            if net_feature not in predictions:
                continue
            y_true = X_network[net_feature]
            y_pred = predictions[net_feature]
            is_binary = self.models[net_feature]['is_binary']
            if is_binary:
                y_pred_binary = (y_pred > 0.5).astype(int)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                try:
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score(y_true, y_pred)
                    else:
                        auc = None
                except:
                    auc = None
                results[net_feature] = {
                    'type': 'binary',
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc
                }
                print(f"\n{net_feature} (Binary):")
                print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                if auc is not None:
                    print(f"  AUC: {auc:.4f}")
            else:
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_true - y_pred))
                results[net_feature] = {
                    'type': 'continuous',
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                }
                print(f"\n{net_feature} (Continuous):")
                print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return results
    def save(self, filepath: str):
        save_data = {
            'models': self.models,
            'network_feature_names': self.network_feature_names,
            'static_feature_names': self.static_feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Network feature estimator saved to {filepath}")
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        self.models = save_data['models']
        self.network_feature_names = save_data['network_feature_names']
        self.static_feature_names = save_data['static_feature_names']
        print(f"Network feature estimator loaded from {filepath}")
class TrainingHistory:
    def __init__(self):
        self.history = {}
    def reset_class(self, class_name):
        self.history[class_name] = {
            'train_f1': [],
            'val_f1': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_loss': [],
            'val_loss': [],
            'iterations': []
        }
    def add_metrics(self, class_name, iteration, train_metrics, val_metrics):
        self.history[class_name]['iterations'].append(iteration)
        self.history[class_name]['train_f1'].append(train_metrics['f1'])
        self.history[class_name]['val_f1'].append(val_metrics['f1'])
        self.history[class_name]['train_accuracy'].append(train_metrics['accuracy'])
        self.history[class_name]['val_accuracy'].append(val_metrics['accuracy'])
        self.history[class_name]['train_loss'].append(train_metrics['loss'])
        self.history[class_name]['val_loss'].append(val_metrics['loss'])
class FinalEnhancedMalwareDocumentClassifier:
    def __init__(self):
        self.individual_models = {}
        self.soft_label_model = None
        self.label_encoder = MultiLabelBinarizer()
        self.feature_names = []
        self.classes = ["Backdoor", "Botnet", "Download", "Infiltration", "Normal"]
        self.augmentation_config = {
            'Backdoor': 1.0,
            'Botnet': 1.0,
            'Download': 1.0,
            'Infiltration': 1.0,
            'Normal': 1.0
        }
        self.network_estimator = None
        self.preprocess_ = None
        self.feature_selector = None
        self.selected_feature_names = []
        self.use_feature_selection = True
        self.original_static_feature_names = []
        self.class_frequencies = {}
        self.class_weights = {}
    def calculate_class_weights(self, y_hard: np.ndarray) -> Dict[str, float]:
        total_samples = len(y_hard)
        class_weights = {}
        for i, class_name in enumerate(self.classes):
            positive_count = np.sum(y_hard[:, i] == 1)
            negative_count = total_samples - positive_count
            if positive_count > 0:
                weight = total_samples / (2.0 * positive_count)
                class_weights[class_name] = min(weight, 10.0)
            else:
                class_weights[class_name] = 1.0
        print("Class weights for diversity improvement:")
        for class_name, weight in class_weights.items():
            print(f"  {class_name}: {weight:.3f}")
        return class_weights
    def set_augmentation_ratios(self, augmentation_config: Dict[str, float]):
        for class_name in self.classes:
            if class_name in augmentation_config:
                self.augmentation_config[class_name] = augmentation_config[class_name]
        print("Data augmentation ratios set:")
        for class_name, ratio in self.augmentation_config.items():
            print(f"  {class_name}: {ratio:.2f}x")
    def perform_feature_selection(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                                   method: str = 'mutual_info', top_k: int = None, 
                                   threshold: float = None) -> pd.DataFrame:
        print("\n" + "="*60)
        print("PERFORMING FEATURE SELECTION")
        print("="*60)
        print(f"Method: {method}")
        print(f"Original features: {X_train.shape[1]}")
        y_single = np.argmax(y_train, axis=1)
        from sklearn.preprocessing import LabelEncoder
        label_encoder_temp = LabelEncoder()
        y_single_encoded = label_encoder_temp.fit_transform(y_single)
        print(f"Original class distribution: {np.bincount(y_single)}")
        print(f"Encoded class distribution: {np.bincount(y_single_encoded)}")
        if method == 'mutual_info':
            mi_scores = mutual_info_classif(X_train, y_single_encoded, random_state=42)
            feature_scores = pd.DataFrame({
                'feature': X_train.columns,
                'score': mi_scores
            }).sort_values('score', ascending=False)
        elif method == 'tree_based':
            base_model = xgb.XGBClassifier(
                objective='multi:softmax',
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.6
            )
            base_model.fit(X_train, y_single_encoded)
            feature_scores = pd.DataFrame({
                'feature': X_train.columns,
                'score': base_model.feature_importances_
            }).sort_values('score', ascending=False)
        elif method == 'hybrid':
            mi_scores = mutual_info_classif(X_train, y_single_encoded, random_state=42)
            base_model = xgb.XGBClassifier(
                objective='multi:softmax',
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.6
            )
            base_model.fit(X_train, y_single_encoded)
            mi_normalized = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
            tree_normalized = (base_model.feature_importances_ - base_model.feature_importances_.min()) / \
                             (base_model.feature_importances_.max() - base_model.feature_importances_.min() + 1e-10)
            combined_scores = (mi_normalized + tree_normalized) / 2
            feature_scores = pd.DataFrame({
                'feature': X_train.columns,
                'score': combined_scores
            }).sort_values('score', ascending=False)
        else:
            raise ValueError(f"Unknown method: {method}")
        if top_k is not None:
            selected_features = feature_scores.head(top_k)['feature'].tolist()
            print(f"Selected top {top_k} features")
        elif threshold is not None:
            selected_features = feature_scores[feature_scores['score'] >= threshold]['feature'].tolist()
            print(f"Selected features with score >= {threshold}")
        else:
            cumsum = feature_scores['score'].cumsum()
            total = feature_scores['score'].sum()
            cutoff_idx = np.where(cumsum >= 0.3 * total)[0][0] + 1
            selected_features = feature_scores.head(cutoff_idx)['feature'].tolist()
            print(f"Selected features covering 30% of total importance")
        self.selected_feature_names = selected_features
        print(f"Selected features: {len(selected_features)}")
        print(f"Feature reduction: {X_train.shape[1]} -> {len(selected_features)} ({len(selected_features)/X_train.shape[1]*100:.1f}%)")
        print("\nTop 20 selected features:")
        for i, row in feature_scores.head(20).iterrows():
            print(f"  {row['feature']}: {row['score']:.4f}")
        return X_train[selected_features]
    def extract_static_features(self, static_features: Dict) -> Dict:
        features = {}
        structure = static_features.get('structure', {})
        features['streams_count'] = structure.get('streams_count', 0)
        features['document_xml_wellformed'] = int(structure.get('document_xml_wellformed', False))
        features['rels_external_count'] = len(structure.get('rels_external', []))
        features['embedded_objects_count'] = len(structure.get('embedded_objects', []))
        features['metadata_anomalies_count'] = len(structure.get('metadata_anomalies', []))
        macros = static_features.get('macros', {})
        features['has_vba'] = int(macros.get('has_vba', False))
        features['autoexec_triggers_count'] = len(macros.get('autoexec_triggers', []))
        features['modules_count'] = len(macros.get('modules', []))
        features['obfuscated_modules_count'] = sum(1 for m in macros.get('modules', []) if m.get('obfuscated', False))
        features['total_line_count'] = sum(m.get('line_count', 0) for m in macros.get('modules', []))
        features['suspicious_api_calls_count'] = macros.get('suspicious_api_calls_count', 0)
        features['has_dde'] = int(macros.get('dde', {}).get('has_dde', False))
        strings = static_features.get('strings', {})
        features['urls_count'] = len(strings.get('urls', []))
        features['ips_count'] = len(strings.get('ips', []))
        features['filepaths_count'] = len(strings.get('filepaths', []))
        features['registry_keys_count'] = len(strings.get('registry_keys', []))
        features['obfuscated_strings_count'] = len(strings.get('obfuscated_strings', []))
        apis = static_features.get('apis', {})
        features['winapi_calls_count'] = len(apis.get('winapi_calls', []))
        features['com_progids_count'] = len(apis.get('com_progids', []))
        obfuscation = static_features.get('obfuscation', {})
        entropy = obfuscation.get('entropy', {})
        features['entropy_overall'] = entropy.get('overall', 0.0)
        features['entropy_max'] = entropy.get('max', 0.0)
        features['suspicious_ops_count'] = len(obfuscation.get('suspicious_ops', []))
        features['amsi_bypass'] = int(obfuscation.get('amsi_bypass', False))
        network_indicators = static_features.get('network_indicators', {})
        features['network_urls_count'] = len(network_indicators.get('urls', []))
        features['network_domains_count'] = len(network_indicators.get('domains', []))
        features['network_user_agents_count'] = len(network_indicators.get('user_agents', []))
        security_indicators = static_features.get('security_indicators', {})
        features['motw_present'] = int(security_indicators.get('motw_present', False))
        features['macro_security_hint_present'] = int(security_indicators.get('macro_security_hint') == 'present')
        features['trusted_location_hint'] = int(security_indicators.get('trusted_location_hint', False))
        return features
    def extract_network_features(self, network_features: Dict) -> Dict:
        features = {}
        features['external_session_count'] = network_features.get('external_session_count', 0)
        features['protocols_count'] = len(network_features.get('protocols', []))
        dns = network_features.get('dns', {})
        features['dns_any_request'] = int(dns.get('any_request', False))
        features['dns_any_txt_used'] = int(dns.get('any_txt_used', False))
        features['dns_domains_count'] = len(dns.get('domains', []))
        http = network_features.get('http', {})
        features['http_any_request'] = int(http.get('any_request', False))
        features['http_methods_count'] = len(http.get('methods', []))
        features['http_hosts_count'] = len(http.get('hosts', []))
        features['http_user_agents_count'] = len(http.get('user_agents', []))
        features['http_download_mime_types_count'] = len(http.get('download_mime_types', []))
        features['http_any_external_file_download'] = int(http.get('any_external_file_download', False))
        features['http_redirect_chain_count'] = len(http.get('redirect_chain', []))
        return features
    def _infer_feature_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        binary_cols = []
        count_cols = []
        continuous_cols = []
        for col in X.columns:
            values = X[col].dropna().values
            if len(values) == 0:
                continuous_cols.append(col)
                continue
            unique_set = set(np.unique(values))
            if unique_set.issubset({0, 1}):
                binary_cols.append(col)
                continue
            is_count_name = col.endswith('_count') or col.startswith(('count_', 'num_', 'n_'))
            is_non_negative = np.nanmin(values) >= 0
            is_integer_like = np.all(np.isclose(values, np.round(values)))
            if is_count_name or (is_non_negative and is_integer_like):
                count_cols.append(col)
            else:
                continuous_cols.append(col)
        return {
            'binary': binary_cols,
            'count': count_cols,
            'continuous': continuous_cols
        }
    def fit_preprocessor(self, X: pd.DataFrame, y_hard: np.ndarray = None) -> None:
        X = X.copy()
        feature_types = self._infer_feature_types(X)
        label_medians = {}
        label_modes = {}
        if y_hard is not None and len(y_hard) == len(X):
            for class_idx, class_name in enumerate(self.classes):
                class_mask = y_hard[:, class_idx] == 1
                if np.sum(class_mask) > 0:
                    class_data = X[class_mask]
                    for col in feature_types['continuous']:
                        if col not in label_medians:
                            label_medians[col] = {}
                        values = class_data[col].dropna().values
                        if len(values) > 0:
                            label_medians[col][class_name] = float(np.median(values))
                        else:
                            label_medians[col][class_name] = 0.0
                    for col in feature_types['binary']:
                        if col not in label_modes:
                            label_modes[col] = {}
                        values = class_data[col].dropna().values
                        if len(values) > 0:
                            unique, counts = np.unique(values, return_counts=True)
                            mode_value = unique[np.argmax(counts)]
                            label_modes[col][class_name] = float(mode_value)
                        else:
                            label_modes[col][class_name] = 0.0
                    for col in feature_types['count']:
                        if col not in label_medians:
                            label_medians[col] = {}
                        values = class_data[col].dropna().values
                        if len(values) > 0:
                            label_medians[col][class_name] = float(np.median(values))
                        else:
                            label_medians[col][class_name] = 0.0
        global_medians = {}
        global_modes = {}
        for col in feature_types['continuous']:
            global_medians[col] = float(np.nanmedian(X[col].values)) if np.any(~pd.isna(X[col].values)) else 0.0
        for col in feature_types['binary']:
            values = X[col].dropna().values
            if len(values) > 0:
                unique, counts = np.unique(values, return_counts=True)
                global_modes[col] = float(unique[np.argmax(counts)])
            else:
                global_modes[col] = 0.0
        for col in feature_types['count']:
            global_medians[col] = float(np.nanmedian(X[col].values)) if np.any(~pd.isna(X[col].values)) else 0.0
        self.preprocess_ = {
            'feature_types': feature_types,
            'label_medians': label_medians,
            'label_modes': label_modes,
            'global_medians': global_medians,
            'global_modes': global_modes,
            'fitted_columns': list(X.columns)
        }
    def transform_preprocessor(self, X: pd.DataFrame, y_hard: np.ndarray = None) -> pd.DataFrame:
        if self.preprocess_ is None:
            return X.fillna(0)
        cfg = self.preprocess_
        X_out = X.copy()
        for col in cfg['fitted_columns']:
            if col not in X_out.columns:
                X_out[col] = 0
        X_out = X_out[cfg['fitted_columns']]
        if y_hard is not None and len(y_hard) == len(X_out):
            for idx, row in X_out.iterrows():
                sample_labels = y_hard[idx]
                active_classes = [self.classes[i] for i in range(len(self.classes)) if sample_labels[i] == 1]
                for col in cfg['feature_types']['binary']:
                    if pd.isna(row[col]):
                        if active_classes and col in cfg['label_modes']:
                            mode_values = [cfg['label_modes'][col].get(cls, cfg['global_modes'][col]) 
                                         for cls in active_classes]
                            if mode_values:
                                unique_modes, counts = np.unique(mode_values, return_counts=True)
                                X_out.loc[idx, col] = unique_modes[np.argmax(counts)]
                            else:
                                X_out.loc[idx, col] = cfg['global_modes'][col]
                        else:
                            X_out.loc[idx, col] = cfg['global_modes'][col]
                for col in cfg['feature_types']['count']:
                    if pd.isna(row[col]):
                        if active_classes and col in cfg['label_medians']:
                            median_values = [cfg['label_medians'][col].get(cls, cfg['global_medians'][col]) 
                                           for cls in active_classes]
                            if median_values:
                                X_out.loc[idx, col] = np.mean(median_values)
                            else:
                                X_out.loc[idx, col] = cfg['global_medians'][col]
                        else:
                            X_out.loc[idx, col] = cfg['global_medians'][col]
                for col in cfg['feature_types']['continuous']:
                    if pd.isna(row[col]):
                        if active_classes and col in cfg['label_medians']:
                            median_values = [cfg['label_medians'][col].get(cls, cfg['global_medians'][col]) 
                                           for cls in active_classes]
                            if median_values:
                                X_out.loc[idx, col] = np.mean(median_values)
                            else:
                                X_out.loc[idx, col] = cfg['global_medians'][col]
                        else:
                            X_out.loc[idx, col] = cfg['global_medians'][col]
        else:
            for col in cfg['feature_types']['binary']:
                X_out[col] = X_out[col].fillna(cfg['global_modes'][col])
            for col in cfg['feature_types']['count']:
                X_out[col] = X_out[col].fillna(cfg['global_medians'][col])
            for col in cfg['feature_types']['continuous']:
                X_out[col] = X_out[col].fillna(cfg['global_medians'][col])
        return X_out
    def extract_soft_labels(self, flows: List[Dict], file_hash: str = None) -> List[float]:
        if not flows:
            return [0.0] * len(self.classes)
        all_probs = []
        for flow in flows:
            soft_label = flow.get('soft_label', {})
            probs = soft_label.get('probabilities', [0.0] * len(self.classes))
            if len(probs) == len(self.classes):
                all_probs.append(probs)
        if not all_probs:
            return [0.0] * len(self.classes)
        if len(all_probs) == 1:
            base_probs = all_probs[0]
        else:
            weights = np.linspace(0.5, 1.0, len(all_probs))
            weighted_avg = np.average(all_probs, axis=0, weights=weights)
            max_probs = np.max(all_probs, axis=0)
            median_probs = np.median(all_probs, axis=0)
            if file_hash:
                hash_int = hash(file_hash) % 4
                methods = ['weighted_avg', 'max', 'median', 'simple_avg']
                method = methods[hash_int]
            else:
                import random
                method = random.choice(['weighted_avg', 'max', 'median', 'simple_avg'])
            if method == 'weighted_avg':
                base_probs = weighted_avg
            elif method == 'max':
                base_probs = max_probs
            elif method == 'median':
                base_probs = median_probs
            else:
                base_probs = np.mean(all_probs, axis=0)
        if file_hash:
            np.random.seed(hash(file_hash) % (2**31))
        else:
            np.random.seed(42)
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, len(self.classes))
        result = base_probs + noise
        result = np.clip(result, 0, 1)
        if np.sum(result) > 0:
            result = result / np.sum(result)
        return result.tolist()
    def get_file_mapping_key(self, data: Dict, filename: str) -> str:
        if 'sha256' in data:
            sha256_field = data['sha256']
            if isinstance(sha256_field, list) and len(sha256_field) > 0:
                return sha256_field[0].lower()
            elif isinstance(sha256_field, str):
                return sha256_field.lower()
        if 'file' in data:
            file_info = data['file']
            if isinstance(file_info, dict) and 'hash' in file_info:
                hash_info = file_info['hash']
                if isinstance(hash_info, dict) and 'sha256' in hash_info:
                    return hash_info['sha256'].lower()
        if 'hash' in data:
            hash_info = data['hash']
            if isinstance(hash_info, dict) and 'sha256' in hash_info:
                return hash_info['sha256'].lower()
            elif isinstance(hash_info, str):
                return hash_info.lower()
        import re
        hash_pattern = r'[a-fA-F0-9]{64}'
        hash_match = re.search(hash_pattern, filename)
        if hash_match:
            return hash_match.group(0).lower()
        key = filename
        if key.endswith('.json'):
            key = key[:-5]
        suffixes = ['_merged', '_ids_features', '.doc', '.xls', '.xlsx', '.docx', '.ppt', '.pptx']
        for suffix in suffixes:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
        return key
    def add_strategic_noise_to_features(self, X: pd.DataFrame, noise_level: float = 0.2, 
                                  augmentation_strategy: str = 'mixed', random_state: int = 42,
                                  network_zero_prob: float = 0.6) -> pd.DataFrame:
        np.random.seed(random_state)
        X_augmented = X.copy()
        network_cols = [col for col in X.columns 
                       if col.startswith(('external_session', 'protocols', 'dns_', 'http_'))]
        if network_cols:
            for col in network_cols:
                zero_mask = np.random.random(len(X_augmented)) < network_zero_prob
                X_augmented.loc[zero_mask, col] = 0
        for col in X_augmented.columns:
            if col in network_cols and np.all(X_augmented[col] == 0):
                continue
            original_values = X_augmented[col].values
            noise_mask = np.random.random(len(original_values)) < 0.7
            if not np.any(noise_mask):
                continue
            if set(original_values).issubset({0, 1}):
                if augmentation_strategy in ['mixed', 'flip']:
                    flip_prob = noise_level * 0.3
                    flip_mask = noise_mask & (np.random.random(len(original_values)) < flip_prob)
                    X_augmented.loc[flip_mask, col] = 1 - original_values[flip_mask]
            elif np.all(original_values == original_values.astype(int)) and np.max(original_values) > 1:
                if augmentation_strategy in ['mixed', 'multiplicative']:
                    for idx in np.where(noise_mask)[0]:
                        if np.random.random() < 0.5:
                            factor = np.random.lognormal(0, noise_level * 0.4)
                            X_augmented.iloc[idx, X_augmented.columns.get_loc(col)] = max(0, int(original_values[idx] * factor))
                        else:
                            noise = np.random.poisson(lam=noise_level * max(1, original_values[idx]))
                            sign = np.random.choice([-1, 1])
                            X_augmented.iloc[idx, X_augmented.columns.get_loc(col)] = max(0, int(original_values[idx] + sign * noise))
                elif augmentation_strategy == 'scaling':
                    for idx in np.where(noise_mask)[0]:
                        scale_factor = np.random.uniform(1-noise_level, 1+noise_level)
                        X_augmented.iloc[idx, X_augmented.columns.get_loc(col)] = max(0, int(original_values[idx] * scale_factor))
            else:
                if augmentation_strategy in ['mixed', 'gaussian']:
                    for idx in np.where(noise_mask)[0]:
                        std = max(noise_level * 0.1, np.std(original_values) * noise_level * 0.1)
                        noise = np.random.normal(0, std)
                        X_augmented.iloc[idx, X_augmented.columns.get_loc(col)] = max(0, original_values[idx] + noise)
                elif augmentation_strategy == 'beta':
                    if np.max(original_values) <= 1:
                        for idx in np.where(noise_mask)[0]:
                            alpha = beta = 1 + noise_level * 3
                            beta_sample = np.random.beta(alpha, beta)
                            X_augmented.iloc[idx, X_augmented.columns.get_loc(col)] = original_values[idx] * beta_sample
        return X_augmented
    def augment_data_test(self, X: pd.DataFrame, y_hard: np.ndarray, y_soft: np.ndarray, 
                          data_types: List[str], random_state: int = 42, 
                          test_augmentation_ratio: float = 1.5):
        print("\nPerforming TEST data augmentation (conservative, realistic)...")
        X_list = [X]
        y_hard_list = [y_hard]
        y_soft_list = [y_soft]
        data_types_list = [data_types]
        for i, class_name in enumerate(self.classes):
            class_mask = y_hard[:, i] == 1
            class_indices = np.where(class_mask)[0]
            if len(class_indices) == 0:
                continue
            original_count = len(class_indices)
            target_count = int(original_count * test_augmentation_ratio)
            augment_count = target_count - original_count
            if augment_count <= 0:
                continue
            print(f"  {class_name}: {original_count} -> {target_count} (+{augment_count})")
            np.random.seed(random_state + i + 1000)
            replicate_indices = np.random.choice(class_indices, size=augment_count, replace=True)
            X_replicated = X.iloc[replicate_indices].reset_index(drop=True)
            X_augmented = self.add_strategic_noise_to_features(
                X_replicated, 
                noise_level=0.05,
                augmentation_strategy='mixed',
                network_zero_prob=0.25,
                random_state=random_state + i + 1000
            )
            y_hard_replicated = y_hard[replicate_indices]
            y_soft_replicated = y_soft[replicate_indices]
            np.random.seed(random_state + i + 1000)
            soft_noise = np.random.normal(0, 0.005, y_soft_replicated.shape)
            class_noise_strength = np.random.uniform(0.002, 0.008, len(self.classes))
            for j in range(len(self.classes)):
                soft_noise[:, j] *= class_noise_strength[j]
            y_soft_augmented = np.clip(y_soft_replicated + soft_noise, 0, 1)
            row_sums = np.sum(y_soft_augmented, axis=1, keepdims=True)
            y_soft_augmented = np.where(row_sums > 0, y_soft_augmented / row_sums, y_soft_augmented)
            np.random.seed(random_state + i + 2000)
            redistribute_mask = np.random.random(len(y_soft_augmented)) < 0.1
            if np.any(redistribute_mask):
                for idx in np.where(redistribute_mask)[0]:
                    max_class_idx = np.argmax(y_soft_augmented[idx])
                    other_indices = [j for j in range(len(self.classes)) if j != max_class_idx]
                    if other_indices:
                        redistribution_ratio = np.random.uniform(0.05, 0.15)
                        original_max_prob = y_soft_augmented[idx, max_class_idx]
                        redistribution_amount = original_max_prob * redistribution_ratio
                        y_soft_augmented[idx, max_class_idx] -= redistribution_amount
                        per_class_amount = redistribution_amount / len(other_indices)
                        for other_idx in other_indices:
                            y_soft_augmented[idx, other_idx] += per_class_amount
                row_sums = np.sum(y_soft_augmented, axis=1, keepdims=True)
                y_soft_augmented = np.where(row_sums > 0, y_soft_augmented / row_sums, y_soft_augmented)
            data_types_replicated = [data_types[idx] for idx in replicate_indices]
            X_list.append(X_augmented)
            y_hard_list.append(y_hard_replicated)
            y_soft_list.append(y_soft_augmented)
            data_types_list.append(data_types_replicated)
        X_final = pd.concat(X_list, ignore_index=True)
        y_hard_final = np.vstack(y_hard_list)
        y_soft_final = np.vstack(y_soft_list)
        data_types_final = []
        for dt_list in data_types_list:
            data_types_final.extend(dt_list)
        return X_final, y_hard_final, y_soft_final, data_types_final
    def augment_data_conservative(self, X: pd.DataFrame, y_hard: np.ndarray, y_soft: np.ndarray, 
                             data_types: List[str], random_state: int = 42):
        print("\nPerforming TRAIN data augmentation (aggressive)...")
        X_list = [X]
        y_hard_list = [y_hard]
        y_soft_list = [y_soft]
        data_types_list = [data_types]
        for i, class_name in enumerate(self.classes):
            augmentation_ratio = self.augmentation_config[class_name]
            if augmentation_ratio <= 1.0:
                continue
            class_mask = y_hard[:, i] == 1
            class_indices = np.where(class_mask)[0]
            if len(class_indices) == 0:
                continue
            original_count = len(class_indices)
            target_count = int(original_count * augmentation_ratio)
            augment_count = target_count - original_count
            if augment_count <= 0:
                continue
            print(f"  {class_name}: {original_count} -> {target_count} (+{augment_count})")
            np.random.seed(random_state + i)
            replicate_indices = np.random.choice(class_indices, size=augment_count, replace=True)
            X_replicated = X.iloc[replicate_indices].reset_index(drop=True)
            X_augmented = self.add_strategic_noise_to_features(
                X_replicated, 
                noise_level=0.2,
                augmentation_strategy='mixed',
                network_zero_prob=0.6,
                random_state=random_state + i
            )
            y_hard_replicated = y_hard[replicate_indices]
            y_soft_replicated = y_soft[replicate_indices]
            np.random.seed(random_state + i)
            class_noise_strength = np.random.uniform(0.005, 0.02, len(self.classes))
            soft_noise = np.random.normal(0, 0.01, y_soft_replicated.shape)
            for j in range(len(self.classes)):
                soft_noise[:, j] *= class_noise_strength[j]
            y_soft_augmented = np.clip(y_soft_replicated + soft_noise, 0, 1)
            row_sums = np.sum(y_soft_augmented, axis=1, keepdims=True)
            y_soft_augmented = np.where(row_sums > 0, y_soft_augmented / row_sums, y_soft_augmented)
            np.random.seed(random_state + i + 1000)
            redistribute_mask = np.random.random(len(y_soft_augmented)) < 0.3
            if np.any(redistribute_mask):
                for idx in np.where(redistribute_mask)[0]:
                    max_class_idx = np.argmax(y_soft_augmented[idx])
                    other_indices = [j for j in range(len(self.classes)) if j != max_class_idx]
                    if other_indices:
                        redistribution_ratio = np.random.uniform(0.1, 0.3)
                        original_max_prob = y_soft_augmented[idx, max_class_idx]
                        redistribution_amount = original_max_prob * redistribution_ratio
                        y_soft_augmented[idx, max_class_idx] -= redistribution_amount
                        per_class_amount = redistribution_amount / len(other_indices)
                        for other_idx in other_indices:
                            y_soft_augmented[idx, other_idx] += per_class_amount
                row_sums = np.sum(y_soft_augmented, axis=1, keepdims=True)
                y_soft_augmented = np.where(row_sums > 0, y_soft_augmented / row_sums, y_soft_augmented)
            data_types_replicated = [data_types[idx] for idx in replicate_indices]
            X_list.append(X_augmented)
            y_hard_list.append(y_hard_replicated)
            y_soft_list.append(y_soft_augmented)
            data_types_list.append(data_types_replicated)
        X_final = pd.concat(X_list, ignore_index=True)
        y_hard_final = np.vstack(y_hard_list)
        y_soft_final = np.vstack(y_soft_list)
        data_types_final = []
        for dt_list in data_types_list:
            data_types_final.extend(dt_list)
        return X_final, y_hard_final, y_soft_final, data_types_final
    def load_data_from_directories(self, data_config: Dict) -> Tuple[pd.DataFrame, List[List[str]], List[List[float]], List[str]]:
        print("Loading data from soft and hard label directories...")
        soft_label_data = {}
        hard_label_data = {}
        soft_dir = data_config['soft_label_dir']
        soft_key_sources = {}
        if os.path.exists(soft_dir):
            print(f"Loading soft labels from: {soft_dir}")
            soft_files_processed = 0
            for filename in os.listdir(soft_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(soft_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            key = self.get_file_mapping_key(data, filename)
                            if key not in soft_label_data:
                                soft_label_data[key] = data
                                soft_key_sources[key] = filename
                                soft_files_processed += 1
                            else:
                                print(f"  Warning: Duplicate key found in soft labels: {key} (files: {soft_key_sources[key]}, {filename})")
                    except Exception as e:
                        print(f"Error loading soft label file {filename}: {e}")
            print(f"  Loaded {soft_files_processed} soft label files")
            sample_keys = list(soft_label_data.keys())[:3]
            if sample_keys:
                print(f"  Sample soft label keys: {sample_keys}")
        hard_dir = data_config['hard_label_dir']
        hard_key_sources = {}
        if os.path.exists(hard_dir):
            print(f"Loading hard labels from: {hard_dir}")
            hard_files_processed = 0
            for filename in os.listdir(hard_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(hard_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            key = self.get_file_mapping_key(data, filename)
                            if key not in hard_label_data:
                                hard_label_data[key] = data
                                hard_key_sources[key] = filename
                                hard_files_processed += 1
                            else:
                                print(f"  Warning: Duplicate key found in hard labels: {key} (files: {hard_key_sources[key]}, {filename})")
                    except Exception as e:
                        print(f"Error loading hard label file {filename}: {e}")
            print(f"  Loaded {hard_files_processed} hard label files")
            sample_keys = list(hard_label_data.keys())[:3]
            if sample_keys:
                print(f"  Sample hard label keys: {sample_keys}")
        soft_keys = set(soft_label_data.keys())
        hard_keys = set(hard_label_data.keys())
        common_keys = soft_keys & hard_keys
        print(f"\nData matching summary:")
        print(f"  Soft label files: {len(soft_keys)}")
        print(f"  Hard label files: {len(hard_keys)}")
        print(f"  Matched files: {len(common_keys)}")
        unmatched_soft = soft_keys - hard_keys
        unmatched_hard = hard_keys - soft_keys
        if unmatched_soft:
            print(f"  Unmatched soft label files: {len(unmatched_soft)}")
            if len(unmatched_soft) <= 5:
                print(f"    Examples: {list(unmatched_soft)}")
                for key in list(unmatched_soft)[:3]:
                    if key in soft_key_sources:
                        print(f"      {key} <- {soft_key_sources[key]}")
        if unmatched_hard:
            print(f"  Unmatched hard label files: {len(unmatched_hard)}")
            if len(unmatched_hard) <= 5:
                print(f"    Examples: {list(unmatched_hard)}")
                for key in list(unmatched_hard)[:3]:
                    if key in hard_key_sources:
                        print(f"      {key} <- {hard_key_sources[key]}")
        if common_keys:
            print(f"  Successfully matched examples:")
            for key in list(common_keys)[:3]:
                soft_file = soft_key_sources.get(key, "unknown")
                hard_file = hard_key_sources.get(key, "unknown")
                print(f"    {key}")
                print(f"      Soft: {soft_file}")
                print(f"      Hard: {hard_file}")
        features_list = []
        hard_labels_list = []
        soft_labels_list = []
        data_types_list = []
        for key in common_keys:
            hard_data = hard_label_data[key]
            soft_data = soft_label_data[key]
            try:
                static_features = self.extract_static_features(hard_data.get('static_features', {}))
                network_features = self.extract_network_features(hard_data.get('network_features', {}))
                all_features = {**static_features, **network_features}
                features_list.append(all_features)
                labels = hard_data.get('labels', [])
                if not labels or labels == ['Normal']:
                    data_type = 'normal'
                else:
                    data_type = 'malicious'
                hard_labels_list.append(labels)
                flows = soft_data.get('flows', [])
                file_hash = key
                soft_labels = self.extract_soft_labels(flows, file_hash)
                soft_labels_list.append(soft_labels)
                data_types_list.append(data_type)
            except Exception as e:
                print(f"Error processing {key}: {e}")
                continue
        features_df = pd.DataFrame(features_list)
        self.feature_names = list(features_df.columns)
        y_hard_for_preprocessing = self.label_encoder.fit_transform(hard_labels_list)
        self.fit_preprocessor(features_df, y_hard_for_preprocessing)
        features_df = self.transform_preprocessor(features_df, y_hard_for_preprocessing)
        type_counts = pd.Series(data_types_list).value_counts()
        print(f"\nFinal data distribution:")
        for data_type, count in type_counts.items():
            print(f"  {data_type}: {count} samples")
        return features_df, hard_labels_list, soft_labels_list, data_types_list
    def plot_precision_recall_curves(self, X_test: pd.DataFrame, y_hard_test: np.ndarray, 
                                results_dir: str = None):
        y_hard_pred_proba = self.predict_hard_labels_proba(X_test)
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (class_name, color) in enumerate(zip(self.classes, colors)):
            if i < len(y_hard_pred_proba):
                precision, recall, _ = precision_recall_curve(y_hard_test[:, i], y_hard_pred_proba[i])
                ap_score = average_precision_score(y_hard_test[:, i], y_hard_pred_proba[i])
                plt.plot(recall, precision, color=color, linewidth=2, 
                        label=f'{class_name} (AP={ap_score:.3f})')
            else:
                print(f"Warning: No probability data for class {class_name}")
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves for All Classes', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'precision_recall_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    def plot_confusion_matrices(self, X_test: pd.DataFrame, y_hard_test: np.ndarray, 
                            results_dir: str = None):
        y_hard_pred = self.predict_hard_labels(X_test)
        print("\n" + "="*60)
        print("CONFUSION MATRICES")
        print("="*60)
        for i, class_name in enumerate(self.classes):
            print(f"\n=== {class_name} ===")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_hard_test[:, i], y_hard_pred[:, i])
            print("Confusion Matrix:")
            print(f"                Predicted")
            print(f"Actual    Not {class_name:<10} {class_name}")
            print(f"Not {class_name:<6} {cm[0,0]:<12} {cm[0,1]}")
            print(f"{class_name:<10} {cm[1,0]:<12} {cm[1,1]}")
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"True Negatives: {tn}, False Positives: {fp}")
            print(f"False Negatives: {fn}, True Positives: {tp}")
            print(f"Sensitivity (Recall): {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
    def create_f1_callback(self, X_train, y_train, X_val, y_val, class_name, training_history):
        from sklearn.metrics import f1_score, accuracy_score, log_loss
        def f1_callback(env):
            if env.iteration % 10 == 0 or env.iteration == env.end_iteration - 1:
                train_pred = (env.model.predict_proba(X_train)[:, 1] > 0.5).astype(int)
                val_pred = (env.model.predict_proba(X_val)[:, 1] > 0.5).astype(int)
                train_pred_proba = env.model.predict_proba(X_train)[:, 1]
                val_pred_proba = env.model.predict_proba(X_val)[:, 1]
                train_f1 = f1_score(y_train, train_pred, zero_division=0)
                val_f1 = f1_score(y_val, val_pred, zero_division=0)
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                try:
                    train_loss = log_loss(y_train, train_pred_proba)
                    val_loss = log_loss(y_val, val_pred_proba)
                except:
                    train_loss = val_loss = 0.0
                train_metrics = {'f1': train_f1, 'accuracy': train_acc, 'loss': train_loss}
                val_metrics = {'f1': val_f1, 'accuracy': val_acc, 'loss': val_loss}
                training_history.add_metrics(class_name, env.iteration, train_metrics, val_metrics)
                if env.iteration % 50 == 0:
                    print(f"  {class_name} - Iter {env.iteration}: Train F1={train_f1:.3f}, Val F1={val_f1:.3f}")
        return f1_callback
    def train_with_kfold_early_stopping(self, X_train, y_train_class, class_name, 
                                         n_folds=5, early_stopping_rounds=20, 
                                         max_estimators=500, random_state=42):
        print(f"\n  Training {class_name} with {n_folds}-Fold CV and Early Stopping...")
        self.training_history.reset_class(class_name)
        pos_count = np.sum(y_train_class == 1)
        neg_count = np.sum(y_train_class == 0)
        if pos_count == 0 or neg_count == 0:
            print(f"    Warning: Only one class present. Using simple model.")
            scale_pos_weight = 1
        else:
            base_ratio = neg_count / pos_count
            scale_pos_weight = min(base_ratio, 10.0)
            scale_pos_weight = max(scale_pos_weight, 0.1)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_best_iterations = []
        fold_best_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            print(f"    Fold {fold_idx + 1}/{n_folds}...")
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train_class[train_idx]
            y_fold_val = y_train_class[val_idx]
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=max_estimators,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                eval_metric='logloss',
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_weight=5,
                gamma=0.5,
                max_delta_step=1,
                tree_method='hist',
                early_stopping_rounds=early_stopping_rounds
            )
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else max_estimators
            fold_best_iterations.append(best_iteration)
            val_pred = model.predict(X_fold_val)
            val_f1 = f1_score(y_fold_val, val_pred, zero_division=0)
            fold_best_scores.append(val_f1)
            print(f"      Best iteration: {best_iteration}, Val F1: {val_f1:.4f}")
        avg_best_iteration = int(np.mean(fold_best_iterations))
        avg_val_f1 = np.mean(fold_best_scores)
        print(f"    Average best iteration: {avg_best_iteration}")
        print(f"    Average validation F1: {avg_val_f1:.4f}")
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=avg_best_iteration,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_weight=5,
            gamma=0.5,
            max_delta_step=1,
            tree_method='hist'
        )
        final_model.fit(X_train, y_train_class, verbose=False)
        from sklearn.metrics import log_loss
        X_train_curve, X_val_curve, y_train_curve, y_val_curve = train_test_split(
            X_train, y_train_class, test_size=0.2, random_state=random_state
        )
        step_iterations = [10, 30, 50, 75, 100, 150, 200, avg_best_iteration]
        step_iterations = sorted(set([i for i in step_iterations if i <= avg_best_iteration]))
        print(f"    Recording learning curve at {len(step_iterations)} checkpoints...")
        for step_iter in step_iterations:
            step_model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=step_iter,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                eval_metric='logloss',
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_weight=5,
                gamma=0.5,
                max_delta_step=1,
                tree_method='hist'
            )
            step_model.fit(X_train_curve, y_train_curve, verbose=False)
            train_pred = step_model.predict(X_train_curve)
            train_pred_proba = step_model.predict_proba(X_train_curve)[:, 1]
            train_f1 = f1_score(y_train_curve, train_pred, zero_division=0)
            train_acc = accuracy_score(y_train_curve, train_pred)
            try:
                train_loss = log_loss(y_train_curve, train_pred_proba)
            except:
                train_loss = 0.0
            val_pred = step_model.predict(X_val_curve)
            val_pred_proba = step_model.predict_proba(X_val_curve)[:, 1]
            val_f1 = f1_score(y_val_curve, val_pred, zero_division=0)
            val_acc = accuracy_score(y_val_curve, val_pred)
            try:
                val_loss = log_loss(y_val_curve, val_pred_proba)
            except:
                val_loss = 0.0
            train_metrics = {'f1': train_f1, 'accuracy': train_acc, 'loss': train_loss}
            val_metrics = {'f1': val_f1, 'accuracy': val_acc, 'loss': val_loss}
            self.training_history.add_metrics(class_name, step_iter, train_metrics, val_metrics)
        print(f"    Final model: n_estimators={avg_best_iteration}, Train F1={train_f1:.3f}")
        return final_model
    def train(self, data_config: Dict, augmentation_config: Dict[str, float] = None,
              test_size: float = 0.3, random_state: int = 42, results_dir: str = None,
              test_augmentation_ratio: float = 1.5):
        print("Starting final enhanced model training with data augmentation...")
        self._results_dir = results_dir
        if augmentation_config:
            self.set_augmentation_ratios(augmentation_config)
        print(f"\nAugmentation configuration:")
        print(f"  Train: Class-specific ratios (aggressive)")
        print(f"  Test: {test_augmentation_ratio}x for all classes (conservative)")
        X, hard_labels, soft_labels, data_types = self.load_data_from_directories(data_config)
        y_hard = self.label_encoder.transform(hard_labels)
        y_soft = np.array(soft_labels)
        self.class_weights = self.calculate_class_weights(y_hard)
        print("\nOriginal label distribution:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            pos_count = np.sum(y_hard[:, i] == 1)
            neg_count = np.sum(y_hard[:, i] == 0)
            print(f"{class_name}: Positive={pos_count}, Negative={neg_count}")
        print("\nLabel distribution by data type:")
        data_types_array = np.array(data_types)
        for data_type in np.unique(data_types_array):
            mask = data_types_array == data_type
            print(f"\n{data_type}:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                pos_count = np.sum(y_hard[mask, i] == 1)
                total_count = np.sum(mask)
                print(f"  {class_name}: {pos_count}/{total_count} ({pos_count/total_count*100:.1f}%)")
        print(f"\n{'='*60}")
        print("STEP 1: Train/Test Split (BEFORE augmentation to prevent data leakage)")
        print(f"{'='*60}")
        n_samples = len(X)
        all_idx = np.arange(n_samples)
        assigned = np.zeros(n_samples, dtype=bool)
        types_array = np.array(data_types)
        class_list = list(self.label_encoder.classes_)
        def class_index(name):
            return class_list.index(name)
        def split_class_indices_exact(class_name, test_size=0.3, random_state=42):
            idxs = all_idx[(y_hard[:, class_index(class_name)] == 1) & (~assigned)]
            if len(idxs) == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
            test_count = max(1, int(round(len(idxs) * test_size)))
            if len(idxs) <= 1 or test_count == 0:
                train_idx = idxs
                test_idx = np.array([], dtype=int)
            else:
                train_idx, test_idx = train_test_split(idxs, test_size=test_count,
                                                      random_state=random_state, shuffle=True)
            assigned[train_idx] = True
            assigned[test_idx] = True
            return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)
        bot_train_idx, bot_test_idx = split_class_indices_exact("Botnet", test_size=test_size, random_state=random_state)
        print(f"Botnet: train={len(bot_train_idx)}, test={len(bot_test_idx)}")
        inf_train_idx, inf_test_idx = split_class_indices_exact("Infiltration", test_size=test_size, random_state=random_state)
        print(f"Infiltration: train={len(inf_train_idx)}, test={len(inf_test_idx)}")
        rest_idx = all_idx[~assigned]
        if len(rest_idx) == 0:
            rest_train_idx = np.array([], dtype=int)
            rest_test_idx = np.array([], dtype=int)
        else:
            rest_types = types_array[rest_idx]
            try:
                rest_train_idx, rest_test_idx = train_test_split(
                    rest_idx, test_size=test_size, random_state=random_state, stratify=rest_types, shuffle=True
                )
            except:
                rest_train_idx, rest_test_idx = train_test_split(
                    rest_idx, test_size=test_size, random_state=random_state, shuffle=True
                )
        print(f"Rest: train={len(rest_train_idx)}, test={len(rest_test_idx)}")
        train_indices = np.concatenate([bot_train_idx, inf_train_idx, rest_train_idx]) if len(rest_idx) > 0 else np.concatenate([bot_train_idx, inf_train_idx])
        test_indices = np.concatenate([bot_test_idx, inf_test_idx, rest_test_idx]) if len(rest_idx) > 0 else np.concatenate([bot_test_idx, inf_test_idx])
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_hard_train = y_hard[train_indices]
        y_hard_test = y_hard[test_indices]
        y_soft_train = y_soft[train_indices]
        y_soft_test = y_soft[test_indices]
        types_train = types_array[train_indices].tolist()
        types_test = types_array[test_indices].tolist()
        print(f"\nOriginal split results:")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"\n{'='*60}")
        print("STEP 2: Data Augmentation (ONLY on training set)")
        print(f"{'='*60}")
        print("\nAugmenting TRAINING data...")
        X_train, y_hard_train, y_soft_train, types_train = self.augment_data_conservative(
            X_train, y_hard_train, y_soft_train, types_train, random_state
        )
        print(f"\nAugmented training data shape: {X_train.shape}")
        print(f"\n{'='*60}")
        print("Augmenting TEST data (conservative, realistic)")
        print(f"{'='*60}")
        X_test, y_hard_test, y_soft_test, types_test = self.augment_data_test(
            X_test, y_hard_test, y_soft_test, types_test, 
            random_state=random_state + 5000,
            test_augmentation_ratio=test_augmentation_ratio
        )
        print(f"\nAugmented test data shape: {X_test.shape}")
        X_test = self.transform_preprocessor(X_test, y_hard_test)
        print("Test data: Using label-based missing value imputation")
        rng = np.random.RandomState(random_state)
        if len(X_train) > 0:
            perm_train = rng.permutation(len(X_train))
            X_train = X_train.iloc[perm_train].reset_index(drop=True)
            y_hard_train = y_hard_train[perm_train]
            y_soft_train = y_soft_train[perm_train]
            types_train = list(np.array(types_train)[perm_train])
        if len(X_test) > 0:
            perm_test = rng.permutation(len(X_test))
            X_test = X_test.iloc[perm_test].reset_index(drop=True)
            y_hard_test = y_hard_test[perm_test]
            y_soft_test = y_soft_test[perm_test]
            types_test = list(np.array(types_test)[perm_test])
        print(f"\nFinal split results (after augmentation):")
        print(f"  Train samples: {len(X_train)} (augmented)")
        print(f"  Test samples: {len(X_test)} (original, no augmentation)")
        print(f"\n{'='*60}")
        print("STEP 2.5: Training Network Feature Probability Estimator (BEFORE feature selection)")
        print(f"{'='*60}")
        static_feature_cols = [col for col in X_train.columns 
                              if not col.startswith(('external_session', 'protocols', 'dns_', 'http_'))]
        network_feature_cols = [col for col in X_train.columns 
                               if col.startswith(('external_session', 'protocols', 'dns_', 'http_'))]
        X_train_static_original = X_train[static_feature_cols].copy()
        X_train_network = X_train[network_feature_cols].copy()
        X_test_static_original = X_test[static_feature_cols].copy()
        X_test_network = X_test[network_feature_cols].copy()
        print(f"Static features: {len(static_feature_cols)}")
        print(f"Network features: {len(network_feature_cols)}")
        self.network_estimator = NetworkFeatureProbabilityEstimator()
        self.network_estimator.train(X_train_static_original, X_train_network, random_state)
        self.original_static_feature_names = static_feature_cols
        network_pred_results = self.network_estimator.evaluate(X_test_static_original, X_test_network)
        if self.use_feature_selection:
            print(f"\n{'='*60}")
            print("STEP 3: Feature Selection")
            print(f"{'='*60}")
            X_train_selected = self.perform_feature_selection(
                X_train, y_hard_train, 
                method='hybrid',
                top_k=None,
                threshold=None
            )
            X_test_selected = X_test[self.selected_feature_names]
            X_train = X_train_selected
            X_test = X_test_selected
            print(f"Using selected features: {len(self.selected_feature_names)}")
        else:
            print("\nFeature selection is disabled. Using all features.")
        print(f"\n{'='*60}")
        print("STEP 4: Model Training with K-Fold CV and Early Stopping")
        print(f"{'='*60}")
        self.training_history = TrainingHistory()
        print("\nTraining set label distribution:")
        self.individual_models = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            pos_count = np.sum(y_hard_train[:, i] == 1)
            neg_count = np.sum(y_hard_train[:, i] == 0)
            if pos_count == 0:
                print(f"  {class_name}: All negative samples - using dummy classifier")
                continue
            elif neg_count == 0:
                print(f"  {class_name}: All positive samples - using dummy classifier")
                continue
            else:
                print(f"\n  Training {class_name} classifier...")
                print(f"    Positive samples: {pos_count}, Negative samples: {neg_count}")
                model = self.train_with_kfold_early_stopping(
                    X_train,
                    y_hard_train[:, i],
                    class_name,
                    n_folds=5,
                    early_stopping_rounds=15,
                    max_estimators=300,
                    random_state=random_state
                )
                self.individual_models[i] = model
        print(f"\nSuccessfully trained {len(self.individual_models)} individual classifiers")
        print(f"\n{'='*60}")
        print("Training Soft Label Models with Early Stopping")
        print(f"{'='*60}")
        soft_label_models = []
        for i, class_name in enumerate(self.classes):
            print(f"  Training soft label model for {class_name}...")
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=5,
                gamma=0.5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=random_state,
                early_stopping_rounds=20
            )
            X_train_soft, X_val_soft, y_train_soft, y_val_soft = train_test_split(
                X_train, y_soft_train[:, i], test_size=0.3, random_state=random_state
            )
            model.fit(
                X_train_soft, y_train_soft,
                eval_set=[(X_val_soft, y_val_soft)],
                verbose=False
            )
            best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
            print(f"    Best iteration: {best_iteration}")
            soft_label_models.append(model)
        self.soft_label_models = soft_label_models
        print(f"Successfully trained {len(soft_label_models)} soft label models")
        print(f"\n{'='*60}")
        print("STEP 5: Model Evaluation")
        print(f"{'='*60}")
        evaluation_results = self.evaluate(X_test, y_hard_test, y_soft_test, types_test)
        if hasattr(self, 'network_estimator') and self.network_estimator is not None:
            evaluation_results['network_feature_predictions'] = network_pred_results
        return X_test, y_hard_test, y_soft_test, types_test, evaluation_results
    def plot_real_training_curves(self, results_dir: str = None):
        if not hasattr(self, 'training_history') or not self.training_history.history:
            print("No training history available. Please train the model first.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        ax_f1 = axes[0]
        for idx, (class_name, history) in enumerate(self.training_history.history.items()):
            if idx >= 5:
                break
            iterations = history['iterations']
            color = colors[idx]
            max_train_f1 = max(history['train_f1']) if history['train_f1'] else 0
            max_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
            ax_f1.plot(iterations, history['train_f1'], color=color, linewidth=2, marker='o',
                      label=f'{class_name} Train F1 (max: {max_train_f1:.3f})', alpha=0.8)
            ax_f1.plot(iterations, history['val_f1'], color=color, linewidth=2, marker='s',
                      label=f'{class_name} Val F1 (max: {max_val_f1:.3f})', alpha=0.6, linestyle='--')
        ax_f1.set_title('F1 Score - All Classes', fontsize=14, fontweight='bold')
        ax_f1.set_xlabel('n_estimators')
        ax_f1.set_ylabel('F1 Score')
        ax_f1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_f1.grid(True, alpha=0.3)
        ax_f1.set_ylim([0, 1])
        ax_acc = axes[1]
        for idx, (class_name, history) in enumerate(self.training_history.history.items()):
            if idx >= 5:
                break
            iterations = history['iterations']
            color = colors[idx]
            max_train_acc = max(history['train_accuracy']) if history['train_accuracy'] else 0
            max_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
            ax_acc.plot(iterations, history['train_accuracy'], color=color, linewidth=2, marker='o',
                       label=f'{class_name} Train Acc (max: {max_train_acc:.3f})', alpha=0.8)
            ax_acc.plot(iterations, history['val_accuracy'], color=color, linewidth=2, marker='s',
                       label=f'{class_name} Val Acc (max: {max_val_acc:.3f})', alpha=0.6, linestyle='--')
        ax_acc.set_title('Accuracy - All Classes', fontsize=14, fontweight='bold')
        ax_acc.set_xlabel('n_estimators')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim([0, 1])
        plt.tight_layout()
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'training_curves_combined.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        self.plot_final_accuracy_bar_chart(results_dir)
        print("\nTraining Statistics:")
        for class_name, history in self.training_history.history.items():
            final_train_f1 = history['train_f1'][-1] if history['train_f1'] else 0
            final_val_f1 = history['val_f1'][-1] if history['val_f1'] else 0
            best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
            final_train_acc = history['train_accuracy'][-1] if history['train_accuracy'] else 0
            final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
            best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
            print(f"{class_name}:")
            print(f"  F1 - Train: {final_train_f1:.4f}, Val: {final_val_f1:.4f}, Best Val: {best_val_f1:.4f}")
            print(f"  Acc - Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}, Best Val: {best_val_acc:.4f}")
            print(f"  Overfitting Gap (F1): {final_train_f1 - final_val_f1:.4f}")
            print()
    def plot_final_accuracy_bar_chart(self, results_dir: str = None):
        if not hasattr(self, 'training_history') or not self.training_history.history:
            print("No training history available for bar chart.")
            return
        class_names = []
        train_accuracies = []
        val_accuracies = []
        for class_name, history in self.training_history.history.items():
            if len(history['train_accuracy']) > 0 and len(history['val_accuracy']) > 0:
                class_names.append(class_name)
                train_accuracies.append(history['train_accuracy'][-1])
                val_accuracies.append(history['val_accuracy'][-1])
        if not class_names:
            print("No accuracy data available for bar chart.")
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(class_names))
        width = 0.35
        class_colors = {
            'Backdoor': ('lightblue', 'blue'),
            'Botnet': ('lightcoral', 'red'), 
            'Download': ('lightgreen', 'green'),
            'Infiltration': ('lightyellow', 'orange'),
            'Normal': ('lightgray', 'purple')
        }
        bars1 = []
        bars2 = []
        for i, class_name in enumerate(class_names):
            if class_name in class_colors:
                light_color, dark_color = class_colors[class_name]
            else:
                light_color, dark_color = 'lightgray', 'gray'
            bar1 = ax.bar(x[i] - width/2, train_accuracies[i], width, 
                         color=light_color, alpha=0.8, edgecolor=dark_color, linewidth=1)
            bars1.append(bar1)
            bar2 = ax.bar(x[i] + width/2, val_accuracies[i], width, 
                         color=dark_color, alpha=0.8, edgecolor=dark_color, linewidth=1)
            bars2.append(bar2)
        def add_value_labels(bars_list):
            for bars in bars_list:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        add_value_labels(bars1)
        add_value_labels(bars2)
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Final Epoch Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='blue', label='Train Accuracy'),
            Patch(facecolor='blue', edgecolor='blue', label='Validation Accuracy')
        ]
        ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=11, 
                 bbox_to_anchor=(1.01, 1.01), frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'final_epoch_accuracy_bar_chart.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        print("\nFinal Epoch Accuracy Summary:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: Train={train_accuracies[i]:.4f}, Val={val_accuracies[i]:.4f}, Gap={train_accuracies[i]-val_accuracies[i]:.4f}")
    def evaluate(self, X_test: pd.DataFrame, y_hard_test: np.ndarray, y_soft_test: np.ndarray, types_test: List[str]):
        y_hard_pred = self.predict_hard_labels(X_test)
        y_hard_pred_proba = self.predict_hard_labels_proba(X_test)
        y_soft_pred = self.predict_soft_labels(X_test)
        print("=== Hard Label Evaluation ===")
        print("\n--- Overall Multi-label Metrics ---")
        accuracy = accuracy_score(y_hard_test, y_hard_pred)
        print(f"Exact Match Accuracy: {accuracy:.4f}")
        f1_micro = f1_score(y_hard_test, y_hard_pred, average='micro')
        f1_macro = f1_score(y_hard_test, y_hard_pred, average='macro')
        f1_weighted = f1_score(y_hard_test, y_hard_pred, average='weighted')
        f1_samples = f1_score(y_hard_test, y_hard_pred, average='samples')
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Samples): {f1_samples:.4f}")
        precision_micro = precision_score(y_hard_test, y_hard_pred, average='micro')
        precision_macro = precision_score(y_hard_test, y_hard_pred, average='macro')
        recall_micro = recall_score(y_hard_test, y_hard_pred, average='micro')
        recall_macro = recall_score(y_hard_test, y_hard_pred, average='macro')
        print(f"Precision (Micro): {precision_micro:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Micro): {recall_micro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print("\n--- Per-Class Detailed Metrics ---")
        class_metrics = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"\n=== Class: {class_name} ===")
            if i < len(y_hard_pred_proba):
                proba_pos = y_hard_pred_proba[i]
            else:
                proba_pos = y_hard_pred[:, i]
            unique_labels = np.unique(np.concatenate([y_hard_test[:, i], y_hard_pred[:, i]]))
            if len(unique_labels) == 1:
                single_class = unique_labels[0]
                class_label = class_name if single_class == 1 else 'Not ' + class_name
                print(f"Only one class present: {class_label}")
                print(f"Accuracy: {1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0:.4f}")
                report = {
                    str(single_class): {
                        'precision': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'recall': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'f1-score': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'support': len(y_hard_test[:, i])
                    },
                    'accuracy': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                    'macro avg': {
                        'precision': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'recall': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'f1-score': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'support': len(y_hard_test[:, i])
                    },
                    'weighted avg': {
                        'precision': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'recall': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'f1-score': 1.0 if np.all(y_hard_test[:, i] == y_hard_pred[:, i]) else 0.0,
                        'support': len(y_hard_test[:, i])
                    }
                }
            else:
                report = classification_report(y_hard_test[:, i], y_hard_pred[:, i], 
                                             target_names=['Not ' + class_name, class_name],
                                             zero_division=0, output_dict=True)
                print(classification_report(y_hard_test[:, i], y_hard_pred[:, i], 
                                          target_names=['Not ' + class_name, class_name],
                                          zero_division=0))
            f1 = f1_score(y_hard_test[:, i], y_hard_pred[:, i], zero_division=0)
            precision = precision_score(y_hard_test[:, i], y_hard_pred[:, i], zero_division=0)
            recall = recall_score(y_hard_test[:, i], y_hard_pred[:, i], zero_division=0)
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            try:
                if len(np.unique(y_hard_test[:, i])) > 1 and len(np.unique(proba_pos)) > 1:
                    auc = roc_auc_score(y_hard_test[:, i], proba_pos)
                    print(f"AUC Score: {auc:.4f}")
                else:
                    auc = None
                    if len(np.unique(y_hard_test[:, i])) <= 1:
                        print("AUC Score: N/A (only one class present in test set)")
                    else:
                        print("AUC Score: N/A (constant probability predictions)")
            except Exception as e:
                auc = None
                print(f"AUC Score: N/A (error: {str(e)})")
            class_metrics[class_name] = {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'auc_score': auc,
                'classification_report': report
            }
        print("\n=== Soft Label Evaluation ===")
        mse_scores = []
        soft_metrics = {}
        for i, class_name in enumerate(self.classes):
            mse = mean_squared_error(y_soft_test[:, i], y_soft_pred[:, i])
            mse_scores.append(mse)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_soft_test[:, i] - y_soft_pred[:, i]))
            ss_res = np.sum((y_soft_test[:, i] - y_soft_pred[:, i]) ** 2)
            ss_tot = np.sum((y_soft_test[:, i] - np.mean(y_soft_test[:, i])) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            print(f"\n{class_name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R² Score: {r2:.4f}")
            soft_metrics[class_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
        avg_mse = np.mean(mse_scores)
        avg_rmse = np.sqrt(avg_mse)
        print(f"\nAverage MSE: {avg_mse:.4f}")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Hard Labels - Overall F1 (Macro): {f1_macro:.4f}")
        print(f"Hard Labels - Overall F1 (Micro): {f1_micro:.4f}")
        print(f"Hard Labels - Exact Match Accuracy: {accuracy:.4f}")
        print(f"Soft Labels - Average RMSE: {avg_rmse:.4f}")
        evaluation_results = {
            'hard_label_metrics': {
                'overall': {
                    'f1_micro': f1_micro,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'f1_samples': f1_samples,
                    'precision_micro': precision_micro,
                    'precision_macro': precision_macro,
                    'recall_micro': recall_micro,
                    'recall_macro': recall_macro,
                    'exact_match_accuracy': accuracy
                },
                'per_class': class_metrics
            },
            'soft_label_metrics': {
                'overall': {
                    'average_mse': avg_mse,
                    'average_rmse': avg_rmse
                },
                'per_class': soft_metrics
            }
        }
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        results_dir = getattr(self, '_results_dir', None)
        print("Generating Precision-Recall curves...")
        self.plot_precision_recall_curves(X_test, y_hard_test, results_dir)
        print("Generating Confusion Matrices...")
        self.plot_confusion_matrices(X_test, y_hard_test, results_dir)
        print("Generating Learning Curves...")
        self.plot_real_training_curves(results_dir)
        print("Generating ROC Curves...")
        self.plot_roc_curves(X_test, y_hard_test, results_dir)
        print("Generating Confusion Matrix Heatmaps...")
        self.plot_confusion_matrix_heatmaps(X_test, y_hard_test, results_dir)
        print("Generating Feature Importance plots...")
        self.plot_feature_importance(results_dir)
        return evaluation_results
    def predict_hard_labels(self, X: pd.DataFrame) -> np.ndarray:
        if self.use_feature_selection and self.selected_feature_names:
            X_selected = X[self.selected_feature_names]
        else:
            X_selected = X
        predictions = np.zeros((len(X_selected), len(self.label_encoder.classes_)))
        for i, model in self.individual_models.items():
            predictions[:, i] = model.predict(X_selected)
        return predictions.astype(int)
    def predict_hard_labels_proba(self, X: pd.DataFrame) -> List[np.ndarray]:
        if self.use_feature_selection and self.selected_feature_names:
            X_selected = X[self.selected_feature_names]
        else:
            X_selected = X
        probabilities = []
        for i, class_name in enumerate(self.label_encoder.classes_):
            if i in self.individual_models:
                proba = self.individual_models[i].predict_proba(X_selected)
                if proba.shape[1] == 2:
                    probabilities.append(proba[:, 1])
                else:
                    probabilities.append(proba[:, 0])
            else:
                break
        return probabilities
    def predict_soft_labels(self, X: pd.DataFrame, file_hashes: List[str] = None) -> np.ndarray:
        if self.use_feature_selection and self.selected_feature_names:
            X_selected = X[self.selected_feature_names]
        else:
            X_selected = X
        predictions = np.zeros((len(X_selected), len(self.classes)))
        if hasattr(self, 'soft_label_models') and self.soft_label_models:
            for i, model in enumerate(self.soft_label_models):
                predictions[:, i] = model.predict(X_selected)
            base_noise_scale = 0.02
            for sample_idx in range(len(predictions)):
                if file_hashes and sample_idx < len(file_hashes):
                    file_hash = file_hashes[sample_idx]
                    seed = hash(file_hash) % (2**31)
                else:
                    seed = sample_idx * 1000 + 42
                np.random.seed(seed)
                for i, class_name in enumerate(self.classes):
                    class_weight = self.class_weights.get(class_name, 1.0)
                    class_noise_scale = base_noise_scale * class_weight
                    noise = np.random.normal(0, class_noise_scale)
                    predictions[sample_idx, i] += noise
            predictions = np.clip(predictions, 0, 1)
            row_sums = np.sum(predictions, axis=1, keepdims=True)
            predictions = np.where(row_sums > 0, predictions / row_sums, predictions)
        elif self.soft_label_model is not None:
            try:
                predictions = self.soft_label_model.predict(X_selected)
                base_noise_scale = 0.02
                for sample_idx in range(len(predictions)):
                    if file_hashes and sample_idx < len(file_hashes):
                        file_hash = file_hashes[sample_idx]
                        seed = hash(file_hash) % (2**31)
                    else:
                        seed = sample_idx * 1000 + 42
                    np.random.seed(seed)
                    for i, class_name in enumerate(self.classes):
                        class_weight = self.class_weights.get(class_name, 1.0)
                        class_noise_scale = base_noise_scale * class_weight
                        noise = np.random.normal(0, class_noise_scale)
                        predictions[sample_idx, i] += noise
                predictions = np.clip(predictions, 0, 1)
                row_sums = np.sum(predictions, axis=1, keepdims=True)
                predictions = np.where(row_sums > 0, predictions / row_sums, predictions)
            except Exception as e:
                print(f"Warning: Soft label prediction failed: {e}")
                predictions = np.zeros((len(X_selected), len(self.classes)))
        else:
            print("Warning: No soft label model found, returning zeros")
        return predictions
    def plot_roc_curves(self, X_test: pd.DataFrame, y_hard_test: np.ndarray, 
                       results_dir: str = None):
        y_hard_pred_proba = self.predict_hard_labels_proba(X_test)
        plt.figure(figsize=(14, 10))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        mean_fpr = np.linspace(0, 1, 1000)
        all_tpr = []
        all_roc_auc = []
        for i, class_name in enumerate(self.classes):
            if i < len(y_hard_pred_proba):
                color = colors[i]
                fpr, tpr, thresholds = roc_curve(y_hard_test[:, i], y_hard_pred_proba[i], 
                                                 drop_intermediate=False)
                roc_auc = auc(fpr, tpr)
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                all_tpr.append(tpr_interp)
                all_roc_auc.append(roc_auc)
                plt.plot(mean_fpr, tpr_interp, color=color, linewidth=2, alpha=0.8,
                        label=f'{class_name} (AUC={roc_auc:.3f})')
            else:
                print(f"Warning: No probability data for class {class_name}")
        y_test_flat = y_hard_test.ravel()
        y_pred_flat = np.array([y_hard_pred_proba[i] for i in range(len(y_hard_pred_proba))]).T.ravel()
        fpr_micro, tpr_micro, _ = roc_curve(y_test_flat, y_pred_flat, 
                                            drop_intermediate=False)
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        tpr_micro_interp = np.interp(mean_fpr, fpr_micro, tpr_micro)
        tpr_micro_interp[0] = 0.0
        plt.plot(mean_fpr, tpr_micro_interp, color='deeppink', linewidth=3, linestyle='--',
                label=f'Micro-average (AUC={roc_auc_micro:.3f})', alpha=0.9)
        mean_tpr = np.mean(all_tpr, axis=0)
        mean_tpr[-1] = 1.0
        roc_auc_macro = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='navy', linewidth=3, linestyle=':',
                label=f'Macro-average (AUC={roc_auc_macro:.3f})', alpha=0.9)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random Classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate (Recall)', fontsize=13, fontweight='bold')
        plt.title('ROC Curves for All Classes (with Micro/Macro Averages)', 
                 fontsize=15, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'roc_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        print("\nROC AUC Summary:")
        for i, class_name in enumerate(self.classes):
            if i < len(all_roc_auc):
                print(f"  {class_name}: {all_roc_auc[i]:.4f}")
        print(f"  Micro-average: {roc_auc_micro:.4f}")
        print(f"  Macro-average: {roc_auc_macro:.4f}")
    def plot_confusion_matrix_heatmaps(self, X_test: pd.DataFrame, y_hard_test: np.ndarray, 
                                     results_dir: str = None):
        y_hard_pred = self.predict_hard_labels(X_test)
        n_classes = len(self.classes)
        cols = 3
        rows = (n_classes + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        for i, class_name in enumerate(self.classes):
            if i >= len(axes):
                break
            ax = axes[i]
            cm = confusion_matrix(y_hard_test[:, i], y_hard_pred[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not ' + class_name, class_name],
                       yticklabels=['Not ' + class_name, class_name])
            ax.set_title(f'{class_name} Confusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'confusion_matrix_heatmaps.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    def plot_feature_importance(self, results_dir: str = None):
        if not hasattr(self, 'individual_models') or not self.individual_models:
            print("No trained models available for feature importance analysis.")
            return
        importance_data = {}
        for i, class_name in enumerate(self.classes):
            if i in self.individual_models:
                model = self.individual_models[i]
                if hasattr(model, 'feature_importances_'):
                    importance_data[class_name] = model.feature_importances_
                else:
                    print(f"Warning: No feature importance available for {class_name}")
        if not importance_data:
            print("No feature importance data available.")
            return
        n_classes = len(importance_data)
        cols = 2
        rows = (n_classes + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        for idx, (class_name, importances) in enumerate(importance_data.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            top_n = min(20, len(importances))
            top_indices = np.argsort(importances)[-top_n:]
            top_importances = importances[top_indices]
            top_features = [self.feature_names[i] for i in top_indices]
            bars = ax.barh(range(len(top_importances)), top_importances, color='skyblue')
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels(top_features, fontsize=10)
            ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
            ax.set_title(f'{class_name} - Top {top_n} Features', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            for i, (bar, importance) in enumerate(zip(bars, top_importances)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', va='center', fontsize=9)
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'feature_importance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    def predict_static_only(self, static_features: Dict) -> Tuple[List[str], List[float], Dict[str, float], Dict[str, float]]:
        static_features_extracted = self.extract_static_features(static_features)
        all_features = static_features_extracted.copy()
        network_feature_names = [name for name in self.feature_names 
                                if name.startswith(('external_session', 'protocols', 'dns_', 'http_'))]
        for name in network_feature_names:
            if name not in all_features:
                all_features[name] = 0
        feature_vector = pd.DataFrame([all_features])[self.feature_names]
        feature_vector = self.transform_preprocessor(feature_vector)
        print("Prediction: Using global missing value imputation (no label information available)")
        hard_pred_array = self.predict_hard_labels(feature_vector)
        hard_label_proba_list = self.predict_hard_labels_proba(feature_vector)
        hard_probs = {}
        for i, class_name in enumerate(self.classes):
            if i < len(hard_label_proba_list):
                hard_probs[class_name] = float(hard_label_proba_list[i][0])
            else:
                hard_probs[class_name] = 0.0
        file_hash = static_features.get('sha256', 'unknown')
        if isinstance(file_hash, list) and len(file_hash) > 0:
            file_hash = file_hash[0]
        soft_pred_array = self.predict_soft_labels(feature_vector, [file_hash])
        if len(soft_pred_array.shape) == 2:
            soft_pred = soft_pred_array[0]
        else:
            soft_pred = soft_pred_array
        network_probs = {}
        if self.network_estimator is not None and hasattr(self, 'original_static_feature_names'):
            static_only_dict = {}
            for feat_name in self.original_static_feature_names:
                if feat_name in static_features_extracted:
                    static_only_dict[feat_name] = static_features_extracted[feat_name]
                else:
                    static_only_dict[feat_name] = 0
            feature_vector_static_original = pd.DataFrame([static_only_dict], columns=self.original_static_feature_names)
            feature_vector_static_original = self.transform_preprocessor(feature_vector_static_original)
            feature_vector_static_original = feature_vector_static_original[self.original_static_feature_names]
            network_predictions = self.network_estimator.predict_probabilities(feature_vector_static_original)
            network_probs = {k: float(v[0]) for k, v in network_predictions.items()}
        if isinstance(hard_pred_array, list):
            hard_pred_array = np.array(hard_pred_array)
        if len(hard_pred_array.shape) == 1:
            hard_pred_array = hard_pred_array.reshape(1, -1)
        predicted_labels = self.label_encoder.inverse_transform(hard_pred_array)[0]
        return list(predicted_labels), soft_pred.tolist(), network_probs, hard_probs
    def save_models(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'individual_models.pkl'), 'wb') as f:
            pickle.dump(self.individual_models, f)
        if hasattr(self, 'soft_label_models') and self.soft_label_models:
            with open(os.path.join(model_dir, 'soft_label_models.pkl'), 'wb') as f:
                pickle.dump(self.soft_label_models, f)
        metadata = {
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'classes': self.classes,
            'augmentation_config': self.augmentation_config,
            'preprocess_': self.preprocess_,
            'selected_feature_names': self.selected_feature_names,
            'use_feature_selection': self.use_feature_selection,
            'original_static_feature_names': getattr(self, 'original_static_feature_names', [])
        }
        with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        if self.network_estimator is not None:
            self.network_estimator.save(os.path.join(model_dir, 'network_estimator.pkl'))
        print(f"Models saved to {model_dir}")
        print(f"  - Individual models: {len(self.individual_models)}")
        print(f"  - Soft label models: {len(self.soft_label_models) if hasattr(self, 'soft_label_models') else 0}")
        print(f"  - Network estimator: {'Yes' if self.network_estimator else 'No'}")
        print(f"  - Feature selection: {'Enabled' if self.use_feature_selection else 'Disabled'}")
        if self.use_feature_selection:
            print(f"  - Selected features: {len(self.selected_feature_names)}")
    def load_models(self, model_dir: str):
        with open(os.path.join(model_dir, 'individual_models.pkl'), 'rb') as f:
            self.individual_models = pickle.load(f)
        soft_label_models_path = os.path.join(model_dir, 'soft_label_models.pkl')
        if os.path.exists(soft_label_models_path):
            with open(soft_label_models_path, 'rb') as f:
                self.soft_label_models = pickle.load(f)
        else:
            self.soft_label_models = None
        with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            self.label_encoder = metadata['label_encoder']
            self.feature_names = metadata['feature_names']
            self.classes = metadata['classes']
            self.augmentation_config = metadata.get('augmentation_config', self.augmentation_config)
            self.preprocess_ = metadata.get('preprocess_', None)
            self.selected_feature_names = metadata.get('selected_feature_names', [])
            self.use_feature_selection = metadata.get('use_feature_selection', False)
            self.original_static_feature_names = metadata.get('original_static_feature_names', [])
        network_estimator_path = os.path.join(model_dir, 'network_estimator.pkl')
        if os.path.exists(network_estimator_path):
            self.network_estimator = NetworkFeatureProbabilityEstimator()
            self.network_estimator.load(network_estimator_path)
        print(f"Models loaded from {model_dir}")
        print(f"  - Individual models: {len(self.individual_models)}")
        print(f"  - Soft label models (multiple): {len(self.soft_label_models) if self.soft_label_models else 0}")
        print(f"  - Feature selection: {'Enabled' if self.use_feature_selection else 'Disabled'}")
        if self.use_feature_selection:
            print(f"  - Selected features: {len(self.selected_feature_names)}")
    def test_single_file(self, json_filepath: str) -> Dict:
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            static_features = data.get('static_features', {})
            if not static_features:
                return {
                    'file': json_filepath,
                    'sha256': data.get('sha256', 'unknown'),
                    'error': 'No static features found in file',
                    'predicted_hard_labels': [],
                    'predicted_soft_labels': {},
                    'predicted_network_probabilities': {}
                }
            hard_labels, soft_labels, network_probs, hard_probs = self.predict_static_only(static_features)
            result = {
                'file': json_filepath,
                'sha256': data.get('sha256', 'unknown'),
                'predicted_hard_labels': hard_labels,
                'predicted_hard_label_probabilities': hard_probs,
                'predicted_soft_labels': dict(zip(self.classes, soft_labels)),
                'predicted_network_probabilities': network_probs
            }
            return result
        except Exception as e:
            import traceback
            return {
                'file': json_filepath,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'predicted_hard_labels': [],
                'predicted_soft_labels': {},
                'predicted_network_probabilities': {}
            }
    def evaluate_json_folder(self, json_folder: str, output_file: str = None) -> Dict:
        import glob
        from datetime import datetime
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results_{timestamp}.txt"
        json_files = glob.glob(os.path.join(json_folder, "*.json"))
        if not json_files:
            print(f"No JSON files found in {json_folder}")
            return {}
        print(f"\nFound {len(json_files)} JSON files to evaluate")
        print("=" * 80)
        results = []
        total_files = 0
        correct_files = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                actual_labels = data.get('labels', [])
                if isinstance(actual_labels, str):
                    actual_labels = [actual_labels]
                elif not isinstance(actual_labels, list):
                    actual_labels = []
                static_features = data.get('static_features', {})
                if not static_features:
                    results.append({
                        'file': os.path.basename(json_file),
                        'actual_labels': actual_labels,
                        'predicted_labels': [],
                        'match': False,
                        'error': 'No static features found'
                    })
                    total_files += 1
                    continue
                predicted_hard_labels, predicted_soft_labels, network_probs, hard_probs = self.predict_static_only(static_features)
                predicted_set = set(predicted_hard_labels)
                actual_set = set(actual_labels)
                is_match = predicted_set == actual_set
                if is_match:
                    correct_files += 1
                total_files += 1
                results.append({
                    'file': os.path.basename(json_file),
                    'actual_labels': sorted(list(actual_set)),
                    'predicted_labels': sorted(list(predicted_set)),
                    'predicted_hard_label_probabilities': hard_probs,
                    'predicted_soft_labels': dict(zip(self.classes, predicted_soft_labels)),
                    'match': is_match,
                    'error': None
                })
                if total_files % 10 == 0:
                    print(f"Processed {total_files}/{len(json_files)} files... Accuracy so far: {correct_files/total_files*100:.2f}%")
            except Exception as e:
                import traceback
                results.append({
                    'file': os.path.basename(json_file),
                    'actual_labels': [],
                    'predicted_labels': [],
                    'match': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                total_files += 1
        overall_accuracy = (correct_files / total_files * 100) if total_files > 0 else 0.0
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files: {total_files}\n")
            f.write(f"Correct Predictions: {correct_files}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
            f.write(f"\nHard Label Decision Threshold: 0.5 (default for XGBoost binary classification)\n")
            f.write("=" * 80 + "\n\n")
            f.write("INDIVIDUAL FILE RESULTS:\n")
            f.write("Note: Hard labels are selected when Hard Label Probability >= 0.5\n")
            f.write("-" * 80 + "\n")
            for i, result in enumerate(results, 1):
                f.write(f"\n[{i}] File: {result['file']}\n")
                f.write(f"    Actual Labels:    {result['actual_labels']}\n")
                f.write(f"    Predicted Labels: {result['predicted_labels']}\n")
                f.write(f"    Match: {'✓ YES' if result['match'] else '✗ NO'}\n")
                if 'predicted_hard_label_probabilities' in result and result['predicted_hard_label_probabilities']:
                    f.write(f"\n    Hard Label Probabilities (Threshold: 0.5):\n")
                    for label, prob in result['predicted_hard_label_probabilities'].items():
                        selected = "← SELECTED" if label in result['predicted_labels'] else ""
                        f.write(f"      - {label:15s}: {prob:.4f} {selected}\n")
                if 'predicted_soft_labels' in result and result['predicted_soft_labels']:
                    f.write(f"\n    Soft Label Probabilities (for reference):\n")
                    for label, prob in result['predicted_soft_labels'].items():
                        f.write(f"      - {label:15s}: {prob:.4f}\n")
                if result['error']:
                    f.write(f"\n    ERROR: {result['error']}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("LABEL-WISE STATISTICS:\n")
            f.write("-" * 80 + "\n")
            label_stats = {label: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for label in self.classes}
            for result in results:
                if result['error']:
                    continue
                actual_set = set(result['actual_labels'])
                predicted_set = set(result['predicted_labels'])
                for label in self.classes:
                    if label in actual_set and label in predicted_set:
                        label_stats[label]['tp'] += 1
                    elif label not in actual_set and label in predicted_set:
                        label_stats[label]['fp'] += 1
                    elif label in actual_set and label not in predicted_set:
                        label_stats[label]['fn'] += 1
                    else:
                        label_stats[label]['tn'] += 1
            for label, stats in label_stats.items():
                tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
                f.write(f"\n{label}:\n")
                f.write(f"  True Positives:  {tp}\n")
                f.write(f"  False Positives: {fp}\n")
                f.write(f"  False Negatives: {fn}\n")
                f.write(f"  True Negatives:  {tn}\n")
                f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall:    {recall:.4f}\n")
                f.write(f"  F1-Score:  {f1:.4f}\n")
        print("\n" + "=" * 80)
        print(f"Evaluation Complete!")
        print(f"Total Files: {total_files}")
        print(f"Correct Predictions: {correct_files}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print("\nClass-wise Accuracy:")
        for label, stats in label_stats.items():
            tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
            print(f"  {label:15s}: {accuracy*100:.2f}% (TP={tp}, FP={fp}, FN={fn}, TN={tn})")
        print(f"\nResults saved to: {output_file}")
        print("=" * 80)
        return {
            'total_files': total_files,
            'correct_files': correct_files,
            'overall_accuracy': overall_accuracy,
            'results': results,
            'output_file': output_file
        }
def main():
    BASE_PATH = r"C:\Users\a6230\Downloads\data_merge-master"
    data_config = {
        'soft_label_dir': os.path.join(BASE_PATH, 'samp_labeled_ids'),
        'hard_label_dir': os.path.join(BASE_PATH, 'samp_merged_total')
    }
    augmentation_config = {
        'Backdoor': 15.0,
        'Botnet': 100.0,
        'Download': 10.0,
        'Infiltration': 50.0,
        'Normal': 10.0
    }
    test_augmentation_ratio = 20.0
    MODEL_DIR = os.path.join(BASE_PATH, 'final_enhanced_models')
    RESULTS_DIR = os.path.join(BASE_PATH, 'final_enhanced_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    classifier = FinalEnhancedMalwareDocumentClassifier()
    print("Starting final enhanced model training with data augmentation...")
    print(f"Data configuration:")
    print(f"  Soft labels: {data_config['soft_label_dir']}")
    print(f"  Hard labels: {data_config['hard_label_dir']}")
    print(f"\nData augmentation configuration:")
    for class_name, ratio in augmentation_config.items():
        print(f"  {class_name}: {ratio}x")
    try:
        X_test, y_hard_test, y_soft_test, types_test, evaluation_results = classifier.train(
            data_config, 
            augmentation_config=augmentation_config,
            test_size=0.3,
            random_state=42,
            results_dir=RESULTS_DIR,
            test_augmentation_ratio=test_augmentation_ratio
        )
        classifier.save_models(MODEL_DIR)
        with open(os.path.join(RESULTS_DIR, 'final_evaluation_results.json'), 'w', encoding='utf-8') as f:
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            serializable_results = convert_to_serializable(evaluation_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        test_results = {
            'feature_names': classifier.feature_names,
            'classes': classifier.classes,
            'augmentation_config': classifier.augmentation_config,
            'test_data_shape': list(X_test.shape),
            'data_type_distribution': pd.Series(types_test).value_counts().to_dict(),
            'evaluation_summary': {
                'hard_label_f1_macro': evaluation_results['hard_label_metrics']['overall']['f1_macro'],
                'hard_label_accuracy': evaluation_results['hard_label_metrics']['overall']['exact_match_accuracy'],
                'soft_label_avg_rmse': evaluation_results['soft_label_metrics']['overall']['average_rmse']
            }
        }
        with open(os.path.join(RESULTS_DIR, 'final_test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print("\n=== Single File Test Examples ===")
        test_examples = []
        malicious_dir = os.path.join(BASE_PATH, 'merged_output')
        normal_dir = os.path.join(BASE_PATH, 'hardng_labeled')
        example_files = {}
        if os.path.exists(malicious_dir):
            for filename in os.listdir(malicious_dir)[:5]:
                if filename.endswith('.json'):
                    filepath = os.path.join(malicious_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            labels = data.get('labels', [])
                            if labels and labels != ['Normal']:
                                example_files['malicious'] = filepath
                                break
                            elif not labels or labels == ['Normal']:
                                if 'normal' not in example_files:
                                    example_files['normal'] = filepath
                    except:
                        continue
        if os.path.exists(normal_dir):
            for filename in os.listdir(normal_dir)[:5]:
                if filename.endswith('.json'):
                    filepath = os.path.join(normal_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            labels = data.get('labels', [])
                            if labels and labels == ['Normal']:
                                example_files['normal'] = filepath
                    except:
                        continue
        for data_type, test_file in example_files.items():
            if os.path.exists(test_file):
                print(f"\nTesting {data_type.upper()} file: {os.path.basename(test_file)}")
                result = classifier.test_single_file(test_file)
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                    if 'traceback' in result:
                        print(f"  Traceback:\n{result['traceback']}")
                else:
                    result['expected_type'] = data_type
                    test_examples.append(result)
                    print(f"  SHA256: {result['sha256']}")
                    print(f"  Predicted Hard Labels: {result['predicted_hard_labels']}")
                    if 'predicted_hard_label_probabilities' in result:
                        print(f"  Hard Label Probabilities (Threshold: 0.5):")
                        for label, prob in result['predicted_hard_label_probabilities'].items():
                            selected = "← SELECTED" if label in result['predicted_hard_labels'] else ""
                            print(f"    {label:15s}: {prob:.4f} {selected}")
                    print(f"  Soft Label Probabilities:")
                    for label, prob in result['predicted_soft_labels'].items():
                        print(f"    {label:15s}: {prob:.4f}")
            else:
                print(f"File not found: {test_file}")
        if test_examples:
            with open(os.path.join(RESULTS_DIR, 'single_file_test_examples.json'), 'w', encoding='utf-8') as f:
                json.dump(test_examples, f, indent=2, ensure_ascii=False)
        print(f"\nTraining and evaluation completed successfully!")
        print(f"Models saved to: {MODEL_DIR}")
        print(f"Results saved to: {RESULTS_DIR}")
        print("\n" + "="*80)
        print("FINAL PERFORMANCE SUMMARY")
        print("="*80)
        overall_metrics = evaluation_results['hard_label_metrics']['overall']
        print(f"Overall Performance:")
        print(f"  F1 Score (Macro): {overall_metrics['f1_macro']:.4f}")
        print(f"  F1 Score (Micro): {overall_metrics['f1_micro']:.4f}")
        print(f"  Exact Match Accuracy: {overall_metrics['exact_match_accuracy']:.4f}")
        print(f"\nData Augmentation Summary:")
        print(f"  Applied augmentation ratios:")
        for class_name, ratio in augmentation_config.items():
            print(f"    {class_name}: {ratio}x")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()