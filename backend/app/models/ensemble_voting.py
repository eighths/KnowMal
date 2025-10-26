import numpy as np
import pandas as pd
import pickle
import json
import os
from typing import Dict, List, Tuple
import sys
from datetime import datetime
from pathlib import Path
import glob

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from kd_xai import FinalEnhancedMalwareDocumentClassifier

class MultiLabelVotingEnsemble:
    def __init__(self, 
                 hard_model_dir: str = None,
                 kd_xai_model_dir: str = None,
                 merged_output_dir: str = None,
                 output_dir: str = None,
                 soft_label_thresholds: Dict[str, float] = None):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        self.hard_model_dir = hard_model_dir or str(project_root / "models")
        self.kd_xai_model_dir = kd_xai_model_dir or str(project_root / "models")
        self.merged_output_dir = merged_output_dir or str(project_root / "data")
        self.output_dir = output_dir or str(project_root / "output")
        self.hard_model = None
        self.kd_xai_model = None
        self.soft_label_model = None
        self.hard_metadata = None
        self.kd_xai_metadata = None
        self.classes = None
        self.soft_label_thresholds = soft_label_thresholds
        self.kd_xai_classifier = FinalEnhancedMalwareDocumentClassifier()
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f"ensemble_log_{timestamp}.txt")
        self.log_buffer = []
    def log(self, message: str, print_to_console: bool = True):
        self.log_buffer.append(message)
        if print_to_console:
            print(message)
    def save_log(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_buffer))
    def save_ensemble(self, filepath: str = None):
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"ensemble_model_{timestamp}.pkl")
        ensemble_data = {
            'hard_model': self.hard_model,
            'kd_xai_model': self.kd_xai_model,
            'soft_label_model': self.soft_label_model,
            'hard_metadata': self.hard_metadata,
            'kd_xai_metadata': self.kd_xai_metadata,
            'classes': self.classes,
            'soft_label_thresholds': self.soft_label_thresholds,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        self.log(f"✓ Ensemble model saved to: {filepath}")
        if self.soft_label_thresholds:
            self.log(f"  Soft label thresholds: {self.soft_label_thresholds}")
        return filepath
    def load_ensemble(self, filepath: str):
        print("")
        print("=" * 100)
        print(f"[MODEL LOAD] Loading from: {filepath}")
        print(f"[MODEL LOAD] Absolute path: {os.path.abspath(filepath)}")
        print(f"[MODEL LOAD] File exists: {os.path.exists(filepath)}")
        if os.path.exists(filepath):
            print(f"[MODEL LOAD] File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
            print(f"[MODEL LOAD] >>> FILENAME: {os.path.basename(filepath)} <<<")
        print("=" * 100)
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        self.hard_model = ensemble_data['hard_model']
        self.kd_xai_model = ensemble_data['kd_xai_model']
        self.soft_label_model = ensemble_data['soft_label_model']
        self.hard_metadata = ensemble_data['hard_metadata']
        self.kd_xai_metadata = ensemble_data['kd_xai_metadata']
        self.classes = ensemble_data['classes']
        self.soft_label_thresholds = ensemble_data.get('soft_label_thresholds', None)
        print(f"✓ Ensemble model loaded from: {os.path.basename(filepath)}")
        print(f"  Classes: {self.classes}")
        if self.soft_label_thresholds:
            print(f"  Soft label thresholds: {self.soft_label_thresholds}")
        if self.hard_model:
            print(f"\n[MODEL INFO] Hard models count: {len(self.hard_model)}")
            for i, (class_idx, model) in enumerate(self.hard_model.items()):
                if model and hasattr(model, 'n_features_in_'):
                    print(f"  >>> {self.classes[class_idx]}: {model.n_features_in_} features <<<")
        if self.soft_label_model and hasattr(self.soft_label_model, 'n_features_in_'):
            print(f"\n[MODEL INFO] >>> Soft label model features: {self.soft_label_model.n_features_in_} <<<")
    def load_models(self):
        print("=" * 60)
        print("Loading Models for Ensemble")
        print("=" * 60)
        print("\n[1] Loading hard_model...")
        self._load_hard_model()
        if self.kd_xai_model_dir:
            print("\n[2] Loading kd_xai model...")
            self._load_kd_xai_model()
            print("\n[2-1] Loading soft_label_models...")
            self._load_soft_label_models()
        else:
            print("\n[2] kd_xai model directory not provided. Using hard_model only.")
        self._verify_classes()
        print("\n" + "=" * 60)
        print("Models loaded successfully!")
        print("=" * 60)
    def _load_hard_model(self):
        try:
            with open(os.path.join(self.hard_model_dir, 'individual_models.pkl'), 'rb') as f:
                self.hard_model = pickle.load(f)
            with open(os.path.join(self.hard_model_dir, 'metadata.pkl'), 'rb') as f:
                self.hard_metadata = pickle.load(f)
            print(f"  ✓ Loaded {len(self.hard_model)} classifiers")
            print(f"  ✓ Classes: {self.hard_metadata['classes']}")
            print(f"  ✓ Features: {len(self.hard_metadata['feature_names'])}")
        except Exception as e:
            print(f"  ✗ Failed to load hard_model: {e}")
            raise
    def _load_kd_xai_model(self):
        try:
            with open(os.path.join(self.kd_xai_model_dir, 'individual_models.pkl'), 'rb') as f:
                kd_xai_individual = pickle.load(f)
            with open(os.path.join(self.kd_xai_model_dir, 'metadata.pkl'), 'rb') as f:
                self.kd_xai_metadata = pickle.load(f)
            self.kd_xai_model = kd_xai_individual
            print(f"  ✓ Loaded {len(self.kd_xai_model)} classifiers")
            print(f"  ✓ Classes: {self.kd_xai_metadata['classes']}")
            print(f"  ✓ Features: {len(self.kd_xai_metadata['feature_names'])}")
        except Exception as e:
            print(f"  ✗ Failed to load kd_xai model: {e}")
            raise
    def _load_soft_label_models(self):
        try:
            soft_label_path = os.path.join(self.kd_xai_model_dir, 'soft_label_models.pkl')
            if os.path.exists(soft_label_path):
                with open(soft_label_path, 'rb') as f:
                    self.soft_label_model = pickle.load(f)
                print(f"  ✓ Loaded soft_label_models.pkl")
                print(f"  ✓ Number of soft label models: {len(self.soft_label_model)}")
            else:
                print(f"  ⚠ soft_label_models.pkl not found at {soft_label_path}")
        except Exception as e:
            print(f"  ⚠ Failed to load soft_label_models: {e}")
    def _verify_classes(self):
        hard_classes = self.hard_metadata['classes']
        if self.kd_xai_model:
            kd_xai_classes = self.kd_xai_metadata['classes']
            if set(hard_classes) != set(kd_xai_classes):
                print(f"\n⚠ Warning: Classes mismatch!")
                print(f"  hard_model: {hard_classes}")
                print(f"  kd_xai: {kd_xai_classes}")
                self.classes = sorted(list(set(hard_classes) & set(kd_xai_classes)))
                print(f"  Using common classes: {self.classes}")
            else:
                self.classes = hard_classes
                print(f"\n✓ Classes match: {self.classes}")
        else:
            self.classes = hard_classes
            print(f"\n✓ Using hard_model classes: {self.classes}")
    def predict_proba_hard_model(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for i, model in self.hard_model.items():
            try:
                if hasattr(model, 'feature_names_in_'):
                    model_features = model.feature_names_in_
                    X_model = pd.DataFrame()
                    missing_features = []
                    for feat in model_features:
                        if feat in X.columns:
                            X_model[feat] = X[feat]
                        else:
                            X_model[feat] = 0
                            missing_features.append(feat)
                    if len(missing_features) > len(model_features) * 0.3:
                        self.log(f"  ⚠ Many features missing for hard_model class {i}: {len(missing_features)}/{len(model_features)}", print_to_console=False)
                    pred_proba = model.predict_proba(X_model)[:, 1]
                else:
                    pred_proba = model.predict_proba(X)[:, 1]
                predictions.append(pred_proba)
            except Exception as e:
                error_msg = str(e)[:200]
                self.log(f"Warning: hard_model prediction failed for class {i}: {error_msg}", print_to_console=False)
                predictions.append(np.zeros(len(X)))
        return np.column_stack(predictions)
    def predict_proba_kd_xai_model(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for i, model in self.kd_xai_model.items():
            try:
                if hasattr(model, 'feature_names_in_'):
                    model_features = model.feature_names_in_
                    X_model = pd.DataFrame()
                    missing_features = []
                    for feat in model_features:
                        if feat in X.columns:
                            X_model[feat] = X[feat]
                        else:
                            X_model[feat] = 0
                            missing_features.append(feat)
                    if len(missing_features) > len(model_features) * 0.3:
                        self.log(f"  ⚠ Many features missing for kd_xai_model class {i}: {len(missing_features)}/{len(model_features)}", print_to_console=False)
                    pred_proba = model.predict_proba(X_model)[:, 1]
                else:
                    pred_proba = model.predict_proba(X)[:, 1]
                predictions.append(pred_proba)
            except Exception as e:
                error_msg = str(e)[:200]
                self.log(f"Warning: kd_xai prediction failed for class {i}: {error_msg}", print_to_console=False)
                predictions.append(np.zeros(len(X)))
        return np.column_stack(predictions)
    def get_soft_label_threshold(self, class_name: str) -> float:
        if self.soft_label_thresholds is None:
            return 0.5
        elif isinstance(self.soft_label_thresholds, dict):
            return self.soft_label_thresholds.get(class_name, 0.5)
        elif isinstance(self.soft_label_thresholds, (int, float)):
            return float(self.soft_label_thresholds)
        else:
            return 0.5
    def apply_soft_label_thresholds(self, soft_proba: np.ndarray) -> np.ndarray:
        predictions = np.zeros_like(soft_proba, dtype=int)
        for i, class_name in enumerate(self.classes):
            threshold = self.get_soft_label_threshold(class_name)
            predictions[:, i] = (soft_proba[:, i] > threshold).astype(int)
        return predictions
    def predict_proba_soft_label_model(self, X: pd.DataFrame, file_hashes: List[str] = None) -> np.ndarray:
        if self.soft_label_model is None:
            raise ValueError("soft_label_model not loaded")
        if X is None or X.empty:
            raise ValueError("Input DataFrame is empty or None")
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                except:
                    X[col] = 0
        predictions = []
        if isinstance(self.soft_label_model, dict):
            for i, model in self.soft_label_model.items():
                try:
                    if hasattr(model, 'feature_names_in_'):
                        model_features = model.feature_names_in_
                        X_model = pd.DataFrame()
                        missing_features = []
                        for feat in model_features:
                            if feat in X.columns:
                                X_model[feat] = X[feat]
                            else:
                                X_model[feat] = 0
                                missing_features.append(feat)
                        if len(missing_features) > len(model_features) * 0.3:
                            self.log(f"  ⚠ Many features missing for class {i}: {len(missing_features)}/{len(model_features)}", print_to_console=False)
                    else:
                        X_model = X
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_model)[:, 1]
                    else:
                        pred = model.predict(X_model)
                    predictions.append(pred)
                except Exception as e:
                    error_msg = str(e)[:200]
                    self.log(f"Warning: soft_label prediction failed for class {i}: {error_msg}", print_to_console=False)
                    predictions.append(np.zeros(len(X)))
        elif isinstance(self.soft_label_model, list):
            for i, model in enumerate(self.soft_label_model):
                try:
                    if hasattr(model, 'feature_names_in_'):
                        model_features = model.feature_names_in_
                        X_model = pd.DataFrame()
                        missing_features = []
                        for feat in model_features:
                            if feat in X.columns:
                                X_model[feat] = X[feat]
                            else:
                                X_model[feat] = 0
                                missing_features.append(feat)
                        if len(missing_features) > len(model_features) * 0.3:
                            self.log(f"  ⚠ Many features missing for model {i}: {len(missing_features)}/{len(model_features)}", print_to_console=False)
                    else:
                        X_model = X
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_model)[:, 1]
                    else:
                        pred = model.predict(X_model)
                    predictions.append(pred)
                except Exception as e:
                    error_msg = str(e)[:200]
                    self.log(f"Warning: soft_label prediction failed for model {i}: {error_msg}", print_to_console=False)
                    predictions.append(np.zeros(len(X)))
        result = np.column_stack(predictions)
        if len(result) > 0:
            base_noise_scale = 0.02
            for sample_idx in range(len(result)):
                if file_hashes and sample_idx < len(file_hashes):
                    file_hash = file_hashes[sample_idx]
                    seed = hash(file_hash) % (2**31)
                else:
                    seed = sample_idx * 1000 + 42
                np.random.seed(seed)
                for i, class_name in enumerate(self.classes):
                    class_noise_scale = base_noise_scale * (1.0 + i * 0.1)
                    noise = np.random.normal(0, class_noise_scale)
                    result[sample_idx, i] += noise
            result = np.clip(result, 0, 1)
            row_sums = np.sum(result, axis=1, keepdims=True)
            result = np.where(row_sums > 0, result / row_sums, result)
        return result
    def predict_proba(self, 
                      X_hard: pd.DataFrame = None, 
                      X_kd_xai: pd.DataFrame = None,
                      voting: str = 'soft',
                      weights: List[float] = None,
                      silent: bool = True) -> np.ndarray:
        if voting not in ['soft', 'hard']:
            raise ValueError("voting must be 'soft' or 'hard'")
        predictions = []
        model_names = []
        if X_hard is not None and self.hard_model is not None:
            hard_proba = self.predict_proba_hard_model(X_hard)
            predictions.append(hard_proba)
            model_names.append('hard_model')
        if X_kd_xai is not None and self.kd_xai_model is not None:
            kd_xai_proba = self.predict_proba_kd_xai_model(X_kd_xai)
            predictions.append(kd_xai_proba)
            model_names.append('kd_xai')
        if len(predictions) == 0:
            raise ValueError("No valid predictions available")
        self.log(f"\nEnsemble using models: {model_names}", print_to_console=not silent)
        if voting == 'soft':
            if weights is None:
                weights = [1.0] * len(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_proba = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_proba += pred * weights[i]
                self.log(f"  {model_names[i]}: weight={weights[i]:.3f}", print_to_console=not silent)
            return ensemble_proba
        else:
            hard_preds = [pred > 0.5 for pred in predictions]
            ensemble_pred = np.mean(hard_preds, axis=0)
            return ensemble_pred
    def predict(self, 
                X_hard: pd.DataFrame = None, 
                X_kd_xai: pd.DataFrame = None,
                voting: str = 'soft',
                weights: List[float] = None,
                threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X_hard, X_kd_xai, voting, weights)
        if voting == 'soft':
            return (proba > threshold).astype(int)
        else:
            return (proba > 0.5).astype(int)
    def evaluate(self, 
                 X_hard: np.ndarray = None,
                 X_kd_xai: pd.DataFrame = None,
                 y_true: np.ndarray = None,
                 voting: str = 'soft',
                 weights: List[float] = None) -> Dict:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            hamming_loss, classification_report
        )
        y_pred = self.predict(X_hard, X_kd_xai, voting, weights)
        y_pred_proba = self.predict_proba(X_hard, X_kd_xai, voting, weights)
        results = {
            'subset_accuracy': accuracy_score(y_true, y_pred),
            'hamming_loss': hamming_loss(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
        class_results = {}
        for i, class_name in enumerate(self.classes):
            class_results[class_name] = {
                'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
            }
        results['class_results'] = class_results
        print("\n" + "=" * 60)
        print("Ensemble Evaluation Results")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Subset Accuracy: {results['subset_accuracy']:.4f}")
        print(f"  Hamming Loss: {results['hamming_loss']:.4f}")
        print(f"  F1 Score (Macro): {results['f1_macro']:.4f}")
        print(f"  F1 Score (Micro): {results['f1_micro']:.4f}")
        print(f"  Precision (Macro): {results['precision_macro']:.4f}")
        print(f"  Recall (Macro): {results['recall_macro']:.4f}")
        print(f"\nPer-Class Metrics:")
        for class_name, metrics in class_results.items():
            print(f"\n  {class_name}:")
            print(f"    F1: {metrics['f1']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
        return results
    def extract_features_from_json(self, json_data: Dict) -> Dict:
        features = {}
        static_features = json_data.get('static_features', {})
        if static_features:
            static_f = self.kd_xai_classifier.extract_static_features(static_features)
            features.update(static_f)
        network_features = json_data.get('network_features', {})
        if network_features:
            network_f = self.kd_xai_classifier.extract_network_features(network_features)
            features.update(network_f)
        return features
    def load_json_file(self, json_path: str) -> Tuple[Dict, List[str]]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = data.get('labels', [])
        return data, labels
    def process_all_json_files(self):
        self.log("=" * 80)
        self.log("Processing All JSON Files from merged_output")
        self.log("=" * 80)
        json_files = glob.glob(os.path.join(self.merged_output_dir, "*.json"))
        self.log(f"\nFound {len(json_files)} JSON files")
        if len(json_files) == 0:
            self.log("No JSON files found!")
            return
        results = []
        comparison_results = []
        print(f"\nProcessing {len(json_files)} files...")
        for i, json_path in enumerate(json_files, 1):
            try:
                filename = os.path.basename(json_path)
                if i % 100 == 0 or i == 1:
                    print(f"Progress: {i}/{len(json_files)} files processed...")
                self.log(f"\n[{i}/{len(json_files)}] Processing: {filename}", print_to_console=False)
                json_data, true_labels = self.load_json_file(json_path)
                features = self.extract_features_from_json(json_data)
                X_df = pd.DataFrame([features])
                try:
                    hard_pred_proba = self.predict_proba(X_hard=X_df, X_kd_xai=X_df, voting='soft')
                    hard_pred = (hard_pred_proba > 0.47).astype(int)[0]
                    hard_pred_labels = [self.classes[j] for j in range(len(self.classes)) if hard_pred[j] == 1]
                except Exception as e:
                    self.log(f"  ⚠ Ensemble prediction failed: {e}", print_to_console=False)
                    hard_pred_proba = None
                    hard_pred_labels = []
                soft_pred_proba = None
                soft_pred_labels = []
                if self.soft_label_model is not None:
                    try:
                        file_hash = json_data.get('sha256', 'unknown')
                        if isinstance(file_hash, list) and len(file_hash) > 0:
                            file_hash = file_hash[0]
                        elif not isinstance(file_hash, str):
                            file_hash = 'unknown'
                        soft_pred_proba = self.predict_proba_soft_label_model(X_df, [file_hash])
                        if soft_pred_proba is not None:
                            soft_pred = self.apply_soft_label_thresholds(soft_pred_proba)[0]
                            soft_pred_labels = [self.classes[j] for j in range(len(self.classes)) if soft_pred[j] == 1]
                    except Exception as e:
                        self.log(f"  ⚠ Soft label prediction failed: {e}", print_to_console=False)
                result = {
                    'filename': filename,
                    'true_labels': true_labels,
                    'ensemble_hard_labels': hard_pred_labels,
                    'ensemble_hard_probabilities': hard_pred_proba[0].tolist() if hard_pred_proba is not None else None,
                    'soft_label_predictions': soft_pred_labels,
                    'soft_label_probabilities': soft_pred_proba[0].tolist() if soft_pred_proba is not None else None,
                }
                results.append(result)
                hard_match = set(true_labels) == set(hard_pred_labels)
                soft_match = set(true_labels) == set(soft_pred_labels) if soft_pred_labels else False
                comparison = {
                    'filename': filename,
                    'true_labels': true_labels,
                    'ensemble_pred': hard_pred_labels,
                    'ensemble_match': hard_match,
                    'soft_label_pred': soft_pred_labels,
                    'soft_label_match': soft_match,
                }
                comparison_results.append(comparison)
                self.log(f"  True: {true_labels}", print_to_console=False)
                self.log(f"  Ensemble: {hard_pred_labels} [{'✓' if hard_match else '✗'}]", print_to_console=False)
                if soft_pred_labels:
                    self.log(f"  Soft: {soft_pred_labels} [{'✓' if soft_match else '✗'}]", print_to_console=False)
            except Exception as e:
                self.log(f"  ✗ Failed to process {filename}: {e}", print_to_console=False)
                continue
        print(f"Completed processing {len(json_files)} files.")
        self._save_results_to_json(results)
        self._save_comparison_to_txt(comparison_results)
        self.save_log()
        ensemble_path = self.save_ensemble()
        self.log("\n" + "=" * 80)
        self.log("Processing Complete!")
        self.log(f"Results saved to: {self.output_dir}")
        self.log(f"Log saved to: {self.log_file}")
        self.log(f"Ensemble model saved to: {ensemble_path}")
        self.log("=" * 80)
    def _save_results_to_json(self, results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"ensemble_predictions_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.log(f"\n✓ Predictions saved to: {output_path}")
    def _save_comparison_to_txt(self, comparison_results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"prediction_comparison_{timestamp}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Ensemble Prediction Comparison with True Labels\n")
            f.write("=" * 80 + "\n\n")
            total = len(comparison_results)
            ensemble_correct = sum(1 for r in comparison_results if r['ensemble_match'])
            soft_label_correct = sum(1 for r in comparison_results if r['soft_label_match'])
            f.write(f"Total files: {total}\n")
            f.write(f"Ensemble correct: {ensemble_correct}/{total} ({ensemble_correct/total*100:.2f}%)\n")
            f.write(f"Soft label correct: {soft_label_correct}/{total} ({soft_label_correct/total*100:.2f}%)\n")
            f.write("\n" + "=" * 80 + "\n\n")
            for i, result in enumerate(comparison_results, 1):
                f.write(f"[{i}] {result['filename']}\n")
                f.write(f"  True labels: {result['true_labels']}\n")
                f.write(f"  Ensemble prediction: {result['ensemble_pred']} ")
                f.write(f"[{'✓ MATCH' if result['ensemble_match'] else '✗ MISMATCH'}]\n")
                if result['soft_label_pred']:
                    f.write(f"  Soft label prediction: {result['soft_label_pred']} ")
                    f.write(f"[{'✓ MATCH' if result['soft_label_match'] else '✗ MISMATCH'}]\n")
                f.write("\n")
        self.log(f"✓ Comparison results saved to: {output_path}")
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Label Voting Ensemble for Malware Classification")
    print("=" * 60)
    soft_thresholds = {
        'Backdoor': 0.03,
        'Botnet': 0.05,
        'Download': 0.31,
        'Infiltration': 0.037,
        'Normal': 0.552
    }
    ensemble = MultiLabelVotingEnsemble(
        soft_label_thresholds=soft_thresholds
    )
    print(f"\nSoft label thresholds configured:")
    if isinstance(soft_thresholds, dict):
        for cls, threshold in soft_thresholds.items():
            print(f"  {cls}: {threshold}")
    elif soft_thresholds is not None:
        print(f"  All classes: {soft_thresholds}")
    else:
        print(f"  Using default (0.5 for all classes)")
    ensemble.load_models()
    print("\n" + "=" * 60)
    print("Processing JSON Files from merged_output")
    print("=" * 60)
    ensemble.process_all_json_files()
    print("\n" + "=" * 60)
    print("Ensemble processing complete!")
    print(f"Results saved to: {ensemble.output_dir}")
    print("\n사용 예시:")
    print("  1. 저장된 앙상블 모델 로드:")
    print("     ensemble.load_ensemble('ensemble_results/ensemble_model_TIMESTAMP.pkl')")
    print("  2. Soft label threshold 변경:")
    print("  3. 새로운 파일 예측:")
    print("     features = ensemble.extract_features_from_json(json_data)")
    print("     X_df = pd.DataFrame([features])")
    print("     ensemble_proba = ensemble.predict_proba(X_hard=X_df, X_kd_xai=X_df)")
    print("     soft_proba = ensemble.predict_proba_soft_label_model(X_df)")
    print("     soft_pred = ensemble.apply_soft_label_thresholds(soft_proba)")
    print("=" * 60)