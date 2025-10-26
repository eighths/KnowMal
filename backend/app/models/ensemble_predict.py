import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime
from pathlib import Path
import glob
import argparse

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from ensemble_voting import MultiLabelVotingEnsemble

class EnsemblePredictor:
    def __init__(self, ensemble_model_path: str, output_dir: str = None):
        self.ensemble_model_path = ensemble_model_path
        self.output_dir = output_dir or "prediction_results"
        self.ensemble = None
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f"prediction_log_{timestamp}.txt")
    def log(self, message: str, print_to_console: bool = True):
        if print_to_console:
            print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    def load_ensemble_model(self):
        try:
            self.log(f"[PREDICTOR] Loading ensemble model from: {self.ensemble_model_path}")
            self.log(f"[PREDICTOR] Absolute path: {os.path.abspath(self.ensemble_model_path)}")
            if not os.path.exists(self.ensemble_model_path):
                raise FileNotFoundError(f"Ensemble model file not found: {self.ensemble_model_path}")
            file_size = os.path.getsize(self.ensemble_model_path) / (1024*1024)
            self.log(f"[PREDICTOR] File size: {file_size:.2f} MB")
            self.ensemble = MultiLabelVotingEnsemble()
            self.ensemble.load_ensemble(self.ensemble_model_path)
            self.log(f"✓ Ensemble model loaded successfully")
            self.log(f"  Classes: {self.ensemble.classes}")
            if self.ensemble.soft_label_thresholds:
                self.log(f"  Soft label thresholds: {self.ensemble.soft_label_thresholds}")
            return True
        except Exception as e:
            self.log(f"✗ Failed to load ensemble model: {e}")
            return False
    def predict_single_file(self, json_file_path: str) -> Dict[str, Any]:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            filename = os.path.basename(json_file_path)
            self.log(f"Predicting: {filename}")
            features = self.ensemble.extract_features_from_json(json_data)
            X_df = pd.DataFrame([features])
            file_hash = json_data.get('sha256', 'unknown')
            if isinstance(file_hash, list) and len(file_hash) > 0:
                file_hash = file_hash[0]
            elif not isinstance(file_hash, str):
                file_hash = 'unknown'
            hard_pred_proba = None
            hard_pred_labels = []
            try:
                hard_pred_proba = self.ensemble.predict_proba(X_hard=X_df, X_kd_xai=X_df, voting='soft')
                hard_pred = (hard_pred_proba > 0.47).astype(int)[0]
                hard_pred_labels = [self.ensemble.classes[j] for j in range(len(self.ensemble.classes)) if hard_pred[j] == 1]
            except Exception as e:
                self.log(f"  ⚠ Hard label prediction failed: {e}", print_to_console=False)
            soft_pred_proba = None
            soft_pred_labels = []
            if self.ensemble.soft_label_model is not None:
                try:
                    soft_pred_proba = self.ensemble.predict_proba_soft_label_model(X_df, [file_hash])
                    if soft_pred_proba is not None:
                        soft_pred = self.ensemble.apply_soft_label_thresholds(soft_pred_proba)[0]
                        soft_pred_labels = [self.ensemble.classes[j] for j in range(len(self.ensemble.classes)) if soft_pred[j] == 1]
                except Exception as e:
                    self.log(f"  ⚠ Soft label prediction failed: {e}", print_to_console=False)
            result = {
                'filename': filename,
                'file_path': json_file_path,
                'sha256': file_hash,
                'prediction_timestamp': datetime.now().isoformat(),
                'hard_label_prediction': {
                    'predicted_labels': hard_pred_labels,
                    'probabilities': {}
                },
                'soft_label_prediction': {
                    'predicted_labels': soft_pred_labels,
                    'probabilities': {}
                }
            }
            if hard_pred_proba is not None:
                for i, class_name in enumerate(self.ensemble.classes):
                    result['hard_label_prediction']['probabilities'][class_name] = float(hard_pred_proba[0][i])
            if soft_pred_proba is not None:
                for i, class_name in enumerate(self.ensemble.classes):
                    result['soft_label_prediction']['probabilities'][class_name] = float(soft_pred_proba[0][i])
            actual_labels = json_data.get('labels', [])
            if actual_labels:
                if isinstance(actual_labels, str):
                    actual_labels = [actual_labels]
                result['actual_labels'] = actual_labels
                hard_accuracy = len(set(hard_pred_labels) & set(actual_labels)) / len(set(hard_pred_labels) | set(actual_labels)) if (hard_pred_labels or actual_labels) else 0
                soft_accuracy = len(set(soft_pred_labels) & set(actual_labels)) / len(set(soft_pred_labels) | set(actual_labels)) if (soft_pred_labels or actual_labels) else 0
                result['accuracy'] = {
                    'hard_label': float(hard_accuracy),
                    'soft_label': float(soft_accuracy)
                }
            self.log(f"  ✓ Prediction completed for {filename}")
            return result
        except Exception as e:
            self.log(f"  ✗ Prediction failed for {os.path.basename(json_file_path)}: {e}")
            return {
                'filename': os.path.basename(json_file_path),
                'file_path': json_file_path,
                'error': str(e),
                'prediction_timestamp': datetime.now().isoformat()
            }
    def predict_directory(self, input_path: str, file_pattern: str = "*.json") -> List[Dict[str, Any]]:
        if os.path.isfile(input_path):
            self.log(f"Predicting single file: {input_path}")
            if not input_path.lower().endswith('.json'):
                self.log(f"Warning: File is not a JSON file: {input_path}")
            result = self.predict_single_file(input_path)
            return [result]
        elif os.path.isdir(input_path):
            self.log(f"Predicting all files in directory: {input_path}")
            json_files = glob.glob(os.path.join(input_path, file_pattern))
            if not json_files:
                self.log(f"No JSON files found in {input_path}")
                return []
            self.log(f"Found {len(json_files)} JSON files")
            results = []
            for i, json_file in enumerate(json_files, 1):
                self.log(f"Processing {i}/{len(json_files)}: {os.path.basename(json_file)}")
                result = self.predict_single_file(json_file)
                results.append(result)
            return results
        else:
            self.log(f"Error: Path does not exist: {input_path}")
            return []
    def save_results(self, results: List[Dict[str, Any]], output_filename: str = None):
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ensemble_predictions_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_filename)
        summary = {
            'prediction_info': {
                'timestamp': datetime.now().isoformat(),
                'total_files': len(results),
                'successful_predictions': len([r for r in results if 'error' not in r]),
                'failed_predictions': len([r for r in results if 'error' in r]),
                'ensemble_model_path': self.ensemble_model_path
            },
            'results': results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.log(f"✓ Results saved to: {output_path}")
        self.print_summary(summary)
        return output_path
    def print_summary(self, summary: Dict[str, Any]):
        info = summary['prediction_info']
        results = summary['results']
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total files processed: {info['total_files']}")
        print(f"Successful predictions: {info['successful_predictions']}")
        print(f"Failed predictions: {info['failed_predictions']}")
        if info['successful_predictions'] > 0:
            hard_label_counts = {}
            soft_label_counts = {}
            accuracy_scores = []
            for result in results:
                if 'error' not in result:
                    for label in result.get('hard_label_prediction', {}).get('predicted_labels', []):
                        hard_label_counts[label] = hard_label_counts.get(label, 0) + 1
                    for label in result.get('soft_label_prediction', {}).get('predicted_labels', []):
                        soft_label_counts[label] = soft_label_counts.get(label, 0) + 1
                    if 'accuracy' in result:
                        accuracy_scores.append(result['accuracy']['hard_label'])
            print(f"\nHard Label Predictions:")
            for label, count in sorted(hard_label_counts.items()):
                percentage = (count / info['successful_predictions']) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
            print(f"\nSoft Label Predictions:")
            for label, count in sorted(soft_label_counts.items()):
                percentage = (count / info['successful_predictions']) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
            if accuracy_scores:
                avg_accuracy = np.mean(accuracy_scores)
                print(f"\nAverage Hard Label Accuracy: {avg_accuracy:.3f}")
        print("="*60)
def main():
    parser = argparse.ArgumentParser(description='Ensemble Model Prediction')
    parser.add_argument('--ensemble_model', required=True, help='Path to ensemble model file')
    parser.add_argument('--input_path', required=True, help='Directory containing JSON files or single JSON file to predict')
    parser.add_argument('--output_dir', default='prediction_results', help='Output directory for results')
    parser.add_argument('--file_pattern', default='*.json', help='File pattern to match (default: *.json)')
    parser.add_argument('--output_filename', help='Output filename (optional)')
    args = parser.parse_args()
    predictor = EnsemblePredictor(
        ensemble_model_path=args.ensemble_model,
        output_dir=args.output_dir
    )
    if not predictor.load_ensemble_model():
        print("Failed to load ensemble model. Exiting.")
        return
    results = predictor.predict_directory(
        input_path=args.input_path,
        file_pattern=args.file_pattern
    )
    if results:
        output_path = predictor.save_results(results, args.output_filename)
        print(f"\nPrediction completed! Results saved to: {output_path}")
    else:
        print("No files were processed.")
if __name__ == "__main__":
    if len(sys.argv) == 1:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        ENSEMBLE_MODEL_PATH = str(project_root / "models" / "ensemble_model_20251022_032750.pkl")
        INPUT_PATH = str(project_root / "data")
        OUTPUT_DIR = str(project_root / "output")
        print("Running with default settings...")
        print(f"Ensemble Model: {ENSEMBLE_MODEL_PATH}")
        print(f"Input Path: {INPUT_PATH}")
        print(f"Output Directory: {OUTPUT_DIR}")
        print()
        predictor = EnsemblePredictor(
            ensemble_model_path=ENSEMBLE_MODEL_PATH,
            output_dir=OUTPUT_DIR
        )
        if not predictor.load_ensemble_model():
            print("Failed to load ensemble model. Please check the model path.")
            sys.exit(1)
        results = predictor.predict_directory(INPUT_PATH)
        if results:
            output_path = predictor.save_results(results)
            print(f"\nPrediction completed! Results saved to: {output_path}")
        else:
            print("No files were processed.")
    else:
        main()