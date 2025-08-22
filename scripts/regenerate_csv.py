#!/usr/bin/env python3
"""
Regenerate CSV files from existing JSON results with fixed parsing logic
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

def generate_csv_from_json(output_dir):
    """Generate CSV files from JSON results in the output directory."""
    output_path = Path(output_dir)
    
    # Find all JSON files
    json_files = list(output_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {output_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Lists to store data
    summary_data = []
    individual_data = []
    
    for json_file in json_files:
        if not json_file.exists():
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model and noise type from filename
            filename = json_file.stem
            parts = filename.split('_')
            
            # Parse model name and noise type
            if 'ai4bharat' in filename:
                model = 'ai4bharat/indic-conformer-600m-multilingual'
                # For ai4bharat filename pattern: ai4bharat_indic_conformer_NOISE_YYYYMMDD_HHMMSS
                # Find the part that contains noise info (before the date)
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():  # Date pattern YYYYMMDD
                        # Look backwards to find noise specification
                        noise_parts = []
                        for j in range(i-1, -1, -1):
                            if parts[j] in ['ai4bharat', 'indic', 'conformer', '600m', 'multilingual']:
                                break
                            noise_parts.insert(0, parts[j])
                        noise_type = '_'.join(noise_parts) if noise_parts else 'unknown'
                        break
                else:
                    noise_type = 'unknown'
            elif 'gemini' in filename:
                if '2_0' in filename or '2.0' in filename:
                    model = 'gemini-2.0-flash'
                elif '2_5' in filename or '2.5' in filename:
                    model = 'gemini-2.5-flash'
                else:
                    model = 'gemini-unknown'
                # For gemini filename pattern: gemini_2.0_flash_NOISE_YYYYMMDD_HHMMSS
                # Find the part that contains noise info (before the date)
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():  # Date pattern YYYYMMDD
                        # Look backwards to find noise specification
                        noise_parts = []
                        for j in range(i-1, -1, -1):
                            if parts[j] in ['gemini', '2', '0', '5', 'flash']:
                                break
                            noise_parts.insert(0, parts[j])
                        noise_type = '_'.join(noise_parts) if noise_parts else 'unknown'
                        break
                else:
                    noise_type = 'unknown'
            else:
                model = 'unknown'
                noise_type = 'unknown'
            
            # Parse noise type and configuration
            if '_' in noise_type:
                # Convert underscore back to colon for proper parsing
                # e.g., gaussian_low -> gaussian:low
                parts_with_underscore = noise_type.split('_')
                if len(parts_with_underscore) >= 2:
                    # Check if the last part is an intensity level
                    intensity_levels = ['low', 'medium', 'high', 'extreme']
                    if parts_with_underscore[-1] in intensity_levels:
                        base_noise_type = '_'.join(parts_with_underscore[:-1])
                        noise_config = parts_with_underscore[-1]
                        
                        # Convert noise configuration to actual values
                        try:
                            from karma.utils.noise.noise_intensity_config import get_noise_intensity_value
                            actual_value = get_noise_intensity_value(base_noise_type, noise_config)
                            if actual_value is not None:
                                noise_config = str(actual_value)
                        except ImportError:
                            # Fallback: manual mapping if import fails
                            intensity_mapping = {
                                'background_noise': {'low': '5', 'medium': '10', 'high': '20', 'extreme': '40'},
                                'gaussian': {'low': '0.001', 'medium': '0.005', 'high': '0.009', 'extreme': '0.013'},
                                'short_noise': {'low': '1', 'medium': '4', 'high': '7', 'extreme': '10'},
                                'color_noise': {'low': '5', 'medium': '10', 'high': '20', 'extreme': '40'},
                                'clip': {'low': '0.5', 'medium': '0.6', 'high': '0.8', 'extreme': '0.9'}
                            }
                            if base_noise_type in intensity_mapping and noise_config in intensity_mapping[base_noise_type]:
                                noise_config = intensity_mapping[base_noise_type][noise_config]
                    else:
                        base_noise_type = noise_type
                        noise_config = 'default'
                else:
                    base_noise_type = noise_type
                    noise_config = 'default'
            elif ':' in noise_type:
                base_noise_type, noise_config = noise_type.split(':', 1)
                
                # Convert noise configuration to actual values
                try:
                    from karma.utils.noise.noise_intensity_config import get_noise_intensity_value
                    actual_value = get_noise_intensity_value(base_noise_type, noise_config)
                    if actual_value is not None:
                        noise_config = str(actual_value)
                except ImportError:
                    # Fallback: manual mapping if import fails
                    intensity_mapping = {
                        'background_noise': {'low': '5', 'medium': '10', 'high': '20', 'extreme': '40'},
                        'gaussian': {'low': '0.001', 'medium': '0.005', 'high': '0.009', 'extreme': '0.013'},
                        'short_noise': {'low': '1', 'medium': '4', 'high': '7', 'extreme': '10'},
                        'color_noise': {'low': '5', 'medium': '10', 'high': '20', 'extreme': '40'},
                        'clip': {'low': '0.5', 'medium': '0.6', 'high': '0.8', 'extreme': '0.9'}
                    }
                    if base_noise_type in intensity_mapping and noise_config in intensity_mapping[base_noise_type]:
                        noise_config = intensity_mapping[base_noise_type][noise_config]
            else:
                base_noise_type = noise_type
                noise_config = 'default'
            
            print(f"Processing {filename}: model={model}, noise_type={base_noise_type}, noise_config={noise_config}")
            
            # Extract dataset and metrics from JSON
            for dataset_key, dataset_data in data.items():
                if dataset_key.startswith('_'):  # Skip _summary
                    continue
                    
                metrics = dataset_data.get('metrics', {})
                
                # Summary row for this evaluation
                summary_row = {
                    'model': model,
                    'dataset': dataset_key,
                    'noise_type': base_noise_type,
                    'noise_config': noise_config,
                    'status': 'success'
                }
                
                # Add aggregate scores if available
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict):
                        # Check if there's a score nested inside
                        if 'score' in metric_data:
                            score_data = metric_data['score']
                            if isinstance(score_data, dict):
                                # Use the metric_name key if available, otherwise try to find a main score
                                if metric_name in score_data:
                                    summary_row[metric_name] = score_data[metric_name]
                                elif 'aggregate' in score_data:
                                    summary_row[metric_name] = score_data['aggregate']
                                else:
                                    # For asr_semantic_jiw, use semantic_wer as the main score
                                    if metric_name == 'asr_semantic_jiw' and 'semantic_wer' in score_data:
                                        summary_row[metric_name] = score_data['semantic_wer']
                                    else:
                                        summary_row[metric_name] = 'N/A'
                            else:
                                summary_row[metric_name] = score_data
                        else:
                            summary_row[metric_name] = metric_data.get('aggregate', 'N/A')
                    else:
                        summary_row[metric_name] = metric_data
                
                summary_data.append(summary_row)
                
                # Individual samples data
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict):
                        # Check if there's a score nested inside
                        score_data = metric_data.get('score', metric_data)
                        
                        # Look for per-sample data in various possible keys
                        per_sample_data = None
                        if 'per_sample' in score_data:
                            per_sample_data = score_data['per_sample']
                        elif 'per_sample_wer' in score_data:
                            per_sample_data = score_data['per_sample_wer']
                        elif 'per_sample_cer' in score_data:
                            per_sample_data = score_data['per_sample_cer']
                        elif 'per_sample_semantic' in score_data:
                            per_sample_data = score_data['per_sample_semantic']
                        
                        if per_sample_data and isinstance(per_sample_data, list):
                            for i, sample_data in enumerate(per_sample_data):
                                if isinstance(sample_data, dict):
                                    # Extract the main score for this metric type
                                    if metric_name == 'wer_jiw' and 'wer' in sample_data:
                                        score_value = sample_data['wer']
                                    elif metric_name == 'cer_jiw' and 'cer' in sample_data:
                                        score_value = sample_data['cer']
                                    elif metric_name == 'asr_semantic_jiw' and 'semantic_wer' in sample_data:
                                        score_value = sample_data['semantic_wer']
                                    else:
                                        score_value = sample_data
                                    
                                    individual_data.append({
                                        'model': model,
                                        'dataset': dataset_key,
                                        'noise_type': base_noise_type,
                                        'noise_config': noise_config,
                                        'sample_index': i,
                                        'file_id': sample_data.get('file_id', f'sample_{i:03d}'),
                                        'metric': metric_name,
                                        'score': score_value,
                                        'prediction': sample_data.get('prediction', ''),
                                        'reference': sample_data.get('reference', '')
                                    })
                                else:
                                    individual_data.append({
                                        'model': model,
                                        'dataset': dataset_key,
                                        'noise_type': base_noise_type,
                                        'noise_config': noise_config,
                                        'sample_index': i,
                                        'file_id': f'sample_{i:03d}',
                                        'metric': metric_name,
                                        'score': sample_data,
                                        'prediction': '',
                                        'reference': ''
                                    })
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Create CSV files
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Rename columns for clarity
        summary_df = summary_df.rename(columns={
            'asr_semantic_jiw': 'semantic_wer',
            'wer_jiw': 'wer',
            'cer_jiw': 'cer'
        })
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_path / f"evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\U0001F4CA Summary CSV: {summary_file}")
        print(f"Summary shape: {summary_df.shape}")
    
    if individual_data:
        individual_df = pd.DataFrame(individual_data)
        # Rename metric column values for clarity
        if 'metric' in individual_df.columns:
            individual_df['metric'] = individual_df['metric'].replace({
                'asr_semantic_jiw': 'semantic_wer',
                'wer_jiw': 'wer',
                'cer_jiw': 'cer'
            })
        individual_file = output_path / f"individual_files_{timestamp}.csv"
        individual_df.to_csv(individual_file, index=False)
        print(f"\U0001F4CB Individual CSV: {individual_file}")
        print(f"Individual shape: {individual_df.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python regenerate_csv.py <output_directory>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    generate_csv_from_json(output_dir)
