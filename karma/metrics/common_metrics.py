import evaluate
from sklearn.metrics import f1_score
from collections import Counter
import re
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
import jiwer


class HfMetric(BaseMetric):
    def __init__(self, metric_name: str, **kwargs):
        super().__init__(metric_name)
        self.metric = evaluate.load(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)


@register_metric(
    name="bleu",
    optional_args=["max_order", "smooth"],
    default_args={"max_order": 4, "smooth": True},
)
class BleuMetric(HfMetric):
    def __init__(self, metric_name: str = "bleu", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        smooth = kwargs.get("smooth", True)
        references = [[ref] for ref in references]
        return self.metric.compute(
            predictions=predictions, references=references, smooth=smooth
        )


@register_metric("exact_match", optional_args=["ignore_case"], default_args={"ignore_case": True})
class ExactMatchMetric(HfMetric):
    def __init__(self, metric_name: str = "exact_match", **kwargs):
        super().__init__(metric_name)
    
    def evaluate(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references, ignore_case=kwargs.get("ignore_case", True))


@register_metric("f1")
class F1Metric(HfMetric):
    def __init__(self, metric_name: str = "f1", **kwargs):
        super().__init__(metric_name)


@register_metric("wer")
class WERMetric(HfMetric):
    def __init__(self, metric_name: str = "wer", **kwargs):
        super().__init__(metric_name)


@register_metric("cer")
class CERMetric(HfMetric):
    def __init__(self, metric_name: str = "cer", **kwargs):
        super().__init__(metric_name)


# ===== PER-SAMPLE JIWER METRICS (for file-wise analysis) =====
@register_metric("wer_jiw")
class WERJiwerMetric(BaseMetric):
    def __init__(self, metric_name: str = "wer_jiw", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        # Compute WER per sample using jiwer
        per_sample_scores = []
        total_wer = 0
        samples = kwargs.get('samples', [])
        
        print(f"[DEBUG WER] Got {len(predictions)} predictions and {len(references)} references")
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            print(f"[DEBUG WER] Sample {i}: pred='{pred}', ref='{ref}'")
            wer_score = jiwer.wer(ref, pred)
            print(f"[DEBUG WER] WER score: {wer_score}")
            total_wer += wer_score
            
            # Include file info if available
            file_id = f"sample_{i+1:03d}"
            if samples and i < len(samples):
                # Try to get filename or ID from sample
                sample_data = samples[i] if hasattr(samples[i], '__dict__') else samples[i]
                if hasattr(sample_data, 'get'):
                    file_id = sample_data.get('id', file_id) or sample_data.get('filename', file_id)
            
            per_sample_scores.append({
                'file_id': file_id,
                'wer': wer_score,
                'prediction': pred,
                'reference': ref
            })
        
        # Calculate average for benchmark (what gets displayed in overall_score)
        avg_wer = total_wer / len(predictions) if predictions else 0
        
        # Return both individual scores and average for benchmark
        return {
            'wer_jiw': avg_wer,  # Benchmark extracts this
            'per_sample_wer': per_sample_scores  # Individual scores stored here
        }


@register_metric("cer_jiw")
class CERJiwerMetric(BaseMetric):
    def __init__(self, metric_name: str = "cer_jiw", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        # Compute CER per sample using jiwer
        per_sample_scores = []
        total_cer = 0
        samples = kwargs.get('samples', [])
        
        print(f"[DEBUG CER] Got {len(predictions)} predictions and {len(references)} references")
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            print(f"[DEBUG CER] Sample {i}: pred='{pred}', ref='{ref}'")
            cer_score = jiwer.cer(ref, pred)
            print(f"[DEBUG CER] CER score: {cer_score}")
            total_cer += cer_score
            
            # Include file info if available
            file_id = f"sample_{i+1:03d}"
            if samples and i < len(samples):
                # Try to get filename or ID from sample
                sample_data = samples[i] if hasattr(samples[i], '__dict__') else samples[i]
                if hasattr(sample_data, 'get'):
                    file_id = sample_data.get('id', file_id) or sample_data.get('filename', file_id)
            
            per_sample_scores.append({
                'file_id': file_id,
                'cer': cer_score,
                'prediction': pred,
                'reference': ref
            })
        
        # Calculate average for benchmark (what gets displayed in overall_score)
        avg_cer = total_cer / len(predictions) if predictions else 0
        
        # Return both individual scores and average for benchmark
        return {
            'cer_jiw': avg_cer,  # Benchmark extracts this
            'per_sample_cer': per_sample_scores  # Individual scores stored here
        }
    


@register_metric("tokenised_f1")
class TokenisedF1Metric(BaseMetric):
    def __init__(self, metric_name: str = "tokenised_f1", **kwargs):
        super().__init__(metric_name)

    def tokenize(self, text):
        text = text.replace('\n', '').replace('.', '')
        return re.findall(r'\w+', text.lower())
    
    def evaluate(self, predictions, references , **kwargs):
        f1_scores = []
        for prediction, reference in zip(predictions, references):
            pred_tokens = self.tokenize(prediction)
            gold_tokens = self.tokenize(reference)
    
            pred_counts = Counter(pred_tokens)
            gold_counts = Counter(gold_tokens)
    
            # Compute overlap
            common = pred_counts & gold_counts
            num_same = sum(common.values())
    
            if num_same == 0:
                f1_scores.append(0.0)
                continue
    
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        return sum(f1_scores)/len(f1_scores)


# ===== INDIVIDUAL ASR SEMANTIC METRIC =====
@register_metric("asr_semantic_jiw")
class ASRSemanticJiwerMetric(BaseMetric):
    def __init__(self, metric_name: str = "asr_semantic_jiw", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        # Import the semantic metric logic
        from karma.metrics.asr.asr_semantic_metrics import ASRSemanticMetrics
        
        # Extract language from kwargs
        language = kwargs.get("language", "hi")
        samples = kwargs.get('samples', [])
        
        print(f"[DEBUG ASR SEMANTIC] Got {len(predictions)} predictions and {len(references)} references")
        print(f"[DEBUG ASR SEMANTIC] Language: {language}")
        
        # Get the aligner for this language
        try:
            aligner = ASRSemanticMetrics.get_aligner(language)
        except Exception as e:
            print(f"[ERROR] Could not get aligner for language {language}: {e}")
            # Fallback to jiwer if aligner fails
            per_sample_scores = []
            total_semantic_wer = 0
            total_semantic_cer = 0
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                wer_score = jiwer.wer(ref, pred)
                cer_score = jiwer.cer(ref, pred)
                total_semantic_wer += wer_score
                total_semantic_cer += cer_score
                
                file_id = f"sample_{i+1:03d}"
                per_sample_scores.append({
                    'file_id': file_id,
                    'semantic_wer': wer_score,
                    'semantic_cer': cer_score,
                    'prediction': pred,
                    'reference': ref,
                    'note': 'fallback_jiwer'
                })
            
            avg_semantic_wer = total_semantic_wer / len(predictions) if predictions else 0
            avg_semantic_cer = total_semantic_cer / len(predictions) if predictions else 0
            
            return {
                'asr_semantic_jiw': avg_semantic_wer,  # Benchmark extracts this
                'per_sample_semantic': per_sample_scores,
                'semantic_wer': avg_semantic_wer,
                'semantic_cer': avg_semantic_cer
            }
        
        # Process each prediction-reference pair individually
        per_sample_scores = []
        total_semantic_wer = 0
        total_semantic_cer = 0
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            print(f"[DEBUG ASR SEMANTIC] Sample {i}: pred='{pred[:50]}...', ref='{ref[:50]}...'")
            
            try:
                # Process single utterance using the aligner
                args = (i, ref, pred, language, 0.4)  # cer_threshold = 0.4
                result = ASRSemanticMetrics._process_utterance_regular(args)
                
                if result.success:
                    # Calculate WER and CER for this individual sample
                    sample_wer = (result.word_substitutions + result.word_deletions + result.word_insertions) / result.total_ref_words if result.total_ref_words > 0 else 0
                    sample_cer = result.edit_distance / result.ref_chars if result.ref_chars > 0 else 0
                else:
                    # Fallback to jiwer for this sample
                    sample_wer = jiwer.wer(ref, pred)
                    sample_cer = jiwer.cer(ref, pred)
                    
            except Exception as e:
                print(f"[ERROR] Error processing sample {i}: {e}")
                # Fallback to jiwer
                sample_wer = jiwer.wer(ref, pred)
                sample_cer = jiwer.cer(ref, pred)
            
            total_semantic_wer += sample_wer
            total_semantic_cer += sample_cer
            
            # Include file info if available
            file_id = f"sample_{i+1:03d}"
            if samples and i < len(samples):
                sample_data = samples[i] if hasattr(samples[i], '__dict__') else samples[i]
                if hasattr(sample_data, 'get'):
                    file_id = sample_data.get('id', file_id) or sample_data.get('filename', file_id)
            
            per_sample_scores.append({
                'file_id': file_id,
                'semantic_wer': sample_wer,
                'semantic_cer': sample_cer,
                'prediction': pred,
                'reference': ref
            })
            
            print(f"[DEBUG ASR SEMANTIC] Sample {i} - semantic_wer: {sample_wer}, semantic_cer: {sample_cer}")
        
        # Calculate averages for benchmark
        avg_semantic_wer = total_semantic_wer / len(predictions) if predictions else 0
        avg_semantic_cer = total_semantic_cer / len(predictions) if predictions else 0
        
        # Return both individual scores and averages for benchmark
        return {
            'asr_semantic_jiw': avg_semantic_wer,  # Benchmark extracts this
            'per_sample_semantic': per_sample_scores,  # Individual scores stored here
            'semantic_wer': avg_semantic_wer,
            'semantic_cer': avg_semantic_cer
        }
    
