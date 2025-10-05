import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from scipy.stats import entropy





"""
TWO-STAGE EXOPLANET CLASSIFICATION SYSTEM
==========================================

Philosophy:
- Stage 1: Make high-confidence decisions (OTHER or CONFIRMED)
- Stage 2: Route everything uncertain to CANDIDATE for human review
- Goal: Minimize false negatives (missing real exoplanets)
"""

# ============================================================================
# STAGE 1: HELPER FUNCTIONS
# ============================================================================

def calculate_uncertainty(proba):
    """Calculate Shannon entropy for each prediction"""
    return np.array([entropy(p) for p in proba])


def analyze_confidence_zones(y_true, y_pred_proba):
    """
    Analyze the data to find natural confidence thresholds
    Returns suggested thresholds based on data distribution
    """
    uncertainties = calculate_uncertainty(y_pred_proba)
    
    print("=" * 70)
    print("CONFIDENCE ZONE ANALYSIS")
    print("=" * 70)
    
    # Analyze by true class
    class_names = ['OTHER', 'CANDIDATE', 'CONFIRMED']
    
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        probs = y_pred_proba[mask]
        uncs = uncertainties[mask]
        
        print(f"\n{class_name}:")
        print(f"  Samples: {mask.sum()}")
        print(f"  P({class_name}) - Mean: {probs[:, class_idx].mean():.3f}, "
              f"Median: {np.median(probs[:, class_idx]):.3f}")
        print(f"  Uncertainty - Mean: {uncs.mean():.3f}, Median: {np.median(uncs):.3f}")
        
        # High confidence samples (top 25%)
        high_conf_threshold = np.percentile(probs[:, class_idx], 75)
        high_conf_mask = probs[:, class_idx] >= high_conf_threshold
        print(f"  High confidence (P >= {high_conf_threshold:.3f}): "
              f"{high_conf_mask.sum()} samples, avg uncertainty: {uncs[high_conf_mask].mean():.3f}")
    
    # Find natural separation points
    print("\n" + "-" * 70)
    print("SUGGESTED THRESHOLDS:")
    print("-" * 70)
    
    # For OTHER: high precision threshold
    other_mask = y_true == 0
    other_probs = y_pred_proba[other_mask, 0]
    other_threshold = np.percentile(other_probs, 70)  # 70th percentile
    print(f"OTHER threshold: {other_threshold:.3f} (70th percentile of true OTHER)")
    
    # For CONFIRMED: high precision threshold
    confirmed_mask = y_true == 2
    confirmed_probs = y_pred_proba[confirmed_mask, 2]
    confirmed_threshold = np.percentile(confirmed_probs, 65)  # 65th percentile
    print(f"CONFIRMED threshold: {confirmed_threshold:.3f} (65th percentile of true CONFIRMED)")
    
    # For uncertainty: median of CANDIDATE class
    candidate_mask = y_true == 1
    candidate_uncs = uncertainties[candidate_mask]
    unc_threshold = np.percentile(candidate_uncs, 50)  # Median
    print(f"Uncertainty threshold: {unc_threshold:.3f} (median of CANDIDATE uncertainty)")
    
    return {
        'other_threshold': other_threshold,
        'confirmed_threshold': confirmed_threshold,
        'uncertainty_threshold': unc_threshold
    }


# ============================================================================
# STAGE 2: TWO-STAGE CLASSIFIER
# ============================================================================

class TwoStageExoplanetClassifier:
    """
    Two-stage classification for exoplanet detection
    
    Stage 1: High-confidence filtering
    - Very confident OTHER (false positives) → reject immediately
    - Very confident CONFIRMED (validated planets) → accept immediately
    
    Stage 2: Route to CANDIDATE
    - Everything else needs human review/follow-up observation
    """
    
    def __init__(self, 
                 other_threshold=0.85,
                 confirmed_threshold=0.75,
                 uncertainty_threshold=0.50,
                 other_unc_max=0.40,
                 confirmed_unc_max=0.50):
        """
        Parameters:
        -----------
        other_threshold : float
            Minimum probability to classify as OTHER with high confidence
        confirmed_threshold : float
            Minimum probability to classify as CONFIRMED with high confidence
        uncertainty_threshold : float
            Maximum uncertainty for high-confidence decisions
        other_unc_max : float
            Maximum uncertainty allowed for OTHER classification
        confirmed_unc_max : float
            Maximum uncertainty allowed for CONFIRMED classification
        """
        self.other_threshold = other_threshold
        self.confirmed_threshold = confirmed_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.other_unc_max = other_unc_max
        self.confirmed_unc_max = confirmed_unc_max
    
    def predict(self, y_pred_proba):
        """
        Two-stage prediction
        
        Returns:
        --------
        predictions : array
            Predicted class labels (0=OTHER, 1=CANDIDATE, 2=CONFIRMED)
        confidence_levels : array
            Confidence level for each prediction ('HIGH', 'MEDIUM', 'LOW')
        stage : array
            Which stage made the decision ('STAGE1_OTHER', 'STAGE1_CONFIRMED', 'STAGE2_CANDIDATE')
        """
        uncertainties = calculate_uncertainty(y_pred_proba)
        
        predictions = []
        confidence_levels = []
        stages = []
        
        for prob, unc in zip(y_pred_proba, uncertainties):
            p_other, p_candidate, p_confirmed = prob
            
            # STAGE 1: HIGH-CONFIDENCE DECISIONS
            # ==================================
            
            # Very confident it's a false positive
            if (p_other >= self.other_threshold and 
                unc <= self.other_unc_max):
                pred = 0  # OTHER
                conf = 'HIGH'
                stage = 'STAGE1_OTHER'
            
            # Very confident it's a real planet
            elif (p_confirmed >= self.confirmed_threshold and 
                  unc <= self.confirmed_unc_max):
                pred = 2  # CONFIRMED
                conf = 'HIGH'
                stage = 'STAGE1_CONFIRMED'
            
            # STAGE 2: ROUTE TO CANDIDATE (HUMAN REVIEW)
            # ===========================================
            
            # High uncertainty - definitely needs review
            elif unc >= self.uncertainty_threshold:
                pred = 1  # CANDIDATE
                conf = 'LOW'
                stage = 'STAGE2_CANDIDATE'
            
            # Moderate evidence for CONFIRMED but not confident enough
            elif p_confirmed >= 0.55:
                pred = 1  # CANDIDATE (conservative: needs validation)
                conf = 'MEDIUM'
                stage = 'STAGE2_CANDIDATE'
            
            # Moderate evidence for OTHER but not confident enough
            elif p_other >= 0.60:
                pred = 1  # CANDIDATE (could be false positive, needs review)
                conf = 'MEDIUM'
                stage = 'STAGE2_CANDIDATE'
            
            # Everything else - ambiguous
            else:
                pred = 1  # CANDIDATE
                conf = 'LOW'
                stage = 'STAGE2_CANDIDATE'
            
            predictions.append(pred)
            confidence_levels.append(conf)
            stages.append(stage)
        
        return np.array(predictions), np.array(confidence_levels), np.array(stages)
    
    def get_config(self):
        """Return current configuration"""
        return {
            'other_threshold': self.other_threshold,
            'confirmed_threshold': self.confirmed_threshold,
            'uncertainty_threshold': self.uncertainty_threshold,
            'other_unc_max': self.other_unc_max,
            'confirmed_unc_max': self.confirmed_unc_max
        }


# ============================================================================
# STAGE 3: OPTIMIZATION AND EVALUATION
# ============================================================================

def optimize_two_stage_thresholds(y_true, y_pred_proba, 
                                 metric='confirmed_recall',
                                 min_other_precision=0.85,
                                 min_confirmed_precision=0.80):
    """
    Find optimal thresholds for two-stage classifier
    
    Parameters:
    -----------
    metric : str
        What to optimize: 'confirmed_recall', 'other_precision', 'balanced'
    min_other_precision : float
        Minimum acceptable precision for OTHER class
    min_confirmed_precision : float
        Minimum acceptable precision for CONFIRMED class
    """
    print("\n" + "=" * 70)
    print("OPTIMIZING TWO-STAGE CLASSIFIER")
    print("=" * 70)
    
    results = []
    
    # Grid search over threshold combinations
    for other_thresh in np.arange(0.70, 0.95, 0.05):
        for conf_thresh in np.arange(0.65, 0.90, 0.05):
            for unc_thresh in np.arange(0.35, 0.65, 0.05):
                for other_unc in np.arange(0.30, 0.55, 0.05):
                    for conf_unc in np.arange(0.35, 0.60, 0.05):
                        
                        # Create classifier with these thresholds
                        clf = TwoStageExoplanetClassifier(
                            other_threshold=other_thresh,
                            confirmed_threshold=conf_thresh,
                            uncertainty_threshold=unc_thresh,
                            other_unc_max=other_unc,
                            confirmed_unc_max=conf_unc
                        )
                        
                        # Predict
                        y_pred, conf_levels, stages = clf.predict(y_pred_proba)
                        
                        # Calculate metrics
                        from sklearn.metrics import precision_score, recall_score
                        
                        # Per-class metrics
                        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
                        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
                        
                        # Check constraints
                        if precision[0] < min_other_precision:
                            continue  # OTHER precision too low
                        if precision[2] < min_confirmed_precision:
                            continue  # CONFIRMED precision too low
                        
                        # Calculate F1 scores
                        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                        
                        # Count CANDIDATE assignments
                        candidate_count = (y_pred == 1).sum()
                        candidate_pct = candidate_count / len(y_pred) * 100
                        
                        results.append({
                            'other_threshold': other_thresh,
                            'confirmed_threshold': conf_thresh,
                            'uncertainty_threshold': unc_thresh,
                            'other_unc_max': other_unc,
                            'confirmed_unc_max': conf_unc,
                            'other_precision': precision[0],
                            'other_recall': recall[0],
                            'other_f1': f1_scores[0],
                            'candidate_precision': precision[1],
                            'candidate_recall': recall[1],
                            'candidate_f1': f1_scores[1],
                            'confirmed_precision': precision[2],
                            'confirmed_recall': recall[2],
                            'confirmed_f1': f1_scores[2],
                            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
                            'accuracy': accuracy_score(y_true, y_pred),
                            'candidate_count': candidate_count,
                            'candidate_pct': candidate_pct
                        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("WARNING: No configurations met the precision constraints!")
        print(f"  Required: OTHER precision >= {min_other_precision}")
        print(f"           CONFIRMED precision >= {min_confirmed_precision}")
        return None, None, None
    
    # Find best based on chosen metric
    if metric == 'confirmed_recall':
        best = results_df.loc[results_df['confirmed_recall'].idxmax()]
        print(f"\nOptimizing for: Maximum CONFIRMED recall")
    elif metric == 'other_precision':
        best = results_df.loc[results_df['other_precision'].idxmax()]
        print(f"\nOptimizing for: Maximum OTHER precision")
    elif metric == 'balanced':
        # Balance between confirmed recall and keeping candidate count reasonable
        results_df['score'] = (results_df['confirmed_recall'] * 0.4 + 
                              results_df['other_recall'] * 0.3 +
                              (100 - results_df['candidate_pct']) / 100 * 0.3)
        best = results_df.loc[results_df['score'].idxmax()]
        print(f"\nOptimizing for: Balanced approach")
    else:
        best = results_df.loc[results_df['f1_weighted'].idxmax()]
        print(f"\nOptimizing for: Weighted F1")
    
    # Also find configuration that minimizes CANDIDATE assignments
    min_candidates = results_df.loc[results_df['candidate_pct'].idxmin()]
    
    return results_df, best, min_candidates


def evaluate_two_stage_classifier(y_true, y_pred_proba, classifier):
    """
    Comprehensive evaluation of two-stage classifier
    """
    print("\n" + "=" * 70)
    print("TWO-STAGE CLASSIFIER EVALUATION")
    print("=" * 70)
    
    # Get predictions
    y_pred, conf_levels, stages = classifier.predict(y_pred_proba)
    
    # Basic metrics
    class_names = ['OTHER', 'CANDIDATE', 'CONFIRMED']
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Stage analysis
    print("\n" + "-" * 70)
    print("STAGE ANALYSIS")
    print("-" * 70)
    
    stage_counts = pd.Series(stages).value_counts()
    print("\nPredictions by Stage:")
    for stage_name, count in stage_counts.items():
        pct = count / len(stages) * 100
        print(f"  {stage_name:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Confidence analysis
    print("\n" + "-" * 70)
    print("CONFIDENCE ANALYSIS")
    print("-" * 70)
    
    conf_counts = pd.Series(conf_levels).value_counts()
    print("\nPredictions by Confidence:")
    for conf_name in ['HIGH', 'MEDIUM', 'LOW']:
        if conf_name in conf_counts:
            count = conf_counts[conf_name]
            pct = count / len(conf_levels) * 100
            print(f"  {conf_name:10s}: {count:4d} ({pct:5.1f}%)")
    
    # Detailed breakdown
    print("\n" + "-" * 70)
    print("DETAILED BREAKDOWN")
    print("-" * 70)
    
    results_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'predicted': [class_names[i] for i in y_pred],
        'confidence': conf_levels,
        'stage': stages
    })
    
    print("\nPredictions by Stage and Confidence:")
    breakdown = results_df.groupby(['stage', 'confidence', 'predicted']).size()
    print(breakdown)
    
    # Key metrics for astronomy
    print("\n" + "-" * 70)
    print("KEY ASTRONOMY METRICS")
    print("-" * 70)
    
    # How many real planets were caught?
    confirmed_mask = y_true == 2
    confirmed_caught = (y_pred[confirmed_mask] == 2).sum()
    confirmed_to_candidate = (y_pred[confirmed_mask] == 1).sum()
    confirmed_lost = (y_pred[confirmed_mask] == 0).sum()
    
    print(f"\nReal CONFIRMED planets (n={confirmed_mask.sum()}):")
    print(f"  Correctly classified as CONFIRMED: {confirmed_caught} ({confirmed_caught/confirmed_mask.sum()*100:.1f}%)")
    print(f"  Routed to CANDIDATE (for review):  {confirmed_to_candidate} ({confirmed_to_candidate/confirmed_mask.sum()*100:.1f}%)")
    print(f"  Incorrectly rejected as OTHER:     {confirmed_lost} ({confirmed_lost/confirmed_mask.sum()*100:.1f}%)")
    print(f"  → Total recoverable: {confirmed_caught + confirmed_to_candidate} ({(confirmed_caught + confirmed_to_candidate)/confirmed_mask.sum()*100:.1f}%)")
    
    # How many false positives were filtered?
    other_mask = y_true == 0
    other_filtered = (y_pred[other_mask] == 0).sum()
    other_to_candidate = (y_pred[other_mask] == 1).sum()
    
    print(f"\nFalse positives/OTHER (n={other_mask.sum()}):")
    print(f"  Correctly filtered out:            {other_filtered} ({other_filtered/other_mask.sum()*100:.1f}%)")
    print(f"  Sent to review (CANDIDATE):        {other_to_candidate} ({other_to_candidate/other_mask.sum()*100:.1f}%)")
    
    # Workload analysis
    candidate_count = (y_pred == 1).sum()
    print(f"\nWorkload Analysis:")
    print(f"  Total requiring human review: {candidate_count} ({candidate_count/len(y_pred)*100:.1f}%)")
    print(f"  Automatic decisions:          {len(y_pred) - candidate_count} ({(len(y_pred) - candidate_count)/len(y_pred)*100:.1f}%)")
    
    return y_pred, conf_levels, stages, results_df


# ============================================================================
# STAGE 4: VISUALIZATION
# ============================================================================

def visualize_two_stage_results(y_true, y_pred_proba, y_pred, conf_levels, stages):
    """
    Create visualizations for two-stage classification
    """
    class_names = ['OTHER', 'CANDIDATE', 'CONFIRMED']
    uncertainties = calculate_uncertainty(y_pred_proba)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Confusion Matrix
    ax1 = plt.subplot(3, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix\n(Two-Stage Classifier)')
    
    # Plot 2: Predictions by Stage
    ax2 = plt.subplot(3, 3, 2)
    stage_pred_df = pd.DataFrame({
        'stage': stages,
        'predicted': [class_names[i] for i in y_pred]
    })
    stage_counts = stage_pred_df.groupby(['stage', 'predicted']).size().unstack(fill_value=0)
    stage_counts.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_xlabel('Stage')
    ax2.set_ylabel('Count')
    ax2.set_title('Predictions by Stage')
    ax2.legend(title='Predicted Class', bbox_to_anchor=(1.05, 1))
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Predictions by Confidence
    ax3 = plt.subplot(3, 3, 3)
    conf_pred_df = pd.DataFrame({
        'confidence': conf_levels,
        'predicted': [class_names[i] for i in y_pred]
    })
    conf_counts = conf_pred_df.groupby(['confidence', 'predicted']).size().unstack(fill_value=0)
    conf_order = ['HIGH', 'MEDIUM', 'LOW']
    conf_counts = conf_counts.reindex([c for c in conf_order if c in conf_counts.index])
    conf_counts.plot(kind='bar', stacked=True, ax=ax3)
    ax3.set_xlabel('Confidence Level')
    ax3.set_ylabel('Count')
    ax3.set_title('Predictions by Confidence')
    ax3.legend(title='Predicted Class')
    ax3.tick_params(axis='x', rotation=0)
    
    # Plot 4: Uncertainty distribution by prediction
    ax4 = plt.subplot(3, 3, 4)
    for pred_class in range(3):
        mask = y_pred == pred_class
        ax4.hist(uncertainties[mask], alpha=0.5, bins=30, label=class_names[pred_class])
    ax4.set_xlabel('Uncertainty (Entropy)')
    ax4.set_ylabel('Count')
    ax4.set_title('Uncertainty by Predicted Class')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Max probability vs Uncertainty colored by stage
    ax5 = plt.subplot(3, 3, 5)
    max_probs = np.max(y_pred_proba, axis=1)
    stage_colors = {'STAGE1_OTHER': 'blue', 'STAGE1_CONFIRMED': 'green', 
                   'STAGE2_CANDIDATE': 'orange'}
    for stage_name, color in stage_colors.items():
        mask = stages == stage_name
        if mask.sum() > 0:
            ax5.scatter(max_probs[mask], uncertainties[mask], 
                       alpha=0.4, s=10, c=color, label=stage_name)
    ax5.set_xlabel('Max Probability')
    ax5.set_ylabel('Uncertainty (Entropy)')
    ax5.set_title('Decision Space by Stage')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: True vs Predicted breakdown
    ax6 = plt.subplot(3, 3, 6)
    true_pred_df = pd.DataFrame({
        'true': [class_names[i] for i in y_true],
        'predicted': [class_names[i] for i in y_pred]
    })
    true_pred_counts = true_pred_df.groupby(['true', 'predicted']).size().unstack(fill_value=0)
    true_pred_counts.plot(kind='bar', ax=ax6)
    ax6.set_xlabel('True Class')
    ax6.set_ylabel('Count')
    ax6.set_title('True vs Predicted Distribution')
    ax6.legend(title='Predicted')
    ax6.tick_params(axis='x', rotation=45)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 7: Recovery rate for real planets
    ax7 = plt.subplot(3, 3, 7)
    confirmed_mask = y_true == 2
    recovery_data = {
        'Correctly\nCONFIRMED': (y_pred[confirmed_mask] == 2).sum(),
        'Routed to\nCANDIDATE': (y_pred[confirmed_mask] == 1).sum(),
        'Lost to\nOTHER': (y_pred[confirmed_mask] == 0).sum()
    }
    colors = ['green', 'orange', 'red']
    ax7.bar(recovery_data.keys(), recovery_data.values(), color=colors, alpha=0.7)
    ax7.set_ylabel('Count')
    ax7.set_title(f'Recovery of Real Planets\n(n={confirmed_mask.sum()} total)')
    ax7.grid(True, alpha=0.3, axis='y')
    for i, (k, v) in enumerate(recovery_data.items()):
        pct = v / confirmed_mask.sum() * 100
        ax7.text(i, v + 5, f'{v}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # Plot 8: False positive filtering
    ax8 = plt.subplot(3, 3, 8)
    other_mask = y_true == 0
    filtering_data = {
        'Filtered\n(OTHER)': (y_pred[other_mask] == 0).sum(),
        'Needs Review\n(CANDIDATE)': (y_pred[other_mask] == 1).sum(),
        'Misclassified\n(CONFIRMED)': (y_pred[other_mask] == 2).sum()
    }
    colors = ['green', 'orange', 'red']
    ax8.bar(filtering_data.keys(), filtering_data.values(), color=colors, alpha=0.7)
    ax8.set_ylabel('Count')
    ax8.set_title(f'False Positive Filtering\n(n={other_mask.sum()} total)')
    ax8.grid(True, alpha=0.3, axis='y')
    for i, (k, v) in enumerate(filtering_data.items()):
        pct = v / other_mask.sum() * 100
        ax8.text(i, v + 10, f'{v}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # Plot 9: Workload summary
    ax9 = plt.subplot(3, 3, 9)
    workload_data = {
        'Automatic\nDecisions': (stages != 'STAGE2_CANDIDATE').sum(),
        'Human\nReview': (stages == 'STAGE2_CANDIDATE').sum()
    }
    colors = ['lightblue', 'lightcoral']
    wedges, texts, autotexts = ax9.pie(workload_data.values(), labels=workload_data.keys(), 
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax9.set_title('Workload Distribution')
    
    plt.tight_layout()
    plt.savefig('two_stage_classification_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'two_stage_classification_results.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TWO-STAGE EXOPLANET CLASSIFICATION SYSTEM")
    print("=" * 70)
    
    # Assuming you have:
    # - y_test: true labels
    # - y_pred_proba: predicted probabilities from XGBoost
    
    # Step 1: Analyze confidence zones in your data
    suggested_thresholds = analyze_confidence_zones(y_test, y_pred_proba)
    
    # Step 2: Optimize two-stage classifier
    print("\nOptimizing classifier thresholds...")
    results_df, best_config, min_candidate_config = optimize_two_stage_thresholds(
        y_test, 
        y_pred_proba,
        metric='balanced',  # Change to 'confirmed_recall' or 'other_precision' as needed
        min_other_precision=0.83,
        min_confirmed_precision=0.78
    )
    
    if best_config is not None:
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        print(f"OTHER threshold: {best_config['other_threshold']:.3f}")
        print(f"CONFIRMED threshold: {best_config['confirmed_threshold']:.3f}")
        print(f"Uncertainty threshold: {best_config['uncertainty_threshold']:.3f}")
        print(f"OTHER uncertainty max: {best_config['other_unc_max']:.3f}")
        print(f"CONFIRMED uncertainty max: {best_config['confirmed_unc_max']:.3f}")
        print(f"\nPerformance:")
        print(f"  OTHER - Precision: {best_config['other_precision']:.3f}, Recall: {best_config['other_recall']:.3f}")
        print(f"  CANDIDATE - Precision: {best_config['candidate_precision']:.3f}, Recall: {best_config['candidate_recall']:.3f}")
        print(f"  CONFIRMED - Precision: {best_config['confirmed_precision']:.3f}, Recall: {best_config['confirmed_recall']:.3f}")
        print(f"  Candidates for review: {best_config['candidate_count']} ({best_config['candidate_pct']:.1f}%)")
        
        # Step 3: Create and evaluate classifier with best config
        best_classifier = TwoStageExoplanetClassifier(
            other_threshold=best_config['other_threshold'],
            confirmed_threshold=best_config['confirmed_threshold'],
            uncertainty_threshold=best_config['uncertainty_threshold'],
            other_unc_max=best_config['other_unc_max'],
            confirmed_unc_max=best_config['confirmed_unc_max']
        )
        
        y_pred, conf_levels, stages, results_detailed = evaluate_two_stage_classifier(
            y_test, y_pred_proba, best_classifier
        )
        
        # Step 4: Visualize results
        visualize_two_stage_results(y_test, y_pred_proba, y_pred, conf_levels, stages)
        
        # Step 5: Save detailed results
        results_detailed['uncertainty'] = calculate_uncertainty(y_pred_proba)
        results_detailed['p_other'] = y_pred_proba[:, 0]
        results_detailed['p_candidate'] = y_pred_proba[:, 1]
        results_detailed['p_confirmed'] = y_pred_proba[:, 2]
        results_detailed.to_csv('two_stage_predictions_detaile