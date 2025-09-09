"""
Execute the advanced modeling pipeline in stages with progress tracking
"""

from advanced_modeling_pipeline_clean import AdvancedModelingPipeline
import time
import json

def main():
    print("=" * 80)
    print("PHASE 7: ADVANCED MACHINE LEARNING MODELING - STAGED EXECUTION")
    print("=" * 80)
    
    # Initialize pipeline
    print("\nSTAGE 1: INITIALIZING PIPELINE")
    print("-" * 40)
    start_time = time.time()
    pipeline = AdvancedModelingPipeline(random_state=42)
    print(f"Pipeline initialized in {time.time() - start_time:.2f}s")
    
    # Load data
    print("\nSTAGE 2: LOADING PROCESSED DATA")
    print("-" * 40)
    stage_start = time.time()
    X_train, y_train, X_test = pipeline.load_processed_data()
    print(f"Data loaded in {time.time() - stage_start:.2f}s")
    
    # Initialize models
    print("\nSTAGE 3: INITIALIZING MODEL PORTFOLIO")
    print("-" * 40)
    stage_start = time.time()
    models = pipeline.initialize_model_portfolio()
    print(f"Models initialized in {time.time() - stage_start:.2f}s")
    
    # Baseline comparison
    print("\nSTAGE 4: BASELINE MODEL COMPARISON")
    print("-" * 40)
    stage_start = time.time()
    baseline_results = pipeline.baseline_model_comparison()
    print(f"Baseline comparison completed in {time.time() - stage_start:.2f}s")
    
    # Hyperparameter optimization
    print("\nSTAGE 5: HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    stage_start = time.time()
    optimization_results = pipeline.hyperparameter_optimization(top_n=5, n_iter=30)
    print(f"Hyperparameter optimization completed in {time.time() - stage_start:.2f}s")
    
    # Train final models
    print("\nSTAGE 6: TRAINING FINAL MODELS")
    print("-" * 40)
    stage_start = time.time()
    final_models = pipeline.train_final_models()
    print(f"Final models trained in {time.time() - stage_start:.2f}s")
    
    # Create ensembles
    print("\nSTAGE 7: CREATING ENSEMBLE METHODS")
    print("-" * 40)
    stage_start = time.time()
    pipeline.create_ensemble_methods()
    print(f"Ensemble methods created in {time.time() - stage_start:.2f}s")
    
    # Comprehensive evaluation
    print("\nSTAGE 8: COMPREHENSIVE MODEL EVALUATION")
    print("-" * 40)
    stage_start = time.time()
    evaluation_results, best_model_name = pipeline.comprehensive_model_evaluation()
    print(f"Model evaluation completed in {time.time() - stage_start:.2f}s")
    
    # Save best model
    print("\nSTAGE 9: SAVING BEST MODEL")
    print("-" * 40)
    stage_start = time.time()
    pipeline.save_best_model()
    print(f"Best model saved in {time.time() - stage_start:.2f}s")
    
    # Generate predictions
    print("\nSTAGE 10: GENERATING PREDICTIONS")
    print("-" * 40)
    stage_start = time.time()
    predictions = pipeline.generate_predictions_and_submission()
    print(f"Predictions generated in {time.time() - stage_start:.2f}s")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PHASE 7 ADVANCED MODELING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total execution time: {total_time/60:.1f} minutes ({total_time:.2f}s)")
    print(f"Best model: {best_model_name}")
    print(f"Best RMSE: {pipeline.best_score:.4f}")
    print(f"Models evaluated: {len(evaluation_results)}")
    print(f"Predictions generated: {len(predictions) if predictions is not None else 0}")
    
    print("\nAll results saved in 'results/' directory:")
    print("  - models/          : Trained models and metadata")
    print("  - metrics/         : Performance metrics and comparisons") 
    print("  - visualizations/  : Charts and analysis plots")
    print("  - submission.csv   : Final predictions for competition")
    
    # Save execution summary
    summary = {
        'execution_time_minutes': total_time / 60,
        'best_model': best_model_name,
        'best_rmse': float(pipeline.best_score),
        'models_evaluated': len(evaluation_results),
        'predictions_generated': len(predictions) if predictions is not None else 0,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/execution_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        'pipeline': pipeline,
        'results': evaluation_results,
        'best_model_name': best_model_name,
        'predictions': predictions
    }

if __name__ == "__main__":
    results = main()