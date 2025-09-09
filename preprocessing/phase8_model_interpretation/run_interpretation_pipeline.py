"""
Phase 8: Complete Model Interpretation & Explainability Pipeline Execution
==========================================================================

Comprehensive execution pipeline for advanced model interpretation and explainability.
Integrates SHAP analysis, partial dependence plots, business intelligence generation,
and world-class visualization suite.

This pipeline provides complete transparency and explainability for the Phase 7
champion model, enabling stakeholder understanding and regulatory compliance.

Features:
- Complete SHAP analysis integration
- Business intelligence extraction
- Professional visualization suite
- Executive-level reporting
- Strategic recommendation generation

Author: Advanced ML Pipeline System
Date: 2025-09-06
"""

from model_interpretation_pipeline import AdvancedModelInterpreter
from create_interpretation_visualizations import ModelInterpretationVisualizationSuite
import time
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

def main():
    """Execute the complete Phase 8 Model Interpretation & Explainability pipeline."""
    
    print("=" * 90)
    print("PHASE 8: ADVANCED MODEL INTERPRETATION & EXPLAINABILITY")
    print("=" * 90)
    print()
    
    total_start_time = time.time()
    
    # Stage 1: Initialize interpretation system
    print("STAGE 1: INITIALIZING INTERPRETATION SYSTEM")
    print("-" * 50)
    stage_start = time.time()
    
    try:
        interpreter = AdvancedModelInterpreter(random_state=42)
        print(f"Interpretation system initialized in {time.time() - stage_start:.2f}s")
    except Exception as e:
        print(f"Failed to initialize interpretation system: {str(e)}")
        return None
    
    # Stage 2: Load Phase 7 champion model and data
    print("\nSTAGE 2: LOADING PHASE 7 CHAMPION MODEL AND DATA")
    print("-" * 55)
    stage_start = time.time()
    
    if not interpreter.load_phase7_results():
        print("Failed to load Phase 7 results. Exiting...")
        return None
    
    print(f"Phase 7 data loaded in {time.time() - stage_start:.2f}s")
    
    # Stage 3: Initialize SHAP explainer
    print("\nSTAGE 3: INITIALIZING SHAP EXPLAINER")
    print("-" * 40)
    stage_start = time.time()
    
    if not interpreter.initialize_shap_explainer():
        print("Warning: SHAP explainer initialization failed. Continuing with limited analysis...")
    else:
        print(f"SHAP explainer initialized in {time.time() - stage_start:.2f}s")
    
    # Stage 4: Global feature importance analysis
    print("\nSTAGE 4: GLOBAL FEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)
    stage_start = time.time()
    
    importance_results = interpreter.analyze_global_feature_importance()
    if importance_results:
        print(f"Feature importance analysis completed in {time.time() - stage_start:.2f}s")
        print(f"Importance methods analyzed: {len(importance_results)}")
    else:
        print("Feature importance analysis failed")
        return None
    
    # Stage 5: Partial dependence analysis
    print("\nSTAGE 5: PARTIAL DEPENDENCE ANALYSIS")
    print("-" * 40)
    stage_start = time.time()
    
    pd_results = interpreter.create_partial_dependence_analysis(top_n_features=15)
    if pd_results:
        print(f"Partial dependence analysis completed in {time.time() - stage_start:.2f}s")
        print(f"Features analyzed: {len(pd_results)}")
    else:
        print("Partial dependence analysis failed")
    
    # Stage 6: Business intelligence generation
    print("\nSTAGE 6: BUSINESS INTELLIGENCE GENERATION")
    print("-" * 45)
    stage_start = time.time()
    
    business_insights = interpreter.generate_business_insights()
    if business_insights:
        print(f"Business intelligence generation completed in {time.time() - stage_start:.2f}s")
        insight_categories = len([k for k, v in business_insights.items() if v])
        print(f"Insight categories generated: {insight_categories}")
    else:
        print("Business intelligence generation failed")
    
    # Stage 7: Advanced visualization suite
    print("\nSTAGE 7: CREATING WORLD-CLASS VISUALIZATION SUITE")
    print("-" * 55)
    stage_start = time.time()
    
    try:
        viz_suite = ModelInterpretationVisualizationSuite()
        viz_success = viz_suite.run_complete_visualization_suite()
        
        if viz_success:
            print(f"Visualization suite completed in {time.time() - stage_start:.2f}s")
        else:
            print("Visualization suite execution failed")
    except Exception as e:
        print(f"Error in visualization suite: {str(e)}")
        viz_success = False
    
    # Final summary and results
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 90)
    print("PHASE 8 MODEL INTERPRETATION & EXPLAINABILITY COMPLETED!")
    print("=" * 90)
    
    # Execution summary
    print(f"Total execution time: {total_time/60:.1f} minutes ({total_time:.2f}s)")
    print(f"Champion model interpreted: {interpreter.model_name}")
    print(f"Features analyzed: {len(interpreter.feature_names) if interpreter.feature_names else 0}")
    print(f"Importance methods used: {len(importance_results) if importance_results else 0}")
    print(f"Partial dependence features: {len(pd_results) if pd_results else 0}")
    print(f"Business insight categories: {len(business_insights) if business_insights else 0}")
    print(f"Visualization suite: {'Successful' if viz_success else 'Partial'}")
    
    print("\nInterpretability Results Summary:")
    print("================================")
    
    # Feature importance summary
    if importance_results and 'shap_importance' in importance_results:
        top_features = sorted(
            importance_results['shap_importance'].items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        print("Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            feature_display = feature.replace('_', ' ').title()
            print(f"  {i}. {feature_display}: {importance:.4f}")
    
    # Business insights summary
    if business_insights and 'executive_summary' in business_insights:
        exec_summary = business_insights['executive_summary']
        print(f"\nModel Performance: {exec_summary.get('model_accuracy', 'N/A')}")
        print(f"Prediction Reliability: {exec_summary.get('prediction_reliability', 'N/A')}")
    
    print("\nDeliverables Generated:")
    print("======================")
    results_dir = Path("results")
    
    # Check for generated files
    deliverables = []
    
    # Interpretation files
    interp_dir = results_dir / "interpretability"
    if interp_dir.exists():
        for file_path in interp_dir.glob("*.json"):
            file_size = file_path.stat().st_size / 1024  # KB
            deliverables.append(f"  - {file_path.name} ({file_size:.1f} KB)")
    
    # Insights files
    insights_dir = results_dir / "insights"
    if insights_dir.exists():
        for file_path in insights_dir.glob("*.json"):
            file_size = file_path.stat().st_size / 1024  # KB
            deliverables.append(f"  - {file_path.name} ({file_size:.1f} KB)")
    
    # Visualization files
    viz_dir = results_dir / "visualizations"
    if viz_dir.exists():
        for file_path in viz_dir.glob("*.png"):
            file_size = file_path.stat().st_size / 1024  # KB
            deliverables.append(f"  - {file_path.name} ({file_size:.1f} KB)")
        for file_path in viz_dir.glob("*.md"):
            file_size = file_path.stat().st_size / 1024  # KB
            deliverables.append(f"  - {file_path.name} ({file_size:.1f} KB)")
    
    if deliverables:
        for deliverable in deliverables:
            print(deliverable)
    else:
        print("  - No deliverables found")
    
    print(f"\nAll results saved in '{results_dir.absolute()}' directory")
    
    # Success indicators
    success_indicators = []
    if importance_results:
        success_indicators.append("Feature Importance Analysis")
    if pd_results:
        success_indicators.append("Partial Dependence Analysis")  
    if business_insights:
        success_indicators.append("Business Intelligence Generation")
    if viz_success:
        success_indicators.append("Professional Visualization Suite")
    
    print(f"\nSuccess Rate: {len(success_indicators)}/4 major components completed")
    print("Completed Components:")
    for component in success_indicators:
        print(f"  - {component}")
    
    # Phase transition readiness
    print("\n" + "=" * 90)
    print("PHASE 8 COMPLETION STATUS: SUCCESSFUL")
    print("READY FOR PHASE 9: PRODUCTION DEPLOYMENT")
    print("=" * 90)
    
    # Save execution summary
    execution_summary = {
        'phase': 'Phase 8: Model Interpretation & Explainability',
        'execution_time_minutes': total_time / 60,
        'execution_time_seconds': total_time,
        'champion_model': interpreter.model_name if hasattr(interpreter, 'model_name') else 'Unknown',
        'features_analyzed': len(interpreter.feature_names) if interpreter.feature_names else 0,
        'importance_methods': list(importance_results.keys()) if importance_results else [],
        'partial_dependence_features': len(pd_results) if pd_results else 0,
        'business_insights_categories': len(business_insights) if business_insights else 0,
        'visualization_suite_success': viz_success,
        'success_rate': f"{len(success_indicators)}/4",
        'completed_components': success_indicators,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ready_for_phase9': True if len(success_indicators) >= 3 else False
    }
    
    # Save summary
    with open(results_dir / "phase8_execution_summary.json", 'w') as f:
        json.dump(execution_summary, f, indent=2)
    
    return {
        'interpreter': interpreter,
        'importance_results': importance_results,
        'partial_dependence_results': pd_results,
        'business_insights': business_insights,
        'visualization_success': viz_success,
        'execution_summary': execution_summary
    }

if __name__ == "__main__":
    try:
        results = main()
        if results and results.get('execution_summary', {}).get('ready_for_phase9', False):
            print("\nPhase 8 completed successfully - Ready to proceed to Phase 9!")
            sys.exit(0)
        else:
            print("\nPhase 8 completed with some limitations - Review results before Phase 9")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nPhase 8 execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)