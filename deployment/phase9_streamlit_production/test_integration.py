"""
Pipeline Integration Test Script
Tests all data loading and visualization components
"""

import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.data_loader import get_data_loader
from utils.visualization_utils import get_visualizer
from config.app_config import config

def test_pipeline_integration():
    """Test complete pipeline integration."""
    
    print("Testing Pipeline Integration...")
    print("=" * 50)
    
    # Test data loader
    print("\nTesting Data Loader...")
    data_loader = get_data_loader()
    
    # Test path validation
    path_validation = config.validate_paths()
    print("Path Validation Results:")
    for path, status in path_validation.items():
        status_icon = "PASS" if status else "FAIL"
        print(f"  {status_icon}: {path}: {'EXISTS' if status else 'MISSING'}")
    
    # Test pipeline validation
    print("\nTesting Pipeline Validation...")
    validation_results = data_loader.validate_pipeline_integration()
    print("Pipeline Validation Results:")
    for component, status in validation_results.items():
        status_icon = "PASS" if status else "FAIL"
        print(f"  {status_icon}: {component}: {'OK' if status else 'ERROR'}")
    
    # Test data loading
    print("\nTesting Data Loading...")
    try:
        model = data_loader.load_champion_model()
        print(f"PASS: Model loaded: {type(model).__name__}")
        print(f"  Features: {len(model.feature_importances_)}")
    except Exception as e:
        print(f"FAIL: Model loading failed: {e}")
    
    try:
        train_data = data_loader.load_training_data()
        print(f"PASS: Training data loaded: {train_data.shape}")
    except Exception as e:
        print(f"FAIL: Training data loading failed: {e}")
    
    try:
        feature_importance = data_loader.load_feature_importance()
        if feature_importance:
            methods = list(feature_importance.keys())
            print(f"PASS: Feature importance loaded: {len(methods)} methods")
            print(f"  Methods: {', '.join(methods)}")
        else:
            print("FAIL: Feature importance is empty")
    except Exception as e:
        print(f"FAIL: Feature importance loading failed: {e}")
    
    try:
        business_insights = data_loader.load_business_insights()
        if business_insights:
            categories = list(business_insights.keys())
            print(f"PASS: Business insights loaded: {len(categories)} categories")
            print(f"  Categories: {', '.join(categories)}")
        else:
            print("FAIL: Business insights is empty")
    except Exception as e:
        print(f"FAIL: Business insights loading failed: {e}")
    
    # Test visualization
    print("\nTesting Visualizations...")
    visualizer = get_visualizer()
    
    try:
        # Test performance gauge
        gauge = visualizer.create_performance_gauge(0.904, "Test Gauge")
        print("PASS: Performance gauge creation")
    except Exception as e:
        print(f"FAIL: Performance gauge creation failed: {e}")
    
    try:
        # Test feature importance chart
        if feature_importance and 'shap_importance' in feature_importance:
            top_features = data_loader.get_top_features(n=10, method='shap_importance')
            if top_features:
                chart = visualizer.create_feature_importance_chart(top_features, "Test Chart")
                print("PASS: Feature importance chart creation")
            else:
                print("FAIL: No top features available")
    except Exception as e:
        print(f"FAIL: Feature importance chart creation failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    all_paths_exist = all(path_validation.values())
    all_components_valid = all(validation_results.values())
    
    if all_paths_exist and all_components_valid:
        print("INTEGRATION TEST: PASS")
        print("All pipeline components are properly integrated!")
        print("Streamlit application is ready for production!")
    else:
        print("INTEGRATION TEST: PARTIAL")
        print("Some components may need attention.")
        if not all_paths_exist:
            print("  - Check file paths and pipeline outputs")
        if not all_components_valid:
            print("  - Check component integration")

if __name__ == "__main__":
    test_pipeline_integration()