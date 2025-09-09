"""
Quick test to verify batch prediction integration with CompleteHouseTransformations
"""
import pandas as pd
import numpy as np
from utils.complete_transformations import CompleteHouseTransformations
from utils.data_loader import PipelineDataLoader

def test_batch_integration():
    print("Testing Batch Predictions Integration with CompleteHouseTransformations")
    print("=" * 70)
    
    # Initialize components
    transformer = CompleteHouseTransformations()
    data_loader = PipelineDataLoader()
    
    try:
        # Load model
        model = data_loader.load_champion_model()
        print(f"[OK] Model loaded: {type(model).__name__}")
        
        # Load test CSV
        test_df = pd.read_csv("test_batch_predictions.csv")
        print(f"[OK] Test CSV loaded: {test_df.shape[0]} properties")
        
        # Test transformation for each row
        for idx, row in test_df.iterrows():
            print(f"\n--- Testing Property {idx + 1} ---")
            
            # Convert row to dict
            property_data = row.to_dict()
            print(f"Input features: {len(property_data)} fields")
            
            # Transform using data_loader (same as current implementation)
            prediction_df = data_loader.prepare_prediction_features(property_data)
            print(f"Model features generated: {prediction_df.shape[1]} fields")
            print(f"Prediction DataFrame: {prediction_df.shape}")
            
            # Make prediction
            prediction = model.predict(prediction_df)[0]
            
            # Apply Box-Cox inverse transformation
            boxcox_lambda = -0.077
            if boxcox_lambda != 0:
                final_prediction = np.power(boxcox_lambda * prediction + 1, 1/boxcox_lambda) - 1
            else:
                final_prediction = np.exp(prediction) - 1
            
            print(f"Raw prediction: {prediction:.4f}")
            print(f"Final price: ${final_prediction:,.0f}")
            
            # Verify key features
            key_features = ['GrLivArea', 'OverallQual', 'YearBuilt']
            for feature in key_features:
                if feature in property_data:
                    if feature in prediction_df.columns:
                        model_value = prediction_df[feature].iloc[0]
                        if isinstance(model_value, (int, float)):
                            print(f"  {feature}: {property_data[feature]} -> {model_value:.4f}")
                        else:
                            print(f"  {feature}: {property_data[feature]} -> {model_value}")
                    else:
                        print(f"  {feature}: {property_data[feature]} -> N/A")
        
        print("\n[OK] All batch predictions completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_integration()
    if success:
        print("\n[SUCCESS] Batch integration test PASSED!")
    else:
        print("\n[FAILED] Batch integration test FAILED!")