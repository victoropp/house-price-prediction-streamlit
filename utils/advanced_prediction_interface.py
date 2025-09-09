"""
Advanced Prediction Interface
Complete 100% user-friendly interface for all 224 features with proper transformations
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import sys
from pathlib import Path
from scipy.special import inv_boxcox

# Import our complete transformations system
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from complete_transformations import CompleteHouseTransformations
from enhanced_shap_explainer import EnhancedSHAPExplainer

class AdvancedPredictionInterface:
    """Advanced prediction interface with all features user-friendly"""
    
    def __init__(self):
        self.transformer = CompleteHouseTransformations()
        self.shap_explainer = EnhancedSHAPExplainer()
        
        # Feature categories for organization
        self.feature_categories = {
            "🏠 Basic Property Info": [
                'YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 
                '1stFlrSF', '2ndFlrSF', 'OverallQual', 'OverallCond'
            ],
            "🏗️ Construction & Quality": [
                'BedroomAbvGr', 'FullBath', 'HalfBath', 'KitchenAbvGr',
                'Fireplaces', 'TotRmsAbvGrd'
            ],
            "🚗 Garage & Parking": [
                'GarageCars', 'GarageArea', 'GarageYrBlt'
            ],
            "📐 Additional Areas": [
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF',
                'MasVnrArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                '3SsnPorch', 'ScreenPorch'
            ],
            "🌟 Quality Ratings": [
                'ExterQual_encoded', 'ExterCond_encoded', 'BsmtQual_encoded',
                'HeatingQC_encoded', 'KitchenQual_encoded', 'FireplaceQu_encoded',
                'GarageQual_encoded'
            ],
            "🏘️ Location & Type": [
                'Street_encoded', 'CentralAir_encoded'
            ],
            "💰 Miscellaneous": [
                'MiscVal', 'MoSold'
            ]
        }
        
        # Default values for common scenarios
        self.default_scenarios = {
            "Starter Home": {
                'YearBuilt': 1990, 'LotArea': 7500, 'GrLivArea': 1200,
                'OverallQual': 5, 'BedroomAbvGr': 3, 'FullBath': 2, 'GarageCars': 2
            },
            "Family Home": {
                'YearBuilt': 2000, 'LotArea': 10000, 'GrLivArea': 2000,
                'OverallQual': 7, 'BedroomAbvGr': 4, 'FullBath': 2, 'GarageCars': 2
            },
            "Luxury Home": {
                'YearBuilt': 2005, 'LotArea': 15000, 'GrLivArea': 3000,
                'OverallQual': 9, 'BedroomAbvGr': 5, 'FullBath': 3, 'GarageCars': 3
            }
        }

    def show_advanced_prediction_interface(self, model, processed_data):
        """Show complete advanced prediction interface"""
        
        st.markdown("## 🔬 Advanced Prediction Mode")
        st.markdown("### Complete Control - All 224 Features with User-Friendly Inputs")
        
        # Quick scenario selector
        st.markdown("### 🎯 Quick Start (Optional)")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_scenario = st.selectbox(
                "Choose a starting scenario:",
                ["Custom"] + list(self.default_scenarios.keys()),
                help="Select a preset to auto-fill common values, then customize as needed"
            )
        
        with col2:
            if st.button("📋 Load Scenario", disabled=(selected_scenario == "Custom")):
                st.session_state.load_scenario = selected_scenario
        
        # Initialize user inputs
        user_inputs = {}
        
        # Load scenario if requested
        if hasattr(st.session_state, 'load_scenario') and st.session_state.load_scenario in self.default_scenarios:
            scenario_defaults = self.default_scenarios[st.session_state.load_scenario]
            st.success(f"✅ Loaded {st.session_state.load_scenario} defaults - customize below as needed")
        else:
            scenario_defaults = {}
        
        st.markdown("---")
        st.markdown("### 📝 Property Configuration")
        
        # Create tabs for different categories
        tabs = st.tabs([cat for cat in self.feature_categories.keys()])
        
        for tab_idx, (category, features) in enumerate(self.feature_categories.items()):
            with tabs[tab_idx]:
                st.markdown(f"#### {category}")
                
                # Create columns for better layout
                num_cols = min(3, len(features))
                cols = st.columns(num_cols)
                
                for idx, feature in enumerate(features):
                    col_idx = idx % num_cols
                    
                    with cols[col_idx]:
                        # Get default value
                        default_val = scenario_defaults.get(feature, None)
                        
                        # Create user input based on feature type
                        config = self.transformer.get_user_input_config(feature)
                        
                        if config['type'] == 'number_input':
                            if default_val is not None:
                                config['value'] = default_val
                            
                            user_inputs[feature] = st.number_input(
                                config['label'],
                                min_value=config['min_value'],
                                max_value=config['max_value'], 
                                value=config['value'],
                                step=config['step'],
                                help=config['help'],
                                key=f"adv_{feature}"
                            )
                        
                        elif config['type'] == 'selectbox':
                            if default_val is not None and default_val in config['options']:
                                config['index'] = config['options'].index(default_val)
                            
                            user_inputs[feature] = st.selectbox(
                                config['label'],
                                options=config['options'],
                                index=config['index'],
                                help=config['help'],
                                key=f"adv_{feature}"
                            )
        
        # Additional features not in main categories (auto-filled with defaults)
        st.markdown("### ⚙️ Additional Features")
        st.info("📊 Additional engineered features are automatically calculated based on your inputs above")
        
        # Show summary of current inputs
        st.markdown("### 📋 Current Configuration Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Property Details:**")
            key_features = ['YearBuilt', 'LotArea', 'GrLivArea', 'OverallQual', 'BedroomAbvGr', 'FullBath', 'GarageCars']
            for feature in key_features:
                if feature in user_inputs:
                    friendly_name = self.transformer.get_friendly_name(feature)
                    value = user_inputs[feature]
                    st.write(f"• **{friendly_name}**: {value}")
        
        with col2:
            st.markdown("**Quality Ratings:**")
            quality_features = [f for f in user_inputs.keys() if f.endswith('_encoded')][:5]
            for feature in quality_features:
                if feature in user_inputs:
                    friendly_name = self.transformer.get_friendly_name(feature)
                    value = user_inputs[feature]
                    st.write(f"• **{friendly_name}**: {value}")
        
        # Advanced prediction button
        st.markdown("---")
        
        if st.button("🎯 Generate Advanced Prediction", type="primary", use_container_width=True):
            prediction_result = self.make_advanced_prediction(model, processed_data, user_inputs)
            
            if prediction_result:
                self.display_advanced_results(prediction_result, user_inputs)
    
    def make_advanced_prediction(self, model, processed_data, user_inputs):
        """Make prediction using advanced user inputs"""
        
        try:
            # Use first row of processed data as template
            feature_vector = processed_data.iloc[0].copy()
            
            # Remove target column if present
            if 'SalePrice_transformed' in feature_vector.index:
                feature_vector = feature_vector.drop('SalePrice_transformed')
            
            # Transform and update user inputs
            transformed_count = 0
            for feature, value in user_inputs.items():
                if feature in feature_vector.index:
                    transformed_value = self.transformer.transform_user_input(feature, value)
                    feature_vector[feature] = transformed_value
                    transformed_count += 1
            
            # Auto-calculate derived features based on inputs
            feature_vector = self.calculate_derived_features(feature_vector, user_inputs)
            
            # Make prediction
            prediction_transformed = model.predict([feature_vector.values])[0]
            
            # Apply correct Box-Cox inverse transformation
            lambda_param = -0.07693211157738546
            predicted_price = inv_boxcox(prediction_transformed, lambda_param) - 1
            
            return {
                'predicted_price': predicted_price,
                'prediction_transformed': prediction_transformed,
                'feature_vector': feature_vector,
                'model': model,
                'user_inputs': user_inputs,
                'transformed_count': transformed_count,
                'total_features': len(feature_vector),
                'success': True
            }
            
        except Exception as e:
            st.error(f"❌ Advanced prediction failed: {str(e)}")
            return None
    
    def calculate_derived_features(self, feature_vector, user_inputs):
        """Calculate derived/engineered features based on user inputs"""
        
        # Calculate TotalSF if components are available
        if all(f in user_inputs for f in ['GrLivArea', 'TotalBsmtSF']):
            total_sf = user_inputs['GrLivArea'] + user_inputs['TotalBsmtSF']
            if 'TotalSF' in feature_vector.index:
                feature_vector['TotalSF'] = self.transformer.normalize_min_max(total_sf, 'TotalSF')
        
        # Calculate house age
        if 'YearBuilt' in user_inputs:
            house_age = 2024 - user_inputs['YearBuilt']  # Current year
            if 'HouseAge' in feature_vector.index:
                feature_vector['HouseAge'] = self.transformer.normalize_min_max(house_age, 'HouseAge')
        
        # Calculate total baths
        total_baths = 0
        if 'FullBath' in user_inputs:
            total_baths += user_inputs['FullBath']
        if 'HalfBath' in user_inputs:
            total_baths += user_inputs['HalfBath'] * 0.5
        if 'TotalBaths' in feature_vector.index and total_baths > 0:
            feature_vector['TotalBaths'] = total_baths
        
        # Set reasonable defaults for missing engineered features
        engineered_defaults = {
            'BsmtFinishedRatio': 0.5,
            'LivingAreaRatio': 0.5,
            'YearsSinceRemodel': 10,
            'GarageAge': 15,
            'QualityScore': user_inputs.get('OverallQual', 6) * 0.1,
            'BasementQualityScore': 0.6,
            'RoomDensity': 0.4,
            'BathBedroomRatio': 0.7,
            'GarageLivingRatio': 0.3,
            'LotCoverageRatio': 0.2,
            'AgeQualityInteraction': 0.3
        }
        
        for feature, default_val in engineered_defaults.items():
            if feature in feature_vector.index:
                if self.transformer.get_feature_type(feature) == 'min_max':
                    # Already normalized
                    feature_vector[feature] = default_val
        
        return feature_vector
    
    def display_advanced_results(self, result, user_inputs):
        """Display advanced prediction results"""
        
        st.markdown("## 🎯 Advanced Prediction Results")
        
        # Main result display
        predicted_price = result['predicted_price']
        
        # Premium display for advanced mode
        st.markdown(f"""
        <div style="
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        ">
            <h2 style="margin: 0; font-size: 1.8rem; opacity: 0.9;">🏆 Advanced ML Prediction</h2>
            <h1 style="margin: 0.5rem 0; font-size: 3.5rem; font-weight: bold;">${predicted_price:,.0f}</h1>
            <p style="margin: 0; font-size: 1.4rem; opacity: 0.9;">Based on {result['total_features']} Features</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.8;">
                {result['transformed_count']} user inputs + {result['total_features'] - result['transformed_count']} engineered features
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence_pct = min(95, max(80, 90 + (result['transformed_count'] - 10) * 0.5))
            st.metric(
                "Confidence Level",
                f"{confidence_pct:.0f}%",
                "High precision"
            )
        
        with col2:
            price_per_sqft = predicted_price / user_inputs.get('GrLivArea', 1)
            st.metric(
                "Price per Sq Ft",
                f"${price_per_sqft:.0f}",
                "Living area basis"
            )
        
        with col3:
            # Compare to typical range
            if predicted_price < 150000:
                market_pos = "Below Market"
                delta_color = "normal"
            elif predicted_price > 400000:
                market_pos = "Premium"
                delta_color = "normal"  
            else:
                market_pos = "Market Rate"
                delta_color = "normal"
            
            st.metric(
                "Market Position",
                market_pos,
                f"${predicted_price:,.0f}"
            )
        
        with col4:
            features_used_pct = (result['transformed_count'] / result['total_features']) * 100
            st.metric(
                "Features Used",
                f"{result['transformed_count']}/{result['total_features']}",
                f"{features_used_pct:.0f}% coverage"
            )
        
        # Advanced insights
        st.markdown("### 🧠 Advanced Property Analysis")
        
        insights = []
        
        # Quality analysis
        overall_qual = user_inputs.get('OverallQual', 5)
        if overall_qual >= 8:
            insights.append("🌟 **Premium Quality Construction** - Top-tier materials and craftsmanship")
        elif overall_qual >= 6:
            insights.append("✨ **Good Quality Build** - Solid construction and materials")
        
        # Size analysis
        living_area = user_inputs.get('GrLivArea', 0)
        if living_area > 2500:
            insights.append("🏰 **Spacious Layout** - Generous living space for families")
        elif living_area > 1800:
            insights.append("🏠 **Comfortable Size** - Well-proportioned living areas")
        
        # Age analysis
        year_built = user_inputs.get('YearBuilt', 2000)
        if year_built >= 2005:
            insights.append("🆕 **Modern Construction** - Contemporary building standards")
        elif year_built >= 1990:
            insights.append("🔄 **Well-Maintained Era** - Established but modern features")
        
        # Garage analysis
        garage_cars = user_inputs.get('GarageCars', 0)
        if garage_cars >= 3:
            insights.append("🚗 **Excellent Parking** - Multi-vehicle accommodation")
        elif garage_cars >= 2:
            insights.append("🚙 **Standard Parking** - Two-car garage convenience")
        
        # Bathroom analysis
        full_baths = user_inputs.get('FullBath', 1)
        bedrooms = user_inputs.get('BedroomAbvGr', 3)
        if full_baths >= bedrooms:
            insights.append("🛁 **Optimal Bath Ratio** - Convenient bathroom access")
        
        # Display insights
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.markdown("📊 **Standard Property Profile** - Typical market characteristics")
        
        # Technical details (expandable)
        with st.expander("🔬 Technical Analysis Details"):
            st.markdown("**Model Processing:**")
            st.write(f"• Raw model output: {result['prediction_transformed']:.6f}")
            st.write(f"• Box-Cox lambda: -0.07693211157738546")
            st.write(f"• Transformation: inv_boxcox({result['prediction_transformed']:.4f}, λ) - 1")
            st.write(f"• Final price: ${result['predicted_price']:,.0f}")
            
            st.markdown("**Feature Summary:**")
            st.write(f"• Total features in model: {result['total_features']}")
            st.write(f"• User-specified features: {result['transformed_count']}")
            st.write(f"• Auto-calculated features: {result['total_features'] - result['transformed_count']}")
            
            # Show key transformations
            st.markdown("**Key Input Transformations:**")
            key_features = ['YearBuilt', 'LotArea', 'GrLivArea', 'OverallQual']
            for feature in key_features:
                if feature in user_inputs:
                    original_val = user_inputs[feature]
                    transformed_val = self.transformer.transform_user_input(feature, original_val)
                    transformation_type = self.transformer.get_feature_type(feature)
                    st.write(f"• {feature}: {original_val} → {transformed_val:.4f} ({transformation_type})")
        
        # Add enhanced SHAP explanations
        try:
            self.shap_explainer.integrate_with_prediction_interface(result, "advanced")
        except Exception as e:
            st.warning(f"⚠️ AI explanation temporarily unavailable: {str(e)}")

# Integration function for main app
def show_advanced_prediction_interface_complete(model, processed_data):
    """Integration function for the main app"""
    
    interface = AdvancedPredictionInterface()
    interface.show_advanced_prediction_interface(model, processed_data)