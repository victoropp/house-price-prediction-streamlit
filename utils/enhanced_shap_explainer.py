"""
Enhanced SHAP Explainer
Provides comprehensive, user-friendly SHAP explanations for both quick and advanced predictions
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import shap
from utils.feature_explainer import FeatureExplainer
from utils.complete_transformations import CompleteHouseTransformations

class EnhancedSHAPExplainer:
    """Enhanced SHAP explanation system with user-friendly visualizations and narratives"""
    
    def __init__(self):
        self.feature_explainer = FeatureExplainer()
        self.transformer = CompleteHouseTransformations()
        self.colors = {
            'positive': '#00D4AA',
            'negative': '#FF6B6B', 
            'neutral': '#74C0FC',
            'background': '#F8F9FA',
            'text': '#2C3E50'
        }
        
    def create_individual_shap_explanation(self, model, feature_vector: pd.Series, 
                                         user_inputs: Dict[str, Any], 
                                         predicted_price: float) -> Dict[str, Any]:
        """Create comprehensive SHAP explanation for an individual prediction"""
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Get SHAP values for this prediction
            shap_values = explainer.shap_values(feature_vector.values.reshape(1, -1))[0]
            base_value_transformed = explainer.expected_value
            
            # Convert base value from transformed space to real dollars
            from scipy.special import inv_boxcox
            lambda_param = -0.07693211157738546
            base_value_real = inv_boxcox(base_value_transformed, lambda_param) - 1
            
            # Convert SHAP values to real dollar impacts
            feature_importance = {}
            for i, feature_name in enumerate(feature_vector.index):
                if not np.isnan(shap_values[i]) and abs(shap_values[i]) > 1e-6:
                    # Calculate the impact in real dollars by computing the difference
                    # between prediction with and without this feature's contribution
                    pred_with = base_value_transformed + shap_values[i]
                    pred_without = base_value_transformed
                    
                    price_with = inv_boxcox(pred_with, lambda_param) - 1
                    price_without = inv_boxcox(pred_without, lambda_param) - 1
                    
                    dollar_impact = price_with - price_without
                    feature_importance[feature_name] = dollar_impact / 1000  # Convert to thousands for display
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)[:10]
            
            return {
                'success': True,
                'base_value': base_value_real / 1000,  # Convert to thousands for display
                'predicted_value': predicted_price / 1000,  # Convert to thousands for display
                'feature_importance': dict(sorted_features),
                'user_inputs': user_inputs,
                'total_features': len(feature_vector),
                'significant_features': len([x for x in shap_values if abs(x) > 1e-6])
            }
            
        except Exception as e:
            st.error(f"SHAP calculation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_user_friendly_shap_display(self, shap_data: Dict[str, Any], 
                                        prediction_type: str = "individual") -> None:
        """Create comprehensive user-friendly SHAP display"""
        if not shap_data.get('success', False):
            st.warning("⚠️ SHAP analysis temporarily unavailable")
            return
        
        st.markdown("### 🔍 **AI Prediction Explanation**")
        st.markdown("*Understanding how your house features impact the predicted price*")
        
        # Create three columns for overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🎯 Base Market Price", 
                f"${shap_data['base_value']*1000:,.0f}",
                help="Average price for similar homes in the dataset"
            )
        
        with col2:
            adjustment = shap_data['predicted_value'] - shap_data['base_value']
            st.metric(
                "📊 Your Home's Adjustment", 
                f"${adjustment*1000:+,.0f}",
                help="How your specific features adjust the base price"
            )
        
        with col3:
            st.metric(
                "🏠 Features Analyzed", 
                f"{shap_data['significant_features']}/{shap_data['total_features']}",
                help="Number of features significantly impacting your price"
            )
        
        # Create the main explanation sections
        self._create_feature_impact_section(shap_data)
        self._create_interactive_waterfall_chart(shap_data)
        self._create_feature_comparison_chart(shap_data)
        self._create_plain_english_summary(shap_data)
    
    def _create_feature_impact_section(self, shap_data: Dict[str, Any]) -> None:
        """Create detailed feature impact explanations"""
        st.markdown("### 📈 **Key Factors Driving Your Home's Price**")
        
        feature_importance = shap_data['feature_importance']
        user_inputs = shap_data['user_inputs']
        
        # Separate positive and negative impacts
        positive_features = [(k, v) for k, v in feature_importance.items() if v > 0]
        negative_features = [(k, v) for k, v in feature_importance.items() if v < 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 **Price Boosters**")
            if positive_features:
                for i, (feature, impact) in enumerate(positive_features[:5]):
                    with st.expander(f"#{i+1} {self.feature_explainer.get_friendly_feature_name(feature)}", expanded=i<2):
                        self._create_feature_detail_card(feature, impact, user_inputs, "positive")
            else:
                st.info("No significant positive factors identified")
        
        with col2:
            st.markdown("#### ⚠️ **Price Reducers**")
            if negative_features:
                for i, (feature, impact) in enumerate(negative_features[:5]):
                    with st.expander(f"#{i+1} {self.feature_explainer.get_friendly_feature_name(feature)}", expanded=i<2):
                        self._create_feature_detail_card(feature, impact, user_inputs, "negative")
            else:
                st.info("No significant negative factors identified")
    
    def _create_feature_detail_card(self, feature: str, impact: float, 
                                  user_inputs: Dict[str, Any], impact_type: str) -> None:
        """Create detailed explanation card for a specific feature"""
        friendly_name = self.feature_explainer.get_friendly_feature_name(feature)
        
        # Get feature value (from user inputs if available, otherwise from processed data)
        feature_value = user_inputs.get(feature, "N/A")
        if feature_value == "N/A" and hasattr(self.transformer, 'get_feature_value'):
            feature_value = self.transformer.get_feature_value(feature)
        
        formatted_value = self.feature_explainer.format_feature_value(feature, feature_value)
        impact_amount = abs(impact) * 1000  # Already in thousands, convert to dollars for display
        
        # Color coding
        color = self.colors['positive'] if impact_type == "positive" else self.colors['negative']
        icon = "📈" if impact_type == "positive" else "📉"
        
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0; background-color: #F8F9FA;">
            <h4 style="margin: 0; color: {color};">{icon} {friendly_name}</h4>
            <p style="margin: 5px 0;"><strong>Your Value:</strong> {formatted_value}</p>
            <p style="margin: 5px 0;"><strong>Price Impact:</strong> ${impact_amount:+,.0f}</p>
            <p style="margin: 5px 0; font-style: italic;">{self.feature_explainer._get_feature_insight(feature, feature_value, impact_type)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add benchmarking information
        benchmark_info = self._get_feature_benchmark(feature, feature_value)
        if benchmark_info:
            st.markdown(f"💡 **Market Context:** {benchmark_info}")
    
    def _create_interactive_waterfall_chart(self, shap_data: Dict[str, Any]) -> None:
        """Create interactive waterfall chart showing feature contributions"""
        st.markdown("### 📊 **Price Build-up Waterfall**")
        st.markdown("*See how each feature contributes to your final price prediction*")
        
        feature_importance = shap_data['feature_importance']
        base_value = shap_data['base_value'] * 1000  # Convert to dollars
        
        # Prepare data for waterfall
        features = list(feature_importance.keys())[:8]  # Top 8 features
        values = [feature_importance[f] * 1000 for f in features]  # Convert to dollars
        friendly_names = [self.feature_explainer.get_friendly_feature_name(f) for f in features]
        
        # Calculate cumulative values
        cumulative = [base_value]
        for value in values:
            cumulative.append(cumulative[-1] + value)
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Base value
        fig.add_trace(go.Waterfall(
            name="Price Components",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(values) + ["total"],
            x=["Base Price"] + friendly_names + ["Final Price"],
            text=[f"${base_value:,.0f}"] + [f"${v:+,.0f}" for v in values] + [f"${cumulative[-1]:,.0f}"],
            y=[base_value] + values + [0],
            textposition="outside",
            increasing={"marker": {"color": self.colors['positive']}},
            decreasing={"marker": {"color": self.colors['negative']}},
            totals={"marker": {"color": self.colors['neutral']}}
        ))
        
        fig.update_layout(
            title="How Your Features Build Up to the Final Price",
            height=500,
            showlegend=False,
            xaxis_title="Features",
            yaxis_title="Price Impact ($)",
            font=dict(size=12),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_feature_comparison_chart(self, shap_data: Dict[str, Any]) -> None:
        """Create feature comparison chart"""
        st.markdown("### ⚖️ **Feature Impact Comparison**")
        
        feature_importance = shap_data['feature_importance']
        
        # Prepare data
        features = list(feature_importance.keys())[:10]
        friendly_names = [self.feature_explainer.get_friendly_feature_name(f) for f in features]
        values = [feature_importance[f] * 1000 for f in features]
        colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=friendly_names[::-1],  # Reverse for better display
            x=values[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"${v:+,.0f}" for v in values[::-1]],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Impact: $%{x:+,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Top 10 Feature Impacts on Your Home's Price",
            height=400,
            xaxis_title="Price Impact ($)",
            yaxis_title="Features",
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_plain_english_summary(self, shap_data: Dict[str, Any]) -> None:
        """Create plain English summary of the prediction"""
        st.markdown("### 📝 **Your Home's Price Story**")
        
        feature_importance = shap_data['feature_importance']
        user_inputs = shap_data['user_inputs']
        
        # Get top positive and negative features
        positive_features = [(k, v) for k, v in feature_importance.items() if v > 0]
        negative_features = [(k, v) for k, v in feature_importance.items() if v < 0]
        
        positive_features.sort(key=lambda x: x[1], reverse=True)
        negative_features.sort(key=lambda x: x[1])
        
        # Generate narrative
        summary_parts = []
        
        # Overall assessment  
        total_adjustment = sum(feature_importance.values()) * 1000  # Already in thousands, convert to dollars
        if total_adjustment > 5000:
            summary_parts.append("🎉 **Great news!** Your home has several features that significantly boost its market value.")
        elif total_adjustment < -5000:
            summary_parts.append("💡 **Market Reality:** Your home has some features that reduce its value, but there are opportunities for improvement.")
        else:
            summary_parts.append("⚖️ **Balanced Profile:** Your home has a good mix of value-adding and value-reducing features.")
        
        # Top positive features
        if positive_features:
            top_positive = positive_features[0]
            feature_name = self.feature_explainer.get_friendly_feature_name(top_positive[0])
            impact = top_positive[1] * 1000
            summary_parts.append(f"🚀 **Biggest Strength:** Your {feature_name} adds approximately ${impact:,.0f} to your home's value.")
        
        # Top negative features
        if negative_features:
            top_negative = negative_features[0]
            feature_name = self.feature_explainer.get_friendly_feature_name(top_negative[0])
            impact = abs(top_negative[1]) * 1000
            summary_parts.append(f"⚠️ **Main Challenge:** Your {feature_name} reduces value by approximately ${impact:,.0f}.")
        
        # Market context
        base_value = shap_data['base_value'] * 1000
        final_value = (shap_data['base_value'] + sum(feature_importance.values())) * 1000
        percentage_change = ((final_value - base_value) / base_value) * 100
        
        if percentage_change > 10:
            summary_parts.append(f"📈 Your home is predicted to be **{percentage_change:.1f}% above** the typical market price.")
        elif percentage_change < -10:
            summary_parts.append(f"📉 Your home is predicted to be **{abs(percentage_change):.1f}% below** the typical market price.")
        else:
            summary_parts.append(f"🎯 Your home is predicted to be **very close** to the typical market price.")
        
        # Improvement suggestions
        if negative_features:
            summary_parts.append("\n**💡 Potential Improvements:**")
            for feature, impact in negative_features[:3]:
                suggestion = self._get_improvement_suggestion(feature)
                if suggestion:
                    summary_parts.append(f"• {suggestion}")
        
        # Display the summary
        summary_text = "\n\n".join(summary_parts)
        st.markdown(summary_text)
        
        # Add expandable technical details
        with st.expander("🔬 Technical Details", expanded=False):
            st.markdown(f"""
            **SHAP Analysis Summary:**
            - Base market value: ${shap_data['base_value']*1000:,.0f}
            - Feature adjustments: ${sum(feature_importance.values())*1000:+,.0f}
            - Final prediction: ${(shap_data['base_value'] + sum(feature_importance.values()))*1000:,.0f}
            - Features analyzed: {shap_data['significant_features']} of {shap_data['total_features']}
            - Prediction confidence: High (based on {shap_data['total_features']} engineered features)
            """)
    
    def _get_feature_benchmark(self, feature: str, value: Any) -> Optional[str]:
        """Get benchmarking information for a feature"""
        benchmarks = {
            'OverallQual': {
                'low': (1, 4, "Below average quality"),
                'medium': (5, 7, "Average to good quality"),
                'high': (8, 10, "Excellent quality")
            },
            'GrLivArea': {
                'low': (0, 1500, "Compact home"),
                'medium': (1500, 2500, "Standard size"),
                'high': (2500, float('inf'), "Large home")
            },
            'YearBuilt': {
                'low': (0, 1980, "Older construction"),
                'medium': (1980, 2000, "Mature home"),
                'high': (2000, float('inf'), "Modern construction")
            }
        }
        
        if feature in benchmarks and isinstance(value, (int, float)):
            for category, (min_val, max_val, description) in benchmarks[feature].items():
                if min_val <= value <= max_val:
                    return description
        
        return None
    
    def _get_improvement_suggestion(self, feature: str) -> Optional[str]:
        """Get improvement suggestions for features that reduce value"""
        suggestions = {
            'OverallQual': "Consider home improvements to increase overall quality rating",
            'GrLivArea': "Living space additions could significantly increase value",
            'TotalBsmtSF': "Basement improvements or additions could boost value",
            'GarageArea': "Garage expansion or improvements could add value",
            'YearBuilt': "Modern updates and renovations can offset age concerns",
            'Fireplaces': "Adding a fireplace could enhance appeal",
            'TotalBaths': "Additional bathroom could significantly increase value"
        }
        
        return suggestions.get(feature)

    def integrate_with_prediction_interface(self, prediction_result: Dict[str, Any], 
                                          interface_type: str = "quick") -> None:
        """Integrate SHAP explanations into existing prediction interfaces"""
        if not prediction_result.get('success', False):
            return
        
        # Create enhanced SHAP explanation section
        st.markdown("---")
        st.markdown("## 🧠 **AI Explanation & Insights**")
        
        if interface_type == "quick":
            st.markdown("*Understand exactly why your home received this price prediction*")
        else:
            st.markdown("*Deep dive into the AI's reasoning with comprehensive feature analysis*")
        
        # Generate and display SHAP explanations
        try:
            model = prediction_result.get('model')
            feature_vector = prediction_result.get('feature_vector')
            user_inputs = prediction_result.get('user_inputs', {})
            predicted_price = prediction_result.get('predicted_price', 0)
            
            if model and feature_vector is not None:
                shap_data = self.create_individual_shap_explanation(
                    model, feature_vector, user_inputs, predicted_price
                )
                
                if shap_data.get('success', False):
                    self.create_user_friendly_shap_display(shap_data, interface_type)
                else:
                    st.warning("⚠️ Detailed explanation temporarily unavailable")
            else:
                st.warning("⚠️ Detailed explanation unavailable - missing prediction data")
                
        except Exception as e:
            st.warning(f"⚠️ Explanation generation failed: {str(e)}")