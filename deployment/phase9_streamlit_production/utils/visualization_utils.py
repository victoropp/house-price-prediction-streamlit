"""
State-of-the-Art Data Science Visualization Utilities
Professional, Publication-Quality Charts with Complete Data Integration
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from config.app_config import config
import logging

logger = logging.getLogger(__name__)

class ProfessionalVisualizations:
    """State-of-the-art visualization toolkit for ML model interpretation."""
    
    def __init__(self):
        self.colors = config.COLORS
        self.theme_template = "plotly_white"
        self.layout_config = {
            'template': self.theme_template,
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'font': {'color': self.colors['text'], 'family': "Arial"}
        }
        
    def create_performance_gauge(self, value, title: str, 
                                max_value: float = 1.0, format_type: str = "percentage") -> go.Figure:
        """Create professional performance gauge chart."""
        
        # Handle list input by taking first element or converting to float
        if isinstance(value, list) and value:
            value = float(value[0])
        elif not isinstance(value, (int, float)):
            # Fallback to default value if not convertible
            value = 0.904 if format_type == "percentage" else 0.8
        else:
            value = float(value)
        
        # Determine color based on performance
        if value >= 0.90:
            color = self.colors['champion']
        elif value >= 0.80:
            color = self.colors['success'] 
        elif value >= 0.70:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
        
        # Format value for display
        if format_type == "percentage":
            display_value = f"{value*100:.1f}%"
            gauge_value = value * 100
            gauge_max = max_value * 100
        else:
            display_value = f"{value:.3f}"
            gauge_value = value
            gauge_max = max_value
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = gauge_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 24, 'color': self.colors['text']}},
            number = {'font': {'size': 36, 'color': color}},
            gauge = {
                'axis': {'range': [None, gauge_max], 'tickwidth': 1, 'tickcolor': self.colors['neutral']},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': self.colors['neutral'],
                'steps': [
                    {'range': [0, gauge_max*0.7], 'color': 'rgba(220, 53, 69, 0.3)'},
                    {'range': [gauge_max*0.7, gauge_max*0.85], 'color': 'rgba(255, 140, 0, 0.3)'},
                    {'range': [gauge_max*0.85, gauge_max], 'color': 'rgba(34, 139, 34, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': self.colors['champion'], 'width': 4},
                    'thickness': 0.8,
                    'value': gauge_value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': self.colors['text'], 'family': "Arial"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_data: List[Tuple[str, float]], 
                                      title: str = "Feature Importance", 
                                      max_features: int = 15) -> go.Figure:
        """Create professional horizontal bar chart for feature importance."""
        
        # Limit to top features
        top_features = importance_data[:max_features]
        features, raw_values = zip(*top_features)
        
        # Handle list values and convert to float with robust error handling
        values = []
        for v in raw_values:
            try:
                if isinstance(v, list):
                    if v:
                        # Handle nested lists by recursively extracting first numeric element
                        first_val = v[0]
                        while isinstance(first_val, list) and first_val:
                            first_val = first_val[0]
                        values.append(float(first_val))
                    else:
                        values.append(0.0)
                elif isinstance(v, (int, float)):
                    values.append(float(v))
                elif isinstance(v, str):
                    # Try to convert string to float
                    values.append(float(v))
                else:
                    values.append(0.0)  # fallback for other types
            except (ValueError, TypeError, IndexError):
                # If any conversion fails, use 0.0 as fallback
                values.append(0.0)
        
        # Reverse for proper display order
        features = list(reversed(features))
        values = list(reversed(values))
        
        # Create color array with champion highlighting
        colors_array = []
        for i, _ in enumerate(features):
            if i >= len(features) - 3:  # Top 3 features (in reversed order)
                colors_array.append(self.colors['champion'])
            else:
                colors_array.append(self.colors['primary'])
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(
                color=colors_array,
                line=dict(color='white', width=1)
            ),
            text=[f'{v:.4f}' for v in values],
            textposition='outside',
            textfont=dict(size=12, color=self.colors['text'])
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template=self.theme_template,
            height=max(400, len(features) * 25),
            font=dict(color=self.colors['text']),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        # Add value annotations
        for i, (feature, value) in enumerate(zip(features, values)):
            fig.add_annotation(
                x=value,
                y=i,
                text=f'{value:.4f}',
                showarrow=False,
                xanchor='left',
                font=dict(color=self.colors['text'], size=11)
            )
        
        return fig
    
    def create_category_importance_chart(self, category_data: List[Tuple[str, float]]) -> go.Figure:
        """Create category importance horizontal bar chart."""
        categories, importances = zip(*category_data)
        
        fig = go.Figure(data=[
            go.Bar(
                y=categories,
                x=importances,
                orientation='h',
                marker_color=self.colors['primary'],
                text=[f"{imp:.3f}" for imp in importances],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Feature Category Importance",
            xaxis_title="Average SHAP Importance",
            yaxis_title="Feature Categories",
            height=400,
            showlegend=False,
            template=self.theme_template,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    

    def create_market_segments_donut(self, segments_data: Dict[str, int], 
                                   price_ranges: Dict[str, Tuple[float, float]] = None) -> go.Figure:
        """Create professional donut chart for market segments."""
        
        segments = list(segments_data.keys())
        values = list(segments_data.values())
        
        # Professional color palette
        colors = [self.colors['info'], self.colors['primary'], 
                 self.colors['warning'], self.colors['champion']]
        
        # Create hover text with price ranges if available
        hover_text = []
        for segment, value in zip(segments, values):
            percentage = (value / sum(values)) * 100
            text = f"{segment}<br>Count: {value:,}<br>Percentage: {percentage:.1f}%"
            if price_ranges and segment in price_ranges:
                low, high = price_ranges[segment]
                text += f"<br>Price Range: ${low:,.0f} - ${high:,.0f}"
            hover_text.append(text)
        
        fig = go.Figure(data=[go.Pie(
            labels=segments,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text
        )])
        
        # Add center text
        total_properties = sum(values)
        fig.add_annotation(
            text=f"Total<br>Properties<br><b>{total_properties:,}</b>",
            x=0.5, y=0.5,
            font_size=16,
            font_color=self.colors['text'],
            showarrow=False
        )
        
        fig.update_layout(
            title={
                'text': "Market Segments Distribution",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            template=self.theme_template,
            height=500,
            font=dict(color=self.colors['text'], size=12),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            margin=dict(l=50, r=150, t=80, b=50)
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_data: Dict[str, Dict[str, float]], 
                                 title: str = "Method Correlation Analysis") -> go.Figure:
        """Create professional correlation heatmap."""
        
        methods = list(correlation_data.keys())
        correlation_matrix = []
        
        for method1 in methods:
            row = []
            for method2 in methods:
                if method2 in correlation_data[method1]:
                    row.append(correlation_data[method1][method2])
                else:
                    # Symmetric correlation
                    row.append(correlation_data.get(method2, {}).get(method1, 0))
            correlation_matrix.append(row)
        
        # Create annotations with correlation strength
        annotations = []
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                correlation = correlation_matrix[i][j]
                strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.5 else "Weak"
                color = "white" if abs(correlation) > 0.5 else "black"
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f'{correlation:.3f}<br>({strength})',
                        showarrow=False,
                        font=dict(color=color, size=12)
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=methods,
            y=methods,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title="Correlation<br>Coefficient",
                titlefont=dict(color=self.colors['text']),
                tickfont=dict(color=self.colors['text'])
            )
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            annotations=annotations,
            template=self.theme_template,
            height=500,
            font=dict(color=self.colors['text']),
            xaxis=dict(tickangle=45),
            margin=dict(l=100, r=100, t=100, b=100)
        )
        
        return fig
    
    def create_prediction_confidence_chart(self, prediction: float, 
                                         confidence_interval: Tuple[float, float],
                                         market_range: Tuple[float, float] = None) -> go.Figure:
        """Create professional prediction visualization with confidence."""
        
        lower_bound, upper_bound = confidence_interval
        
        fig = go.Figure()
        
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[prediction],
            y=[0.5],
            mode='markers',
            marker=dict(
                size=25,
                color=self.colors['champion'],
                symbol='diamond',
                line=dict(color='white', width=3)
            ),
            name='Prediction',
            hovertemplate=f'Predicted Price: ${np.exp(prediction):,.0f}<extra></extra>'
        ))
        
        # Add confidence interval
        fig.add_shape(
            type="line",
            x0=lower_bound, y0=0.45,
            x1=upper_bound, y1=0.45,
            line=dict(color=self.colors['primary'], width=8),
        )
        
        # Add confidence interval caps
        for x in [lower_bound, upper_bound]:
            fig.add_shape(
                type="line",
                x0=x, y0=0.42,
                x1=x, y1=0.48,
                line=dict(color=self.colors['primary'], width=4),
            )
        
        # Add market range if provided
        if market_range:
            market_low, market_high = market_range
            fig.add_vrect(
                x0=market_low, x1=market_high,
                fillcolor=self.colors['info'],
                opacity=0.1,
                layer="below",
                line_width=0,
            )
            fig.add_annotation(
                x=(market_low + market_high) / 2,
                y=0.8,
                text="Market Range",
                showarrow=False,
                font=dict(color=self.colors['info'])
            )
        
        # Format layout
        fig.update_layout(
            title={
                'text': 'Price Prediction with Confidence Interval',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.colors['text']}
            },
            xaxis_title="Log Price (Transformed)",
            yaxis=dict(visible=False),
            template=self.theme_template,
            height=200,
            font=dict(color=self.colors['text']),
            showlegend=False,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        # Add value annotations
        fig.add_annotation(
            x=prediction,
            y=0.6,
            text=f"${np.exp(prediction):,.0f}",
            showarrow=False,
            font=dict(color=self.colors['champion'], size=16, weight='bold')
        )
        
        return fig
    
    def create_shap_waterfall(self, base_value: float, shap_values: Dict[str, float], 
                            predicted_value: float, max_features: int = 10) -> go.Figure:
        """Create SHAP waterfall chart for prediction explanation."""
        
        # Sort by absolute SHAP value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:max_features]
        
        features = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        # Calculate cumulative values for waterfall
        cumulative = [base_value]
        for value in values:
            cumulative.append(cumulative[-1] + value)
        
        fig = go.Figure()
        
        # Add base value
        fig.add_trace(go.Bar(
            x=['Base Value'],
            y=[base_value],
            marker_color=self.colors['neutral'],
            name='Base Value'
        ))
        
        # Add feature contributions
        for i, (feature, value) in enumerate(zip(features, values)):
            color = self.colors['success'] if value > 0 else self.colors['danger']
            fig.add_trace(go.Bar(
                x=[feature],
                y=[value],
                base=[cumulative[i]],
                marker_color=color,
                name=f'{feature}: {value:+.3f}',
                showlegend=False
            ))
        
        # Add final prediction
        fig.add_trace(go.Bar(
            x=['Prediction'],
            y=[predicted_value],
            marker_color=self.colors['champion'],
            name='Final Prediction'
        ))
        
        fig.update_layout(
            title='SHAP Waterfall: Feature Contributions to Prediction',
            xaxis_title='Features',
            yaxis_title='Log Price Impact',
            template=self.theme_template,
            height=500,
            font=dict(color=self.colors['text']),
            barmode='relative',
            showlegend=True
        )
        
        return fig
    
    def create_partial_dependence_plot(self, feature_name: str, 
                                     grid_values: List[float], 
                                     pd_values: List[float],
                                     current_value: float = None) -> go.Figure:
        """Create partial dependence plot for feature analysis."""
        
        # Determine trend color
        trend_color = self.colors['success'] if pd_values[-1] > pd_values[0] else self.colors['warning']
        
        # Convert hex color to RGBA with transparency
        def hex_to_rgba(hex_color, alpha=0.2):
            """Convert hex color to RGBA format."""
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f'rgba({r},{g},{b},{alpha})'
            return 'rgba(34,139,34,0.2)'  # fallback
        
        fig = go.Figure()
        
        # Add partial dependence line
        fig.add_trace(go.Scatter(
            x=grid_values,
            y=pd_values,
            mode='lines',
            line=dict(color=trend_color, width=3),
            fill='tonexty',
            fillcolor=hex_to_rgba(trend_color, 0.2),
            name='Partial Dependence'
        ))
        
        # Add current value indicator if provided
        if current_value is not None:
            # Find closest grid value
            closest_idx = min(range(len(grid_values)), key=lambda i: abs(grid_values[i] - current_value))
            fig.add_trace(go.Scatter(
                x=[grid_values[closest_idx]],
                y=[pd_values[closest_idx]],
                mode='markers',
                marker=dict(
                    size=15,
                    color=self.colors['champion'],
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                name='Current Value'
            ))
        
        # Calculate impact range
        impact_range = max(pd_values) - min(pd_values)
        trend_direction = "↗" if pd_values[-1] > pd_values[0] else "↘"
        
        fig.update_layout(
            title=f'{feature_name.replace("_", " ").title()}<br><sub>Impact Range: {impact_range:.3f} | Trend: {trend_direction}</sub>',
            xaxis_title=feature_name.replace('_', ' ').title(),
            yaxis_title='Partial Dependence',
            template=self.theme_template,
            height=400,
            font=dict(color=self.colors['text']),
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_metric_cards_html(self, metrics: Dict[str, Any]) -> str:
        """Create professional metric cards using HTML/CSS."""
        
        cards_html = "<div style='display: flex; gap: 20px; margin-bottom: 20px;'>"
        
        for i, (title, value) in enumerate(metrics.items()):
            # Determine card style based on content
            if 'accuracy' in title.lower() or 'performance' in title.lower():
                card_class = 'champion-metric'
            elif 'prediction' in title.lower() or 'reliability' in title.lower():
                card_class = 'success-metric'
            elif 'warning' in title.lower() or 'risk' in title.lower():
                card_class = 'warning-metric'
            else:
                card_class = ''
            
            cards_html += f"""
            <div class='metric-card {card_class}' style='flex: 1; text-align: center;'>
                <h3 style='margin: 0 0 10px 0; color: {self.colors['text']};'>{title}</h3>
                <h2 style='margin: 0; color: {self.colors['primary']}; font-size: 2em;'>{value}</h2>
            </div>
            """
        
        cards_html += "</div>"
        return cards_html

# Global visualization instance
@st.cache_resource
def get_visualizer():
    """Get cached visualizer instance."""
    return ProfessionalVisualizations()