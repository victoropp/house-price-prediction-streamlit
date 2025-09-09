"""
Neighborhood Mapping Utilities
Converts encoded neighborhood values to user-friendly names and descriptions
"""

# Neighborhood mapping from encoded values to readable names
NEIGHBORHOOD_MAPPING = {
    117723.62: {"name": "IDOTRR", "tier": "Budget", "description": "Iowa DOT & Rail Road - Budget-friendly area"},
    131342.00: {"name": "OldTown", "tier": "Budget", "description": "Old Town - Historic affordable area"},
    135865.79: {"name": "BrkSide", "tier": "Budget", "description": "Brookside - Established budget community"},
    137170.34: {"name": "Edwards", "tier": "Budget", "description": "Edwards - Affordable residential area"},
    138265.74: {"name": "BrDale", "tier": "Budget", "description": "Briardale - Budget family neighborhood"},
    141406.01: {"name": "MeadowV", "tier": "Mid-Tier", "description": "Meadow Village - Mid-range community"},
    144307.14: {"name": "Sawyer", "tier": "Mid-Tier", "description": "Sawyer - Established mid-tier area"},
    146591.48: {"name": "NAmes", "tier": "Mid-Tier", "description": "North Ames - Popular mid-range area"},
    153182.48: {"name": "SWISU", "tier": "Mid-Tier", "description": "South & West of ISU - Student/professional area"},
    164165.04: {"name": "Mitchel", "tier": "Mid-Tier", "description": "Mitchell - Well-established mid-tier"},
    164748.27: {"name": "NPkVill", "tier": "Mid-Tier", "description": "Northpark Villa - Mid-range development"},
    175746.75: {"name": "Blueste", "tier": "Mid-Tier", "description": "Bluestem - Newer mid-tier community"},
    186582.19: {"name": "Blmngtn", "tier": "Mid-Tier", "description": "Bloomington Heights - Mid-range suburban"},
    186766.73: {"name": "SawyerW", "tier": "Mid-Tier", "description": "Sawyer West - West side mid-tier"},
    186994.37: {"name": "NWAmes", "tier": "Mid-Tier", "description": "Northwest Ames - Established mid-tier"},
    188054.13: {"name": "Gilbert", "tier": "Mid-Tier", "description": "Gilbert - Growing mid-tier community"},
    196200.39: {"name": "CollgCr", "tier": "Mid-Tier", "description": "College Creek - Popular family area"},
    197950.75: {"name": "Veenker", "tier": "Mid-Tier", "description": "Veenker - Golf course community"},
    202267.22: {"name": "ClearCr", "tier": "Premium", "description": "Clear Creek - Premium residential"},
    209292.57: {"name": "Crawfor", "tier": "Premium", "description": "Crawford - Historic premium area"},
    220691.40: {"name": "Somerst", "tier": "Premium", "description": "Somerset - Upscale development"},
    222300.84: {"name": "Timber", "tier": "Premium", "description": "Timberland - Premium wooded area"},
    262044.05: {"name": "StoneBr", "tier": "Premium", "description": "Stone Brook - Luxury community"},
    295211.80: {"name": "NridgHt", "tier": "Premium", "description": "Northridge Heights - Premium hilltop"},
    307844.65: {"name": "NoRidge", "tier": "Premium", "description": "Northridge - Most exclusive area"}
}

def get_neighborhood_info(encoded_value, tolerance=1.0):
    """
    Convert encoded neighborhood value to user-friendly information
    
    Args:
        encoded_value: The numerical encoded neighborhood value
        tolerance: Tolerance for matching (default 1.0)
        
    Returns:
        dict: Neighborhood information with name, tier, and description
    """
    # Find closest match within tolerance
    for mapped_value, info in NEIGHBORHOOD_MAPPING.items():
        if abs(encoded_value - mapped_value) <= tolerance:
            return {
                'name': info['name'],
                'tier': info['tier'], 
                'description': info['description'],
                'avg_price': f"${mapped_value:,.0f}",
                'encoded_value': encoded_value
            }
    
    # Fallback if no exact match found
    tier = "Budget" if encoded_value < 140000 else "Mid-Tier" if encoded_value < 200000 else "Premium"
    return {
        'name': f"Area_{int(encoded_value/1000)}K",
        'tier': tier,
        'description': f"{tier} neighborhood (avg price: ${encoded_value:,.0f})",
        'avg_price': f"${encoded_value:,.0f}",
        'encoded_value': encoded_value
    }

def get_neighborhood_options_for_ui():
    """
    Get neighborhood options formatted for Streamlit UI
    
    Returns:
        dict: Options formatted as {display_name: encoded_value}
    """
    options = {}
    for encoded_value, info in NEIGHBORHOOD_MAPPING.items():
        display_name = f"{info['name']} - {info['tier']} (${encoded_value:,.0f})"
        options[display_name] = encoded_value
    
    return options

def format_neighborhood_for_prediction_display(encoded_value):
    """
    Format neighborhood for prediction result display
    
    Args:
        encoded_value: The encoded neighborhood value
        
    Returns:
        str: Formatted string for display
    """
    info = get_neighborhood_info(encoded_value)
    return f"📍 {info['name']} ({info['tier']} - {info['avg_price']} avg)"

def get_neighborhood_impact_explanation(encoded_value, impact_score):
    """
    Generate user-friendly explanation of neighborhood impact
    
    Args:
        encoded_value: The encoded neighborhood value
        impact_score: The SHAP impact score
        
    Returns:
        str: User-friendly explanation
    """
    info = get_neighborhood_info(encoded_value)
    impact_pct = abs(impact_score) * 100
    
    if impact_score > 0:
        direction = "increases"
        icon = "📈"
    else:
        direction = "decreases" 
        icon = "📉"
        
    return f"{icon} **{info['name']} Neighborhood** ({info['tier']}): {direction} price by {impact_pct:.1f}% - {info['description']}"