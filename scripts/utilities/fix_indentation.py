#!/usr/bin/env python3
"""
Fix indentation for the try block in show_model_analytics function
"""

def fix_indentation():
    file_path = r"C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\streamlit_app.py"
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Find the try block start in show_model_analytics
    try_line_idx = None
    except_line_idx = None
    
    for i, line in enumerate(lines):
        if "try:" in line and i > 850:  # Around show_model_analytics function
            try_line_idx = i
            print(f"Found try at line {i+1}")
        if "except ZeroDivisionError as e:" in line and try_line_idx is not None:
            except_line_idx = i
            print(f"Found except at line {i+1}")
            break
    
    if try_line_idx is None or except_line_idx is None:
        print("Could not find try/except block")
        return
    
    # Add 4 spaces indentation to all lines between try and except
    fixed_lines = []
    for i, line in enumerate(lines):
        if try_line_idx < i < except_line_idx:
            # Only add indentation if line is not already properly indented
            if line.strip() and not line.startswith('        '):  # Not already indented 8 spaces
                if line.startswith('    '):  # Currently 4 spaces, make it 8
                    fixed_lines.append('    ' + line)
                else:
                    fixed_lines.append('        ' + line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Fixed indentation")

if __name__ == "__main__":
    fix_indentation()