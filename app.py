import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. CORE CALCULATION ENGINE (Appendix B)
# ==========================================
def calculate_section_properties(points, d):
    """
    Calculates geometric properties (Area, J, Centroid) of a general punching shear 
    critical section defined by polygon segments (ACI 421.1R-20 Appendix B).
    """
    # 1.1 Calculate Perimeter and Centroid (Line Properties)
    total_length = 0
    sum_mx = 0 # Moment about X-axis (integral y dl)
    sum_my = 0 # Moment about Y-axis (integral x dl)
    
    segments = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        l = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if l == 0: continue
            
        # Segment Midpoint
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        
        total_length += l
        sum_mx += l * ym
        sum_my += l * xm
        segments.append({'p1': (x1,y1), 'p2': (x2,y2), 'l': l, 'xm': xm, 'ym': ym})
        
    if total_length == 0: return None
    
    # Centroid of Critical Section (relative to Column Center)
    x_bar = sum_my / total_length
    y_bar = sum_mx / total_length
    
    # 1.2 Calculate Inertia about Centroidal Axes (Shifted Axes)
    # Ref: ACI 421.1R-20 Eq B.8, B.9, B.11
    Jcx_c = 0
    Jcy_c = 0
    Jxy_c = 0
    
    for seg in segments:
        # Shift coordinates to Critical Section Centroid
        x1_p = seg['p1'][0] - x_bar
        y1_p = seg['p1'][1] - y_bar
        x2_p = seg['p2'][0] - x_bar
        y2_p = seg['p2'][1] - y_bar
        l = seg['l']
        
        # Eq B.8: Jcx (About X-axis)
        term_y = (y1_p**2 + y1_p*y2_p + y2_p**2)
        Jcx_c += d * (l/3) * term_y
        
        # Eq B.9: Jcy (About Y-axis)
        term_x = (x1_p**2 + x1_p*x2_p + x2_p**2)
        Jcy_c += d * (l/3) * term_x
        
        # Eq B.11: Jxy (Product of Inertia)
        term_xy = (2*x1_p*y1_p + x1_p*y2_p + x2_p*y1_p + 2*x2_p*y2_p)
        Jxy_c += d * (l/6) * term_xy

    # 1.3 Calculate Principal Moments (for unsymmetric sections)
    avg_J = (Jcx_c + Jcy_c) / 2
    diff_J = (Jcx_c - Jcy_c) / 2
    radius = math.sqrt(diff_J**2 + Jxy_c**2)
    
    J_max = avg_J + radius # Major Axis
    J_min = avg_J - radius # Minor Axis
    
    # Calculate Principal Angle (Theta)
    if abs(Jcx_c - Jcy_c) < 1e-6:
        theta_rad = 0 if abs(Jxy_c) < 1e-6 else math.pi/4
    else:
        # Eq B.10
        theta_rad = 0.5 * math.atan2(-2*Jxy_c, (Jcx_c - Jcy_c))

    return {
        "bo": total_length,
        "Ac": total_length * d,
        "Centroid": (x_bar, y_bar),
        "Jcx": Jcx_c,
        "Jcy": Jcy_c,
        "Jxy": Jxy_c,
        "J_major": J_max,
        "J_minor": J_min,
        "theta_deg": math.degrees(theta_rad),
        "segments": segments
    }

# ==========================================
# 2. SHAPE GENERATION LOGIC
# ==========================================
def generate_critical_section(Cx, Cy, dist, col_type):
    """
    Generates polygon points based on column type and distance.
    Logic follows ACI 421 Fig. B (Octagon logic).
    """
    hx = Cx / 2
    hy = Cy / 2
    
    # Outer boundaries relative to center
    X_far = hx + dist
    Y_far = hy + dist
    
    points = []
    
    if col_type == "Interior":
        # Octagon (Closed loop)
        points = [
            (-hx, Y_far), (hx, Y_far),   # Top Edge
            (X_far, hy), (X_far, -hy),   # Right Edge
            (hx, -Y_far), (-hx, -Y_far), # Bottom Edge
            (-X_far, -hy), (-X_far, hy), # Left Edge
            (-hx, Y_far)                 # Close Loop
        ]
        
    elif col_type == "Edge (Left Free)":
        # Open C-Shape (Opening at Left -X)
        points = [
            (-hx, Y_far),    # Start Top-Left (at Free Edge)
            (hx, Y_far),     # Top Inner Corner
            (X_far, hy),     # Top-Right Chamfer
            (X_far, -hy),    # Bot-Right Chamfer
            (hx, -Y_far),    # Bot Inner Corner
            (-hx, -Y_far)    # End Bot-Left (at Free Edge)
        ]
        
    elif col_type == "Corner (Top-Left Free)":
        # Open L-Shape (Opening at Top +Y and Left -X)
        points = [
            (X_far, hy),     # Start Top-Right (at Free Edge)
            (X_far, -hy),    # Right-Bot Chamfer
            (hx, -Y_far),    # Bot Inner Corner
            (-hx, -Y_far)    # End Bot-Left (at Free Edge)
        ]
        
    return points

# ==========================================
# 3. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Punching Shear Analysis", page_icon="🏗️", layout="wide")

st.title("🏗️ ACI 421.1R-20 Punching Shear Calculator")
st.markdown("""
Analysis tool for **Section Properties ($J_c, A_c, b_o$)** of punching shear critical sections.
Supports **Interior, Edge, and Corner** columns using the **General Polygon Method (Appendix B)**.
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1.
