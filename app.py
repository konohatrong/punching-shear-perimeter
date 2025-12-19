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
    
    # 1. Unit System Selection
    unit_sys = st.radio("Unit System", ["Imperial (in, psi)", "Metric (mm, MPa)"])
    
    if "Imperial" in unit_sys:
        u_len = "in"
        u_area = "in²"
        u_inertia = "in⁴"
        def_Cx, def_Cy, def_d = 12.0, 20.0, 5.62
        def_so, def_s = 2.25, 2.75
        def_trial = 25.3
    else:
        u_len = "mm"
        u_area = "mm²"
        u_inertia = "mm⁴"
        def_Cx, def_Cy, def_d = 300.0, 500.0, 140.0
        def_so, def_s = 57.0, 70.0
        def_trial = 640.0

    st.markdown("---")
    st.header("1. Column Data")
    col_type = st.selectbox("Column Type", 
                            ["Interior", "Edge (Left Free)", "Corner (Top-Left Free)"])
    
    c1, c2 = st.columns(2)
    Cx = c1.number_input(f"Width Cx ({u_len})", value=def_Cx, step=1.0)
    Cy = c2.number_input(f"Depth Cy ({u_len})", value=def_Cy, step=1.0)
    d = st.number_input(f"Effective Depth d ({u_len})", value=def_d, step=0.1)
    
    st.markdown("---")
    st.header("2. Critical Section")
    calc_mode = st.radio("Calculation Mode:", 
                         ["Manual Distance (Trial)", "From Stud Layout"])
    
    dist_val = 0.0
    if calc_mode == "Manual Distance (Trial)":
        # Default Trial Guess
        def_val_dist = def_trial if col_type == "Interior" else d/2
        dist_val = st.number_input(f"Distance from Face ({u_len})", value=def_val_dist)
        st.caption(f"Note: d/2 = {d/2:.2f} {u_len}")
    else:
        st.subheader("Stud Layout Details")
        so = st.number_input(f"s0 (First spacing) ({u_len})", value=def_so)
        s = st.number_input(f"s (Typ. spacing) ({u_len})", value=def_s)
        n = st.number_input("No. of lines", value=9, min_value=2, step=1)
        
        # Distance = s0 + (n-1)s + d/2
        dist_val = so + (n-1)*s + (d/2)
        st.success(f"Calc. Distance = {dist_val:.2f} {u_len}")

# --- Main Calculation ---
points = generate_critical_section(Cx, Cy, dist_val, col_type)
res = calculate_section_properties(points, d)

if res:
    # --- Results Display ---
    st.subheader(f"Analysis Results: {col_type} Column")
    st.info(f"Critical Section at distance **{dist_val:.2f} {u_len}** from column face.")
    
    # Metrics Row 1
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Perimeter (bo)", f"{res['bo']:.2f} {u_len}")
    m2.metric(f"Area (Ac)", f"{res['Ac']:.2f} {u_area}")
    m3.metric(f"Centroid (x, y)", f"({res['Centroid'][0]:.2f}, {res['Centroid'][1]:.2f}) {u_len}", 
              help="Relative to Column Center (0,0)")

    st.markdown("---")
    
    # J Values
    st.subheader("Moment of Inertia (J)")
    
    # Logic to highlight Principal Moments if Jxy is significant (Corner Case)
    is_unsymmetric = abs(res['Jxy']) > 1.0 # Tolerance
    
    c1, c2 = st.columns(2)
    
    if is_unsymmetric:
        st.warning("⚠️ Unsymmetric Section (Corner): Principal Moments recommended.")
        c1.metric("J_major (Principal)", f"{res['J_major']:,.2f} {u_inertia}")
        c2.metric("J_minor (Principal)", f"{res['J_minor']:,.2f} {u_inertia}")
        st.caption(f"Principal Angle (θ) = {res['theta_deg']:.2f}°")
        
        with st.expander("Show Orthogonal J (x,y)"):
            st.write(f"Jcx: {res['Jcx']:,.2f} {u_inertia}")
            st.write(f"Jcy: {res['Jcy']:,.2f} {u_inertia}")
            st.write(f"Jxy: {res['Jxy']:,.2f} {u_inertia}")
    else:
        st.success("✅ Symmetric Section: Orthogonal axes are Principal axes.")
        c1.metric("Jcx (Major Axis)", f"{res['Jcx']:,.2f} {u_inertia}")
        c2.metric("Jcy (Minor Axis)", f"{res['Jcy']:,.2f} {u_inertia}")

    # --- Interactive Plot (Plotly) ---
    st.markdown("---")
    st.subheader("Section Diagram (Interactive)")
    st.caption("🔍 Use the toolbar in the top right of the chart to **Zoom**, **Pan**, or **Reset View**.")

    # 1. Prepare Data for Plotly
    # Polygon Points
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    
    # Close loop for filling
    if col_type == "Interior":
        px.append(points[0][0])
        py.append(points[0][1])
        fill_opt = 'toself'
    else:
        fill_opt = 'none' # Open shapes usually don't fill nicely unless closed manually

    # Column Box
    cx_box_x = [-Cx/2, Cx/2, Cx/2, -Cx/2, -Cx/2]
    cx_box_y = [Cy/2, Cy/2, -Cy/2, -Cy/2, Cy/2]

    # 2. Create Figure
    fig = go.Figure()

    # Trace: Column
    fig.add_trace(go.Scatter(
        x=cx_box_x, y=cx_box_y,
        fill="toself",
        fillcolor="rgba(128, 128, 128, 0.5)", # Gray transparency
        line=dict(color="gray", width=2),
        name="Column"
    ))

    # Trace: Critical Section
    fig.add_trace(go.Scatter(
        x=px, y=py,
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=6),
        name="Critical Section"
    ))

    # Trace: Centroid
    fig.add_trace(go.Scatter(
        x=[res['Centroid'][0]], y=[res['Centroid'][1]],
        mode='markers',
        marker=dict(color='red', size=12, symbol='cross'),
        name="Centroid"
    ))
    
    # Trace: Column Center
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name="Col Center"
    ))

    # 3. Layout Settings (On-Scale & Interaction)
    fig.update_layout(
        title=f"Critical Section Geometry ({col_type})",
        xaxis_title=f"Distance X ({u_len})",
        yaxis_title=f"Distance Y ({u_len})",
        showlegend=True,
        width=700,
        height=700,
        # Ensure aspect ratio is 1:1 (On-scale)
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Detailed Data Table ---
    with st.expander("See Calculation Details (Segments)"):
        df_seg = pd.DataFrame(res['segments'])
        st.dataframe(df_seg.style.format("{:.2f}"))
