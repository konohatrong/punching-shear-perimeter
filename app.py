import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 0. SESSION STATE & CALLBACKS (For Validation Buttons)
# ==========================================
def init_session_state():
    # Default values (Imperial D.1)
    defaults = {
        "unit_sys": "Imperial (in, psi)",
        "col_type": "Interior",
        "Cx": 12.0, "Cy": 20.0, "d": 5.62,
        "calc_mode": "Manual Distance (Trial)",
        "dist_manual": 25.3,
        "so": 2.25, "s": 2.75, "n": 9
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def load_example(ex_id):
    if ex_id == "D1":
        st.session_state.unit_sys = "Imperial (in, psi)"
        st.session_state.col_type = "Interior"
        st.session_state.Cx = 12.0
        st.session_state.Cy = 20.0
        st.session_state.d = 5.62
        st.session_state.calc_mode = "Manual Distance (Trial)"
        st.session_state.dist_manual = 25.3 # Trial guess from book
        
    elif ex_id == "D2":
        st.session_state.unit_sys = "Imperial (in, psi)"
        st.session_state.col_type = "Edge (Left Free)"
        st.session_state.Cx = 18.0
        st.session_state.Cy = 18.0
        st.session_state.d = 5.62
        st.session_state.calc_mode = "From Stud Layout"
        st.session_state.so = 2.25
        st.session_state.s = 2.75
        st.session_state.n = 9
        
    elif ex_id == "D3":
        st.session_state.unit_sys = "Imperial (in, psi)"
        st.session_state.col_type = "Corner (Top-Left Free)"
        st.session_state.Cx = 20.0
        st.session_state.Cy = 20.0
        st.session_state.d = 5.62
        st.session_state.calc_mode = "From Stud Layout"
        st.session_state.so = 2.25
        st.session_state.s = 2.5  # Ex D.3 uses s=2.5
        st.session_state.n = 7    # Ex D.3 uses 7 lines

# Initialize Session State
init_session_state()

# ==========================================
# 1. CORE CALCULATION ENGINE
# ==========================================
def calculate_section_properties(points, d):
    # 1.1 Calculate Perimeter and Centroid
    total_length = 0
    sum_mx = 0 
    sum_my = 0 
    
    segments = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        l = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if l == 0: continue
            
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        
        total_length += l
        sum_mx += l * ym
        sum_my += l * xm
        segments.append({'p1': (x1,y1), 'p2': (x2,y2), 'l': l, 'xm': xm, 'ym': ym})
        
    if total_length == 0: return None
    
    x_bar = sum_my / total_length
    y_bar = sum_mx / total_length
    
    # 1.2 Calculate Inertia & Extreme Fibers
    Jcx_c, Jcy_c, Jxy_c = 0, 0, 0
    
    # Extreme fibers (Relative to Centroid)
    # c_x_pos (Right), c_x_neg (Left), c_y_pos (Top/Bot depending on sign)
    # We collect all points shifted to find max bounds
    x_shifted_all = []
    y_shifted_all = []

    for seg in segments:
        # Shift coordinates
        x1_p = seg['p1'][0] - x_bar
        y1_p = seg['p1'][1] - y_bar
        x2_p = seg['p2'][0] - x_bar
        y2_p = seg['p2'][1] - y_bar
        l = seg['l']
        
        x_shifted_all.extend([x1_p, x2_p])
        y_shifted_all.extend([y1_p, y2_p])
        
        # Eq B.8, B.9, B.11
        term_y = (y1_p**2 + y1_p*y2_p + y2_p**2)
        Jcx_c += d * (l/3) * term_y
        
        term_x = (x1_p**2 + x1_p*x2_p + x2_p**2)
        Jcy_c += d * (l/3) * term_x
        
        term_xy = (2*x1_p*y1_p + x1_p*y2_p + x2_p*y1_p + 2*x2_p*y2_p)
        Jxy_c += d * (l/6) * term_xy

    # Extreme Fiber Distances (c)
    c_x_max = max(x_shifted_all) # Rightmost fiber
    c_x_min = min(x_shifted_all) # Leftmost fiber
    c_y_max = max(y_shifted_all) # Topmost fiber
    c_y_min = min(y_shifted_all) # Bottommost fiber

    # 1.3 Principal Moments
    avg_J = (Jcx_c + Jcy_c) / 2
    diff_J = (Jcx_c - Jcy_c) / 2
    radius = math.sqrt(diff_J**2 + Jxy_c**2)
    
    J_max = avg_J + radius
    J_min = avg_J - radius
    
    if abs(Jcx_c - Jcy_c) < 1e-6:
        theta_rad = 0 if abs(Jxy_c) < 1e-6 else math.pi/4
    else:
        theta_rad = 0.5 * math.atan2(-2*Jxy_c, (Jcx_c - Jcy_c))

    return {
        "bo": total_length,
        "Ac": total_length * d,
        "Centroid": (x_bar, y_bar),
        "Jcx": Jcx_c, "Jcy": Jcy_c, "Jxy": Jxy_c,
        "J_major": J_max, "J_minor": J_min, "theta_deg": math.degrees(theta_rad),
        "segments": segments,
        "extreme": {"cx_pos": c_x_max, "cx_neg": c_x_min, "cy_pos": c_y_max, "cy_neg": c_y_min},
        "points_shifted": list(zip(x_shifted_all, y_shifted_all)) # For checking
    }

# ==========================================
# 2. SHAPE GENERATION LOGIC
# ==========================================
def generate_critical_section(Cx, Cy, dist, col_type):
    hx = Cx / 2
    hy = Cy / 2
    X_far = hx + dist
    Y_far = hy + dist
    points = []
    
    if col_type == "Interior":
        points = [
            (-hx, Y_far), (hx, Y_far), (X_far, hy), (X_far, -hy),
            (hx, -Y_far), (-hx, -Y_far), (-X_far, -hy), (-X_far, hy), (-hx, Y_far)
        ]
    elif col_type == "Edge (Left Free)":
        points = [
            (-hx, Y_far), (hx, Y_far), (X_far, hy), (X_far, -hy),
            (hx, -Y_far), (-hx, -Y_far)
        ]
    elif col_type == "Corner (Top-Left Free)":
        points = [(X_far, hy), (X_far, -hy), (hx, -Y_far), (-hx, -Y_far)]
        
    return points

# ==========================================
# 3. UI SETUP
# ==========================================
st.set_page_config(page_title="ACI 421 Punching Shear", page_icon="🏗️", layout="wide")

st.title("🏗️ ACI 421.1R-20 Punching Shear Calculator")
st.markdown("Analysis tool for **Section Properties** ($J_c, A_c, b_o$) with **ACI 421 Validation Examples**.")

# --- Sidebar ---
with st.sidebar:
    st.header("📚 Validation Examples")
    c1, c2, c3 = st.columns(3)
    if c1.button("Load D.1"): load_example("D1")
    if c2.button("Load D.2"): load_example("D2")
    if c3.button("Load D.3"): load_example("D3")
    
    st.markdown("---")
    st.header("⚙️ Inputs")
    
    # Bind widgets to session state
    unit_sys = st.radio("Unit System", ["Imperial (in, psi)", "Metric (mm, MPa)"], key="unit_sys")
    
    if "Imperial" in unit_sys:
        u_len, u_area, u_inertia = "in", "in²", "in⁴"
    else:
        u_len, u_area, u_inertia = "mm", "mm²", "mm⁴"

    col_type = st.selectbox("Column Type", ["Interior", "Edge (Left Free)", "Corner (Top-Left Free)"], key="col_type")
    
    c_col1, c_col2 = st.columns(2)
    Cx = c_col1.number_input(f"Cx ({u_len})", key="Cx", step=1.0)
    Cy = c_col2.number_input(f"Cy ({u_len})", key="Cy", step=1.0)
    d = st.number_input(f"Effective Depth d ({u_len})", key="d", step=0.1)
    
    st.markdown("---")
    calc_mode = st.radio("Mode:", ["Manual Distance (Trial)", "From Stud Layout"], key="calc_mode")
    
    dist_val = 0.0
    if calc_mode == "Manual Distance (Trial)":
        dist_val = st.number_input(f"Distance ({u_len})", key="dist_manual")
    else:
        st.caption("Stud Layout Details")
        so = st.number_input(f"s0 ({u_len})", key="so")
        s = st.number_input(f"s ({u_len})", key="s")
        n = st.number_input("No. of lines", key="n", min_value=2)
        dist_val = so + (n-1)*s + (d/2)
        st.success(f"Dist = {dist_val:.2f} {u_len}")

# --- Main Calculation ---
points = generate_critical_section(Cx, Cy, dist_val, col_type)
res = calculate_section_properties(points, d)

if res:
    # 1. Summary Metrics
    st.subheader(f"Analysis Summary: {col_type}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Perimeter (bo)", f"{res['bo']:.2f} {u_len}")
    m2.metric("Area (Ac)", f"{res['Ac']:.2f} {u_area}")
    m3.metric("Centroid (x, y)", f"({res['Centroid'][0]:.2f}, {res['Centroid'][1]:.2f}) {u_len}")
    
    # 2. J Values
    is_unsymmetric = abs(res['Jxy']) > 1.0
    col1, col2 = st.columns(2)
    if is_unsymmetric:
        st.warning("⚠️ Unsymmetric (Corner): Using Principal Moments recommended.")
        col1.metric("J_major (Principal)", f"{res['J_major']:,.2f} {u_inertia}")
        col2.metric("J_minor (Principal)", f"{res['J_minor']:,.2f} {u_inertia}")
    else:
        st.success("✅ Symmetric: Orthogonal axes are Principal axes.")
        col1.metric("Jcx (Major Axis)", f"{res['Jcx']:,.2f} {u_inertia}")
        col2.metric("Jcy (Minor Axis)", f"{res['Jcy']:,.2f} {u_inertia}")

    # 3. Graphic (Plotly)
    st.markdown("---")
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    if col_type == "Interior":
        px.append(points[0][0])
        py.append(points[0][1])

    fig = go.Figure()
    # Column
    fig.add_trace(go.Scatter(
        x=[-Cx/2, Cx/2, Cx/2, -Cx/2, -Cx/2], y=[Cy/2, Cy/2, -Cy/2, -Cy/2, Cy/2],
        fill="toself", fillcolor="rgba(128,128,128,0.3)", line=dict(color="gray"), name="Column"
    ))
    # Section
    fig.add_trace(go.Scatter(
        x=px, y=py, mode='lines+markers', line=dict(color='blue', width=3), name="Critical Section"
    ))
    # Centroid
    fig.add_trace(go.Scatter(
        x=[res['Centroid'][0]], y=[res['Centroid'][1]], mode='markers',
        marker=dict(color='red', size=10, symbol='cross'), name="Centroid"
    ))
    fig.update_layout(
        title="Critical Section Geometry", 
        width=700, height=600, yaxis=dict(scaleanchor="x", scaleratio=1), hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Detailed Report (Tahoma Font)
    st.markdown("---")
    st.subheader("📝 Detailed Analysis Report")
    
    # HTML String Builder for Classic Report
    html_report = f"""
    <div style="font-family: Tahoma, sans-serif; font-size: 14px; line-height: 1.6; color: #333; background-color: #f9f9f9; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
        <h3 style="color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px;">1. Critical Section Geometry</h3>
        <p><strong>Column Dimensions:</strong> {Cx} x {Cy} {u_len} <br>
        <strong>Effective Depth (d):</strong> {d} {u_len} <br>
        <strong>Distance to Critical Section:</strong> {dist_val:.2f} {u_len}</p>
        
        <h4 style="color: #444;">Coordinates of Critical Section (Relative to Column Center)</h4>
        <table style="width: 100%; border-collapse: collapse; font-family: Tahoma, sans-serif;">
            <tr style="background-color: #e0e0e0;">
                <th style="border: 1px solid #ccc; padding: 5px; text-align: center;">Point</th>
                <th style="border: 1px solid #ccc; padding: 5px; text-align: center;">X ({u_len})</th>
                <th style="border: 1px solid #ccc; padding: 5px; text-align: center;">Y ({u_len})</th>
            </tr>
    """
    
    # Loop points for table
    for i, pt in enumerate(points):
        html_report += f"""
            <tr>
                <td style="border: 1px solid #ccc; padding: 5px; text-align: center;">{i+1}</td>
                <td style="border: 1px solid #ccc; padding: 5px; text-align: center;">{pt[0]:.2f}</td>
                <td style="border: 1px solid #ccc; padding: 5px; text-align: center;">{pt[1]:.2f}</td>
            </tr>
        """
        
    html_report += f"""
        </table>
        
        <h3 style="color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 20px;">2. Centroid Calculation</h3>
        <p>The centroid of the critical shear perimeter (line) is calculated as:</p>
        <ul>
            <li><strong>Total Perimeter (bo):</strong> {res['bo']:.2f} {u_len}</li>
            <li><strong>Shear Area (Ac):</strong> {res['Ac']:.2f} {u_area}</li>
            <li><strong>Centroid X (x̄):</strong> {res['Centroid'][0]:.2f} {u_len}</li>
            <li><strong>Centroid Y (ȳ):</strong> {res['Centroid'][1]:.2f} {u_len}</li>
        </ul>
        
        <h3 style="color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 20px;">3. Extreme Fiber Distances (c)</h3>
        <p>Distance from Centroid to extreme points of the critical section (used for stress analysis Mc/J):</p>
        <ul>
            <li><strong>c<sub>x, max</sub> (Right):</strong> {res['extreme']['cx_pos']:.2f} {u_len}</li>
            <li><strong>c<sub>x, min</sub> (Left):</strong> {res['extreme']['cx_neg']:.2f} {u_len}</li>
            <li><strong>c<sub>y, max</sub> (Top):</strong> {res['extreme']['cy_pos']:.2f} {u_len}</li>
            <li><strong>c<sub>y, min</sub> (Bottom):</strong> {res['extreme']['cy_neg']:.2f} {u_len}</li>
        </ul>
        
        <h3 style="color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 20px;">4. Moment of Inertia (J) Analysis</h3>
        <p>Calculated using the summation of segments method (ACI 421.1R-20 Appendix B):</p>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-family: Tahoma, sans-serif;">
            <tr style="background-color: #e0e0e0;">
                <th style="border: 1px solid #ccc; padding: 8px;">Property</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Value ({u_inertia})</th>
                <th style="border: 1px solid #ccc; padding: 8px;">Description</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ccc; padding: 8px;"><strong>J<sub>cx</sub></strong></td>
                <td style="border: 1px solid #ccc; padding: 8px; text-align: right;">{res['Jcx']:,.2f}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">Moment of Inertia about X-axis</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ccc; padding: 8px;"><strong>J<sub>cy</sub></strong></td>
                <td style="border: 1px solid #ccc; padding: 8px; text-align: right;">{res['Jcy']:,.2f}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">Moment of Inertia about Y-axis</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ccc; padding: 8px;"><strong>J<sub>xy</sub></strong></td>
                <td style="border: 1px solid #ccc; padding: 8px; text-align: right;">{res['Jxy']:,.2f}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">Product of Inertia</td>
            </tr>
    """
    
    if is_unsymmetric:
        html_report += f"""
            <tr style="background-color: #fff3cd;">
                <td style="border: 1px solid #ccc; padding: 8px;"><strong>J<sub>major</sub></strong></td>
                <td style="border: 1px solid #ccc; padding: 8px; text-align: right; font-weight: bold;">{res['J_major']:,.2f}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">Principal Moment (Max)</td>
            </tr>
            <tr style="background-color: #fff3cd;">
                <td style="border: 1px solid #ccc; padding: 8px;"><strong>J<sub>minor</sub></strong></td>
                <td style="border: 1px solid #ccc; padding: 8px; text-align: right; font-weight: bold;">{res['J_minor']:,.2f}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">Principal Moment (Min)</td>
            </tr>
        """
    
    html_report += """
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 15px;">* Calculated according to ACI 421.1R-20 Appendix B equations B.8, B.9, B.11</p>
    </div>
    """
    
    st.markdown(html_report, unsafe_allow_html=True)
