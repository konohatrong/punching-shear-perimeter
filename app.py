import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 0. SESSION STATE & CALLBACKS
# ==========================================
def init_session_state():
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
        st.session_state.dist_manual = 25.3
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
        st.session_state.s = 2.5
        st.session_state.n = 7

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
    
    # 1.2 Calculate Inertia & Extreme Fibers (Shifted)
    Jcx_c, Jcy_c, Jxy_c = 0, 0, 0
    x_shifted_all, y_shifted_all = [], []
    
    # Create detailed segment data for report
    detailed_segments = []

    for idx, seg in enumerate(segments):
        # Shift coordinates relative to Centroid
        xi = seg['p1'][0] - x_bar
        yi = seg['p1'][1] - y_bar
        xj = seg['p2'][0] - x_bar
        yj = seg['p2'][1] - y_bar
        l = seg['l']
        
        x_shifted_all.extend([xi, xj])
        y_shifted_all.extend([yi, yj])
        
        # Calculation Terms (ACI 421 App B)
        # Jcx term: (yi^2 + yi*yj + yj^2)
        term_y_val = (yi**2 + yi*yj + yj**2)
        jcx_part = d * (l/3) * term_y_val
        
        # Jcy term: (xi^2 + xi*xj + xj^2)
        term_x_val = (xi**2 + xi*xj + xj**2)
        jcy_part = d * (l/3) * term_x_val
        
        # Jxy term
        term_xy_val = (2*xi*yi + xi*yj + xj*yi + 2*xj*yj)
        jxy_part = d * (l/6) * term_xy_val
        
        Jcx_c += jcx_part
        Jcy_c += jcy_part
        Jxy_c += jxy_part
        
        detailed_segments.append({
            "id": idx + 1,
            "l": l,
            "xi": xi, "yi": yi,
            "xj": xj, "yj": yj,
            "jcx": jcx_part,
            "jcy": jcy_part
        })

    # Extreme Fibers
    c_vals = {
        "cx_pos": max(x_shifted_all), "cx_neg": min(x_shifted_all),
        "cy_pos": max(y_shifted_all), "cy_neg": min(y_shifted_all)
    }

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
        "bo": total_length, "Ac": total_length * d,
        "Centroid": (x_bar, y_bar),
        "Jcx": Jcx_c, "Jcy": Jcy_c, "Jxy": Jxy_c,
        "J_major": J_max, "J_minor": J_min, "theta_deg": math.degrees(theta_rad),
        "detailed_segments": detailed_segments,
        "extreme": c_vals
    }

# ==========================================
# 2. SHAPE GENERATION LOGIC
# ==========================================
def generate_critical_section(Cx, Cy, dist, col_type):
    hx = Cx / 2; hy = Cy / 2
    X_far = hx + dist; Y_far = hy + dist
    points = []
    
    if col_type == "Interior":
        points = [(-hx, Y_far), (hx, Y_far), (X_far, hy), (X_far, -hy),
                  (hx, -Y_far), (-hx, -Y_far), (-X_far, -hy), (-X_far, hy), (-hx, Y_far)]
    elif col_type == "Edge (Left Free)":
        points = [(-hx, Y_far), (hx, Y_far), (X_far, hy), (X_far, -hy),
                  (hx, -Y_far), (-hx, -Y_far)]
    elif col_type == "Corner (Top-Left Free)":
        points = [(X_far, hy), (X_far, -hy), (hx, -Y_far), (-hx, -Y_far)]
        
    return points

# ==========================================
# 3. REPORT GENERATOR (Classic Engineering Style)
# ==========================================
def generate_html_report(res, Cx, Cy, d, dist_val, u_len, u_inertia):
    # Prepare rows for detailed table
    rows_html = ""
    for seg in res['detailed_segments']:
        rows_html += f"""
        <tr>
            <td style="text-align: center;">{seg['id']}</td>
            <td style="text-align: right;">{seg['l']:.2f}</td>
            <td style="text-align: right; color: #555;">{seg['xi']:.2f}</td>
            <td style="text-align: right; color: #555;">{seg['yi']:.2f}</td>
            <td style="text-align: right; color: #555;">{seg['xj']:.2f}</td>
            <td style="text-align: right; color: #555;">{seg['yj']:.2f}</td>
            <td style="text-align: right; font-weight: bold; background-color: #f0f8ff;">{seg['jcx']:,.0f}</td>
            <td style="text-align: right; font-weight: bold; background-color: #fff0f0;">{seg['jcy']:,.0f}</td>
        </tr>
        """

    html = f"""
    <div style="font-family: Tahoma, sans-serif; font-size: 14px; color: #000; padding: 20px; background-color: #fff; border: 1px solid #ccc;">
        
        <div style="border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #000;">Punching Shear Analysis Report</h2>
            <p style="margin: 5px 0 0 0; color: #555; font-size: 12px;">Reference Standard: ACI 421.1R-20 Appendix B</p>
        </div>

        <h4 style="border-bottom: 1px solid #999; padding-bottom: 5px; margin-top: 20px;">1. Geometric Parameters</h4>
        <table style="width: 100%; font-size: 14px;">
            <tr>
                <td style="width: 30%;"><strong>Column Dimensions:</strong></td>
                <td>C<sub>x</sub> = {Cx} {u_len}, C<sub>y</sub> = {Cy} {u_len}</td>
            </tr>
            <tr>
                <td><strong>Effective Depth:</strong></td>
                <td>d = {d} {u_len}</td>
            </tr>
            <tr>
                <td><strong>Critical Section Distance:</strong></td>
                <td>dist = {dist_val:.2f} {u_len} (from face)</td>
            </tr>
        </table>

        <h4 style="border-bottom: 1px solid #999; padding-bottom: 5px; margin-top: 25px;">2. Properties of Critical Section</h4>
        <p style="margin-bottom: 10px;">
            <strong>Perimeter (b<sub>o</sub>):</strong> {res['bo']:.2f} {u_len} &nbsp;|&nbsp; 
            <strong>Area (A<sub>c</sub>):</strong> {res['Ac']:.2f} {u_len}²
        </p>
        <p style="margin-bottom: 10px;">
            <strong>Centroid (Relative to Column Center):</strong><br>
            x̄ = {res['Centroid'][0]:.2f} {u_len} <br>
            ȳ = {res['Centroid'][1]:.2f} {u_len}
        </p>
        <div style="background-color: #f9f9f9; padding: 10px; border: 1px dashed #ccc; font-size: 13px;">
            <strong>Extreme Fiber Distances (from Centroid):</strong><br>
            Horizontal: c<sub>x,left</sub> = {res['extreme']['cx_neg']:.2f}, c<sub>x,right</sub> = {res['extreme']['cx_pos']:.2f} {u_len}<br>
            Vertical: &nbsp;&nbsp;&nbsp;&nbsp;c<sub>y,bot</sub> = {res['extreme']['cy_neg']:.2f}, c<sub>y,top</sub> = {res['extreme']['cy_pos']:.2f} {u_len}
        </div>

        <h4 style="border-bottom: 1px solid #999; padding-bottom: 5px; margin-top: 25px;">3. Moment of Inertia Calculation (J<sub>c</sub>)</h4>
        
        <div style="margin-bottom: 15px; font-style: italic; color: #444; font-size: 13px;">
            <strong>Governing Equations (ACI 421.1R-20 Eq. B.8 & B.9):</strong><br>
            Values are calculated using the summation of segments relative to the centroid (x̄, ȳ).<br>
            J<sub>cx</sub> = d × Σ [ (l/3) × (y<sub>i</sub>² + y<sub>i</sub>y<sub>j</sub> + y<sub>j</sub>² ) ] <br>
            J<sub>cy</sub> = d × Σ [ (l/3) × (x<sub>i</sub>² + x<sub>i</sub>x<sub>j</sub> + x<sub>j</sub>² ) ]
        </div>

        <table style="width: 100%; border-collapse: collapse; font-size: 12px; font-family: Tahoma, sans-serif;">
            <thead>
                <tr style="background-color: #333; color: #fff;">
                    <th style="padding: 6px; border: 1px solid #444;">Seg</th>
                    <th style="padding: 6px; border: 1px solid #444;">Length (l)</th>
                    <th style="padding: 6px; border: 1px solid #444;">x<sub>i</sub></th>
                    <th style="padding: 6px; border: 1px solid #444;">y<sub>i</sub></th>
                    <th style="padding: 6px; border: 1px solid #444;">x<sub>j</sub></th>
                    <th style="padding: 6px; border: 1px solid #444;">y<sub>j</sub></th>
                    <th style="padding: 6px; border: 1px solid #444;">J<sub>cx</sub> Contribution</th>
                    <th style="padding: 6px; border: 1px solid #444;">J<sub>cy</sub> Contribution</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
                <tr style="background-color: #eee; font-weight: bold; font-size: 13px;">
                    <td colspan="6" style="text-align: right; padding: 8px; border-top: 2px solid #000;">Total Σ :</td>
                    <td style="text-align: right; padding: 8px; border-top: 2px solid #000; color: #000;">{res['Jcx']:,.0f}</td>
                    <td style="text-align: right; padding: 8px; border-top: 2px solid #000; color: #000;">{res['Jcy']:,.0f}</td>
                </tr>
            </tbody>
        </table>
        <p style="font-size: 11px; color: #666; margin-top: 5px;">* Coordinates (x, y) shown are relative to the section centroid.</p>

        <h4 style="border-bottom: 1px solid #999; padding-bottom: 5px; margin-top: 25px;">4. Final Section Properties</h4>
        <table style="width: 100%; border: 1px solid #ddd; border-collapse: collapse;">
            <tr style="background-color: #f0f8ff;">
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>J<sub>cx</sub> (Major Axis Inertia):</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;">{res['Jcx']:,.2f} {u_inertia}</td>
            </tr>
            <tr style="background-color: #fff0f0;">
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>J<sub>cy</sub> (Minor Axis Inertia):</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;">{res['Jcy']:,.2f} {u_inertia}</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><strong>J<sub>xy</sub> (Product of Inertia):</strong></td>
                <td style="padding: 10px; text-align: right;">{res['Jxy']:,.2f} {u_inertia}</td>
            </tr>
        </table>
    </div>
    """
    return html

# ==========================================
# 4. UI MAIN
# ==========================================
st.set_page_config(page_title="ACI 421 Analysis", page_icon="🏗️", layout="wide")
st.title("🏗️ ACI 421.1R-20 Punching Shear Calculator")

# Sidebar
with st.sidebar:
    st.header("📚 Validation Examples")
    c1, c2, c3 = st.columns(3)
    if c1.button("Load D.1"): load_example("D1")
    if c2.button("Load D.2"): load_example("D2")
    if c3.button("Load D.3"): load_example("D3")
    
    st.markdown("---")
    st.header("⚙️ Inputs")
    unit_sys = st.radio("Unit System", ["Imperial (in, psi)", "Metric (mm, MPa)"], key="unit_sys")
    if "Imperial" in unit_sys: u_len, u_inertia = "in", "in⁴"
    else: u_len, u_inertia = "mm", "mm⁴"

    col_type = st.selectbox("Column Type", ["Interior", "Edge (Left Free)", "Corner (Top-Left Free)"], key="col_type")
    c_col1, c_col2 = st.columns(2)
    Cx = c_col1.number_input(f"Cx ({u_len})", key="Cx", step=1.0)
    Cy = c_col2.number_input(f"Cy ({u_len})", key="Cy", step=1.0)
    d = st.number_input(f"Effective Depth d ({u_len})", key="d", step=0.1)
    
    st.markdown("---")
    calc_mode = st.radio("Mode:", ["Manual Distance (Trial)", "From Stud Layout"], key="calc_mode")
    if calc_mode == "Manual Distance (Trial)":
        dist_val = st.number_input(f"Distance ({u_len})", key="dist_manual")
    else:
        st.caption("Stud Layout Details")
        so = st.number_input(f"s0 ({u_len})", key="so")
        s = st.number_input(f"s ({u_len})", key="s")
        n = st.number_input("No. of lines", key="n", min_value=2)
        dist_val = so + (n-1)*s + (d/2)
        st.success(f"Dist = {dist_val:.2f} {u_len}")

# Main Calculation
points = generate_critical_section(Cx, Cy, dist_val, col_type)
res = calculate_section_properties(points, d)

if res:
    # Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Perimeter (bo)", f"{res['bo']:.2f} {u_len}")
    m2.metric("Area (Ac)", f"{res['Ac']:.2f} {u_len}²")
    m3.metric("Centroid (x, y)", f"({res['Centroid'][0]:.2f}, {res['Centroid'][1]:.2f}) {u_len}")

    # Plot
    px = [p[0] for p in points]; py = [p[1] for p in points]
    if col_type == "Interior": px.append(points[0][0]); py.append(points[0][1])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-Cx/2, Cx/2, Cx/2, -Cx/2, -Cx/2], y=[Cy/2, Cy/2, -Cy/2, -Cy/2, Cy/2],
                             fill="toself", fillcolor="rgba(128,128,128,0.3)", line=dict(color="gray"), name="Column"))
    fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', line=dict(color='blue', width=3), name="Critical Section"))
    fig.add_trace(go.Scatter(x=[res['Centroid'][0]], y=[res['Centroid'][1]], mode='markers',
                             marker=dict(color='red', size=10, symbol='cross'), name="Centroid"))
    fig.update_layout(title="Critical Section Geometry", width=700, height=600, yaxis=dict(scaleanchor="x", scaleratio=1), hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

    # Detailed HTML Report
    st.markdown("---")
    st.subheader("📝 Detailed Analysis Report")
    html_report = generate_html_report(res, Cx, Cy, d, dist_val, u_len, u_inertia)
    st.markdown(html_report, unsafe_allow_html=True)
