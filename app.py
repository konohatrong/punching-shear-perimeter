import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF

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
    
    detailed_segments = []

    for idx, seg in enumerate(segments):
        xi = seg['p1'][0] - x_bar
        yi = seg['p1'][1] - y_bar
        xj = seg['p2'][0] - x_bar
        yj = seg['p2'][1] - y_bar
        l = seg['l']
        
        x_shifted_all.extend([xi, xj])
        y_shifted_all.extend([yi, yj])
        
        # Calculation Terms
        term_y_val = (yi**2 + yi*yj + yj**2)
        jcx_part = d * (l/3) * term_y_val
        
        term_x_val = (xi**2 + xi*xj + xj**2)
        jcy_part = d * (l/3) * term_x_val
        
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

    c_vals = {
        "cx_pos": max(x_shifted_all), "cx_neg": min(x_shifted_all),
        "cy_pos": max(y_shifted_all), "cy_neg": min(y_shifted_all)
    }

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
# 3. PDF REPORT GENERATOR (FPDF2)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Punching Shear Analysis Report", ln=True)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Reference Standard: ACI 421.1R-20 Appendix B", ln=True)
        self.ln(10) # Line break

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(res, Cx, Cy, d, dist_val, u_len, u_inertia, u_area):
    pdf = PDFReport(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- 1. Geometric Parameters ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. Geometric Parameters", border="B", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 6, "Column Dimensions:", ln=0)
    pdf.cell(0, 6, f"Cx = {Cx} {u_len}, Cy = {Cy} {u_len}", ln=True)
    
    pdf.cell(50, 6, "Effective Depth:", ln=0)
    pdf.cell(0, 6, f"d = {d} {u_len}", ln=True)
    
    pdf.cell(50, 6, "Critical Section Dist:", ln=0)
    pdf.cell(0, 6, f"{dist_val:.2f} {u_len} (from face)", ln=True)
    pdf.ln(5)
    
    # --- 2. Properties of Critical Section ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Properties of Critical Section", border="B", ln=True)
    pdf.ln(3)
    
    # Perimeter / Area
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 6, f"Perimeter (bo):", ln=0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, f"{res['bo']:.2f} {u_len}", ln=0)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(30, 6, f"Area (Ac):", ln=0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, f"{res['Ac']:.2f} {u_area}", ln=True)
    
    # Centroid
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Centroid (Relative to Column Center):", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(10, 6, "", ln=0) # Indent
    pdf.cell(0, 6, f"X-bar = {res['Centroid'][0]:.2f} {u_len},  Y-bar = {res['Centroid'][1]:.2f} {u_len}", ln=True)
    
    # Extreme Fibers (Box)
    pdf.ln(2)
    pdf.set_fill_color(249, 249, 249)
    pdf.set_font("Helvetica", "", 9)
    # Draw a box with text inside
    box_content = (
        f"Extreme Fiber Distances (from Centroid):\n"
        f"  Horizontal: cx_left = {res['extreme']['cx_neg']:.2f}, cx_right = {res['extreme']['cx_pos']:.2f} {u_len}\n"
        f"  Vertical: cy_bot = {res['extreme']['cy_neg']:.2f}, cy_top = {res['extreme']['cy_pos']:.2f} {u_len}"
    )
    pdf.multi_cell(0, 5, box_content, border=1, fill=True)
    pdf.ln(5)
    
    # --- 3. Moment of Inertia Calculation ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Moment of Inertia Calculation (Jc)", border="B", ln=True)
    pdf.ln(2)
    
    # Equations
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    eq_text = (
        "Governing Equations (ACI 421.1R-20 Eq. B.8 & B.9):\n"
        "Values are calculated using the summation of segments relative to the centroid.\n"
        "Jcx = d * Sum [ (l/3) * (yi^2 + yi*yj + yj^2) ]\n"
        "Jcy = d * Sum [ (l/3) * (xi^2 + xi*xj + xj^2) ]"
    )
    pdf.multi_cell(0, 5, eq_text, border=0)
    pdf.ln(3)
    
    # TABLE Header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(50, 50, 50)
    pdf.set_text_color(255, 255, 255)
    
    # Table Column Widths [Seg, l, xi, yi, xj, yj, Jcx, Jcy]
    col_w = [10, 20, 20, 20, 20, 20, 35, 35] 
    headers = ["Seg", "Length", "xi", "yi", "xj", "yj", "Jcx Contrib.", "Jcy Contrib."]
    
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align='C')
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    
    fill = False
    for seg in res['detailed_segments']:
        pdf.cell(col_w[0], 6, str(seg['id']), border=1, align='C', fill=fill)
        pdf.cell(col_w[1], 6, f"{seg['l']:.2f}", border=1, align='R', fill=fill)
        
        pdf.set_text_color(100, 100, 100) # Gray for coords
        pdf.cell(col_w[2], 6, f"{seg['xi']:.2f}", border=1, align='R', fill=fill)
        pdf.cell(col_w[3], 6, f"{seg['yi']:.2f}", border=1, align='R', fill=fill)
        pdf.cell(col_w[4], 6, f"{seg['xj']:.2f}", border=1, align='R', fill=fill)
        pdf.cell(col_w[5], 6, f"{seg['yj']:.2f}", border=1, align='R', fill=fill)
        
        pdf.set_text_color(0, 0, 0) # Back to black
        # Highlight background for J values lightly
        pdf.set_fill_color(240, 248, 255) # AliceBlue
        pdf.cell(col_w[6], 6, f"{seg['jcx']:,.0f}", border=1, align='R', fill=True)
        pdf.set_fill_color(255, 240, 240) # LavenderBlush
        pdf.cell(col_w[7], 6, f"{seg['jcy']:,.0f}", border=1, align='R', fill=True)
        pdf.ln()
        
    # Total Row
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(sum(col_w[:6]), 7, "Total Summation : ", border=1, align='R', fill=True)
    pdf.cell(col_w[6], 7, f"{res['Jcx']:,.0f}", border=1, align='R', fill=True)
    pdf.cell(col_w[7], 7, f"{res['Jcy']:,.0f}", border=1, align='R', fill=True)
    pdf.ln(10)

    # --- 4. Final Section Properties ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Final Section Properties", border="B", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "", 10)
    
    # Final Result Table
    # Jcx
    pdf.set_fill_color(240, 248, 255)
    pdf.cell(60, 8, "Jcx (Major Axis Inertia):", border=1, fill=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, f"{res['Jcx']:,.2f} {u_inertia}", border=1, fill=True, align='R')
    pdf.ln()
    
    # Jcy
    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(255, 240, 240)
    pdf.cell(60, 8, "Jcy (Minor Axis Inertia):", border=1, fill=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, f"{res['Jcy']:,.2f} {u_inertia}", border=1, fill=True, align='R')
    pdf.ln()
    
    # Jxy
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 8, "Jxy (Product of Inertia):", border=1)
    pdf.cell(60, 8, f"{res['Jxy']:,.2f} {u_inertia}", border=1, align='R')
    pdf.ln()

    return bytes(pdf.output())

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
    if "Imperial" in unit_sys: u_len, u_area, u_inertia = "in", "in²", "in⁴"
    else: u_len, u_area, u_inertia = "mm", "mm²", "mm⁴"

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
    m2.metric("Area (Ac)", f"{res['Ac']:.2f} {u_area}")
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

    # Coordinates Table (Simple View On-Screen)
    st.subheader("📍 Critical Section Coordinates")
    coords_data = [{"Point": i+1, f"X ({u_len})": p[0], f"Y ({u_len})": p[1]} for i, p in enumerate(points)]
    st.dataframe(pd.DataFrame(coords_data).set_index("Point").style.format("{:.2f}"))

    # Download PDF Button
    st.markdown("---")
    pdf_bytes = generate_pdf_report(res, Cx, Cy, d, dist_val, u_len, u_inertia, u_area)
    
    st.download_button(
        label="📥 Download Analysis Report (PDF A4)",
        data=pdf_bytes,
        file_name="Punching_Shear_Report_ACI421.pdf",
        mime="application/pdf"
    )
