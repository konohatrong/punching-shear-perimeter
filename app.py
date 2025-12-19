import streamlit as st
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CORE CALCULATION ENGINE (Appendix B)
# ==========================================
def calculate_section_properties(points, d):
    """
    คำนวณ Properties ของหน้าตัดวิกฤตแบบ Polygon ตาม ACI 421.1R-20 Appendix B
    """
    # 1.1 หาเส้นรอบรูปและจุด Centroid (Line Properties)
    total_length = 0
    sum_mx = 0 # Moment about X-axis (integral y dl)
    sum_my = 0 # Moment about Y-axis (integral x dl)
    
    segments = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        l = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if l == 0: continue
            
        # จุดกึ่งกลางของ segment
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        
        total_length += l
        sum_mx += l * ym
        sum_my += l * xm
        segments.append({'p1': (x1,y1), 'p2': (x2,y2), 'l': l, 'xm': xm, 'ym': ym})
        
    if total_length == 0: return None
    
    # Centroid ของเส้นรอบรูป (เทียบกับ Center เสา)
    x_bar = sum_my / total_length
    y_bar = sum_mx / total_length
    
    # 1.2 คำนวณ Inertia รอบแกน Centroid ใหม่ (Shifted Axes)
    # อ้างอิงสมการ B.8, B.9, B.11
    Jcx_c = 0
    Jcy_c = 0
    Jxy_c = 0
    
    for seg in segments:
        # ย้ายพิกัดเทียบกับ Centroid ของหน้าตัดวิกฤต
        x1_p = seg['p1'][0] - x_bar
        y1_p = seg['p1'][1] - y_bar
        x2_p = seg['p2'][0] - x_bar
        y2_p = seg['p2'][1] - y_bar
        l = seg['l']
        
        # Eq B.8: Jcx (รอบแกน X)
        term_y = (y1_p**2 + y1_p*y2_p + y2_p**2)
        Jcx_c += d * (l/3) * term_y
        
        # Eq B.9: Jcy (รอบแกน Y)
        term_x = (x1_p**2 + x1_p*x2_p + x2_p**2)
        Jcy_c += d * (l/3) * term_x
        
        # Eq B.11: Jxy (Product of Inertia)
        term_xy = (2*x1_p*y1_p + x1_p*y2_p + x2_p*y1_p + 2*x2_p*y2_p)
        Jxy_c += d * (l/6) * term_xy

    # 1.3 คำนวณ Principal Moments (สำหรับกรณีไม่สมมาตร)
    avg_J = (Jcx_c + Jcy_c) / 2
    diff_J = (Jcx_c - Jcy_c) / 2
    radius = math.sqrt(diff_J**2 + Jxy_c**2)
    
    J_max = avg_J + radius # Major Axis
    J_min = avg_J - radius # Minor Axis
    
    # หา Principal Angle (Theta)
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
    สร้างจุดพิกัด (Polygon Points) ตามชนิดเสาและระยะวิกฤต
    อ้างอิง ACI 421 Fig. B (Octagon logic)
    """
    hx = Cx / 2
    hy = Cy / 2
    
    # ขอบเขตด้านนอก (Outer boundaries)
    X_far = hx + dist
    Y_far = hy + dist
    
    points = []
    
    if col_type == "Interior":
        # รูป 8 เหลี่ยม (Octagon) - D.1 Logic
        points = [
            (-hx, Y_far), (hx, Y_far),   # Top Edge
            (X_far, hy), (X_far, -hy),   # Right Edge
            (hx, -Y_far), (-hx, -Y_far), # Bottom Edge
            (-X_far, -hy), (-X_far, hy), # Left Edge
            (-hx, Y_far)                 # Close Loop
        ]
        
    elif col_type == "Edge (Left Free)":
        # รูปตัว C เปิดซ้าย (Hexagon open) - D.2 Logic
        points = [
            (-hx, Y_far),    # Start Top-Left (at Free Edge)
            (hx, Y_far),     # Top Inner Corner
            (X_far, hy),     # Top-Right Chamfer
            (X_far, -hy),    # Bot-Right Chamfer
            (hx, -Y_far),    # Bot Inner Corner
            (-hx, -Y_far)    # End Bot-Left (at Free Edge)
        ]
        
    elif col_type == "Corner (Top-Left Free)":
        # รูปตัว L เปิดบน-ซ้าย (Pentagon open) - D.3 Logic
        points = [
            (X_far, hy),     # Start Top-Right (at Free Edge)
            (X_far, -hy),    # Right-Bot Chamfer
            (hx, -Y_far),    # Bot Inner Corner
            (-hx, -Y_far)    # End Bot-Left (at Free Edge)
        ]
        
    return points

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="ACI 421 Punching Shear", page_icon="🏗️", layout="wide")

st.title("🏗️ ACI 421.1R-20 Punching Shear Calculator")
st.markdown("""
เครื่องมือวิเคราะห์หาค่า **Section Properties ($J_c, A_c, b_o$)** สำหรับหน้าตัดวิกฤตแรงเฉือนทะลุ 
รองรับกรณี **Interior, Edge, Corner** โดยใช้วิธี **General Polygon (Appendix B)**
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. ข้อมูลเสา (Column Data)")
    col_type = st.selectbox("ตำแหน่งเสา (Column Type)", 
                            ["Interior", "Edge (Left Free)", "Corner (Top-Left Free)"])
    
    c1, c2 = st.columns(2)
    Cx = c1.number_input("กว้าง Cx (in.)", value=12.0, step=1.0)
    Cy = c2.number_input("ลึก Cy (in.)", value=20.0, step=1.0)
    d = st.number_input("ความลึกประสิทธิผล d (in.)", value=5.62, step=0.1)
    
    st.markdown("---")
    st.header("2. หน้าตัดวิกฤต (Critical Section)")
    calc_mode = st.radio("เลือกโหมดคำนวณ:", 
                         ["ระบุระยะเอง (Trial)", "คำนวณจากเหล็กเสริม (Studs)"])
    
    dist_val = 0.0
    if calc_mode == "ระบุระยะเอง (Trial)":
        # Default Trial Guess for D.1 is 25.3
        def_val = 25.3 if col_type == "Interior" else d/2
        dist_val = st.number_input("ระยะจากผิวเสา (in.)", value=def_val)
        st.caption(f"Note: d/2 = {d/2:.2f} in.")
    else:
        st.subheader("รายละเอียด Stud Layout")
        so = st.number_input("s0 (ระยะแถวแรก) (in.)", value=2.25)
        s = st.number_input("s (ระยะห่างแถว) (in.)", value=2.75)
        n = st.number_input("จำนวนแถว (No. of lines)", value=9, min_value=2, step=1)
        
        # Distance = s0 + (n-1)s + d/2
        dist_val = so + (n-1)*s + (d/2)
        st.success(f"ระยะวิกฤตคำนวณได้ = {dist_val:.2f} in.")

# --- Main Calculation ---
points = generate_critical_section(Cx, Cy, dist_val, col_type)
res = calculate_section_properties(points, d)

if res:
    # --- Results Display ---
    st.subheader(f"ผลลัพธ์การวิเคราะห์: {col_type}")
    st.info(f"หน้าตัดวิกฤตที่ระยะ **{dist_val:.2f} in.** จากผิวเสา")
    
    # Metrics Row 1
    m1, m2, m3 = st.columns(3)
    m1.metric("เส้นรอบรูป (bo)", f"{res['bo']:.2f} in.")
    m2.metric("พื้นที่หน้าตัด (Ac)", f"{res['Ac']:.2f} in.²")
    m3.metric("จุด Centroid (x, y)", f"({res['Centroid'][0]:.2f}, {res['Centroid'][1]:.2f})", 
              help="ระยะเทียบกับจุดศูนย์กลางเสา (0,0)")

    st.markdown("---")
    
    # J Values (Principal vs Orthogonal)
    st.subheader("Moment of Inertia (J)")
    
    # Logic to highlight Principal Moments if Jxy is significant (Corner Case)
    is_unsymmetric = abs(res['Jxy']) > 1.0 # Tolerance
    
    c1, c2 = st.columns(2)
    
    if is_unsymmetric:
        st.warning("⚠️ หน้าตัดไม่สมมาตร (Corner Column): แนะนำให้ใช้ค่า Principal Moments")
        c1.metric("J_major (Principal)", f"{res['J_major']:,.2f} in.⁴")
        c2.metric("J_minor (Principal)", f"{res['J_minor']:,.2f} in.⁴")
        st.caption(f"มุมหมุนแกน Principal (θ) = {res['theta_deg']:.2f}°")
        
        with st.expander("ดูค่า J ในแกน X, Y ปกติ"):
            st.write(f"Jcx: {res['Jcx']:,.2f}")
            st.write(f"Jcy: {res['Jcy']:,.2f}")
            st.write(f"Jxy: {res['Jxy']:,.2f}")
    else:
        st.success("✅ หน้าตัดสมมาตร (Symmetric): ใช้แกน X, Y ได้เลย")
        c1.metric("Jcx (Major Axis)", f"{res['Jcx']:,.2f} in.⁴")
        c2.metric("Jcy (Minor Axis)", f"{res['Jcy']:,.2f} in.⁴")

    # --- Visual Plot ---
    st.markdown("---")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot Segments
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    # ปิดรูปถ้าเป็น Interior
    if col_type == "Interior":
        px.append(points[0][0])
        py.append(points[0][1])
        
    ax.plot(px, py, 'b-', linewidth=2, label='Critical Section')
    ax.fill(px, py, 'blue', alpha=0.1)
    
    # Plot Column Box
    cx_pts = [-Cx/2, Cx/2, Cx/2, -Cx/2, -Cx/2]
    cy_pts = [Cy/2, Cy/2, -Cy/2, -Cy/2, Cy/2]
    ax.fill(cx_pts, cy_pts, 'gray', alpha=0.5, label='Column')
    
    # Plot Centroid
    ax.plot(res['Centroid'][0], res['Centroid'][1], 'ro', markersize=8, label='Centroid')
    ax.plot(0, 0, 'k+', markersize=10, label='Col Center')
    
    # Decoration
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title(f"Critical Section Geometry ({col_type})")
    ax.set_xlabel("Distance X (in.)")
    ax.set_ylabel("Distance Y (in.)")
    
    st.pyplot(fig)