import streamlit as st
import plotly.graph_objects as go
from simulation import run_simulation
from db import save_simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.pdfgen import canvas
from io import BytesIO
import base64
from dotenv import load_dotenv
load_dotenv()

# ---------------- PDF REPORT GENERATION ----------------
def create_simulation_report(inputs, results, stratification_ok, discharge_ok, insulation_ok, power_input_ok, status_message):
    """Generate a comprehensive PDF report of the simulation results"""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        textColor=colors.darkblue
    )

    # Title Page
    story.append(Paragraph("Arctic Heat System", title_style))
    story.append(Paragraph("Simulation Report", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Simulation timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))

    # System Status
    story.append(Paragraph("System Status", heading_style))
    status_color = "green" if "System operating" in status_message else "red" if "violation" in status_message.lower() else "orange"
    story.append(Paragraph(f"<font color='{status_color}'>{status_message}</font>", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Parameters Table
    story.append(Paragraph("Simulation Parameters", heading_style))
    param_data = [
        ["Parameter", "Value"],
        ["Ambient Temperature", f"{inputs['ambient_temp']} °C"],
        ["Heat Capture Rate", f"{inputs['capture_rate']/1000:.1f} kW"],
        ["Tank Thermal Mass", f"{inputs['tank_mass']/1000:.1f} tonnes"],
        ["Insulation Efficiency", f"{inputs['insulation_eff']*100:.0f}%"],
        ["Wind Exposure Factor", f"{inputs['wind_factor']:.1f}x"]
    ]

    param_table = Table(param_data)
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(param_table)
    story.append(Spacer(1, 0.1*inch))

    # Compliance Checks
    story.append(Paragraph("Compliance Checks", heading_style))
    compliance_data = [
        ["Requirement", "Status"],
        ["Stratification ≥ 60 K", "PASS" if stratification_ok else "FAIL"],
        ["40 kW discharge for 8h", "PASS" if discharge_ok else "FAIL"],
        ["Insulation R ≥ 5", "PASS" if insulation_ok else "FAIL"],
        ["Input ≤ 50 kW", "PASS" if power_input_ok else "FAIL"]
    ]

    compliance_table = Table(compliance_data)
    compliance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),  # Smaller font
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.green if stratification_ok else colors.red),
        ('TEXTCOLOR', (1, 2), (1, 2), colors.green if discharge_ok else colors.red),
        ('TEXTCOLOR', (1, 3), (1, 3), colors.green if insulation_ok else colors.red),
        ('TEXTCOLOR', (1, 4), (1, 4), colors.green if power_input_ok else colors.red),
    ]))
    story.append(compliance_table)
    story.append(Spacer(1, 0.1*inch))

    # Key Metrics
    story.append(Paragraph("Key Performance Metrics", heading_style))
    metrics_data = [
        ["Metric", "Value"],
        ["Final Average Temperature", f"{results['avg_temp'][-1]:.1f} °C"],
        ["Stored Energy", f"{results['energy'][-1]/1e6:.1f} MJ"],
        ["Heat Capture Rate", f"{inputs['capture_rate']/1000:.1f} kW"],
        ["Final Stratification", f"{(results['top_temp'][-1] - results['bottom_temp'][-1]):.1f} °C"],
        ["Simulation Duration", f"{len(results['avg_temp']) * 5 / 60:.1f} hours"]
    ]

    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(PageBreak())

    # Generate and add plots
    story.append(Paragraph("Detailed Analysis Plots", title_style))
    story.append(Spacer(1, 0.1*inch))

    # Temperature plot
    fig_temp, ax_temp = plt.subplots(figsize=(8, 5))  # Smaller figure
    time_hours = np.arange(len(results['avg_temp'])) * 5 / 60
    ax_temp.plot(time_hours, results['avg_temp'], 'b-', linewidth=1.5, label='Average Temperature')
    ax_temp.plot(time_hours, results['top_temp'], 'r--', linewidth=1, label='Top Layer')
    ax_temp.plot(time_hours, results['bottom_temp'], 'g--', linewidth=1, label='Bottom Layer')
    ax_temp.set_xlabel('Time (hours)', fontsize=8)
    ax_temp.set_ylabel('Temperature (°C)', fontsize=8)
    ax_temp.set_title('Tank Temperature Evolution', fontsize=10)
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend(fontsize=8)
    ax_temp.tick_params(labelsize=8)
    ax_temp.set_xlim(0, time_hours[-1])

    # Save plot to buffer
    temp_buffer = BytesIO()
    fig_temp.savefig(temp_buffer, format='png', dpi=120, bbox_inches='tight')  # Lower DPI
    temp_buffer.seek(0)
    story.append(Image(temp_buffer, width=400, height=250))  # Smaller image
    plt.close(fig_temp)
    story.append(Spacer(1, 0.05*inch))

    # Energy plot
    fig_energy, ax_energy = plt.subplots(figsize=(8, 5))  # Smaller figure
    ax_energy.plot(time_hours, results['energy']/1e6, 'orange', linewidth=1.5, label='Stored Energy')
    ax_energy.set_xlabel('Time (hours)', fontsize=8)
    ax_energy.set_ylabel('Energy (MJ)', fontsize=8)
    ax_energy.set_title('Thermal Energy Storage', fontsize=10)
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(fontsize=8)
    ax_energy.tick_params(labelsize=8)
    ax_energy.set_xlim(0, time_hours[-1])

    energy_buffer = BytesIO()
    fig_energy.savefig(energy_buffer, format='png', dpi=120, bbox_inches='tight')  # Lower DPI
    energy_buffer.seek(0)
    story.append(Image(energy_buffer, width=400, height=250))  # Smaller image
    plt.close(fig_energy)
    story.append(Spacer(1, 0.05*inch))

    # Final stratification plot
    story.append(Paragraph("Final Tank Stratification", heading_style))
    fig_strat, ax_strat = plt.subplots(figsize=(6, 4))  # Smaller figure
    layer_heights = np.linspace(0, 5.0, len(results['layers'][-1]))
    ax_strat.barh(layer_heights, results['layers'][-1], height=0.2, color='lightcoral', alpha=0.7)
    ax_strat.set_xlabel('Temperature (°C)', fontsize=8)
    ax_strat.set_ylabel('Tank Height (m)', fontsize=8)
    ax_strat.set_title('Vertical Temperature Distribution', fontsize=10)
    ax_strat.grid(True, alpha=0.3)
    ax_strat.tick_params(labelsize=8)

    strat_buffer = BytesIO()
    fig_strat.savefig(strat_buffer, format='png', dpi=120, bbox_inches='tight')  # Lower DPI
    strat_buffer.seek(0)
    story.append(Image(strat_buffer, width=300, height=200))  # Smaller image
    plt.close(fig_strat)
    story.append(Spacer(1, 0.05*inch))

    # Detailed Data Tables
    story.append(Paragraph("Detailed Simulation Data", heading_style))

    # Summary statistics
    story.append(Paragraph("Temperature Statistics", styles['Heading3']))
    temp_stats = [
        ["Statistic", "Average Temp (°C)", "Top Layer (°C)", "Bottom Layer (°C)"],
        ["Initial", f"{results['avg_temp'][0]:.1f}", f"{results['top_temp'][0]:.1f}", f"{results['bottom_temp'][0]:.1f}"],
        ["Final", f"{results['avg_temp'][-1]:.1f}", f"{results['top_temp'][-1]:.1f}", f"{results['bottom_temp'][-1]:.1f}"],
        ["Maximum", f"{np.max(results['avg_temp']):.1f}", f"{np.max(results['top_temp']):.1f}", f"{np.max(results['bottom_temp']):.1f}"],
        ["Minimum", f"{np.min(results['avg_temp']):.1f}", f"{np.min(results['top_temp']):.1f}", f"{np.min(results['bottom_temp']):.1f}"],
        ["Range", f"{np.max(results['avg_temp'])-np.min(results['avg_temp']):.1f}",
         f"{np.max(results['top_temp'])-np.min(results['top_temp']):.1f}",
         f"{np.max(results['bottom_temp'])-np.min(results['bottom_temp']):.1f}"]
    ]

    temp_table = Table(temp_stats)
    temp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightcyan),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(temp_table)
    story.append(Spacer(1, 0.1*inch))

    # Energy statistics
    story.append(Paragraph("Energy Statistics", styles['Heading3']))
    energy_stats = [
        ["Statistic", "Energy (MJ)"],
        ["Initial", f"{results['energy'][0]/1e6:.1f}"],
        ["Final", f"{results['energy'][-1]/1e6:.1f}"],
        ["Maximum", f"{np.max(results['energy'])/1e6:.1f}"],
        ["Minimum", f"{np.min(results['energy'])/1e6:.1f}"],
        ["Net Change", f"{(results['energy'][-1]-results['energy'][0])/1e6:.1f}"]
    ]

    energy_table = Table(energy_stats)
    energy_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),  # Smaller font
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(energy_table)
    story.append(Spacer(1, 0.1*inch))

    # Operating modes summary
    mode_counts = pd.Series(results['mode']).value_counts()
    story.append(Paragraph("Operating Mode Summary", styles['Heading3']))
    mode_data = [["Mode", "Hours", "Percentage"]]
    total_steps = len(results['mode'])
    for mode, count in mode_counts.items():
        hours = count * 5 / 60
        percentage = count / total_steps * 100
        mode_data.append([mode, f"{hours:.1f}", f"{percentage:.1f}%"])

    mode_table = Table(mode_data)
    mode_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),  # Smaller font
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(mode_table)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------- SESSION STATE ----------------
if "clear_results" not in st.session_state:
    st.session_state.clear_results = False
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "simulation_inputs" not in st.session_state:
    st.session_state.simulation_inputs = None
if "generate_pdf" not in st.session_state:
    st.session_state.generate_pdf = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Arctic Heat System",
    layout="wide"
)

# ---------------- CSS STYLING ----------------
with open("style.css", "r", encoding="utf-8") as f:
    css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

st.markdown("""
<script>
function stylePlotlyButtons() {
    const buttons = document.querySelectorAll('.updatemenu-button, .updatemenu-btn');
    buttons.forEach(btn => {
        const rect = btn.querySelector('rect');
        if (rect) {
            btn.addEventListener('mouseenter', function() {
                rect.setAttribute('stroke', 'rgba(56, 189, 248, 0.8)');
                rect.setAttribute('stroke-width', '2');
                rect.style.filter = 'drop-shadow(0 0 8px rgba(56, 189, 248, 0.6))';
            });
            btn.addEventListener('mouseleave', function() {
                rect.setAttribute('stroke', 'rgba(148, 163, 184, 0.3)');
                rect.setAttribute('stroke-width', '1.5');
                rect.style.filter = 'none';
            });
        }
    });
}

// Run after plot is rendered
setTimeout(stylePlotlyButtons, 500);
// Also run when new plots are added
const observer = new MutationObserver(stylePlotlyButtons);
observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("❄️ Arctic Heat System Demo")
st.caption("Simulating heat capture, storage, and release under Arctic conditions")

# ---------------- SIDEBAR ----------------
st.sidebar.header("System Parameters")

ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", -60, 0, -40)
capture_rate = st.sidebar.slider("Heat Capture Rate (W)", 1_000, 50_000, 30_000, step=1000)
tank_mass = st.sidebar.slider("Tank Thermal Mass (kg)", 80_000, 120_000, 103_500, step=500)
initial_tank_volume = st.sidebar.slider("Initial Tank Volume (L)", 0, 1000, 500, step=50)
insulation_eff = st.sidebar.slider("Insulation Efficiency (%)", 50, 100, 80) / 100
wind_factor = st.sidebar.slider("Wind Exposure Factor", 0.5, 3.0, 1.2, step=0.1)
simulation_hours = st.sidebar.slider("Simulation Duration (hours)", 1, 48, 24, step=1)

st.sidebar.markdown("---")

col_run, col_reset = st.sidebar.columns(2)

with col_run:
    run_sim = st.button("▶️ Run Simulation", use_container_width=True, type="primary")

with col_reset:
    reset_sim = st.button("🔄 Reset", use_container_width=True, type="secondary")

if reset_sim:
    st.session_state.clear_results = True
    st.session_state.simulation_results = None
    st.session_state.simulation_inputs = None
    st.rerun()

# ---------------- RUN SIMULATION ----------------
if run_sim or (st.session_state.simulation_results is not None and not st.session_state.clear_results):

    # Only run simulation if button was clicked or if we need to regenerate
    if run_sim:
        total_time_seconds = simulation_hours * 3600
        steps = int(total_time_seconds / 300)  # Assuming 5-minute time steps

        results = run_simulation(
            ambient_temp,
            capture_rate,
            tank_mass,
            insulation_eff,
            wind_factor,
            simulation_hours
        )

        inputs = {
            "ambient_temp": ambient_temp,
            "capture_rate": capture_rate,
            "tank_mass": tank_mass,
            "initial_tank_volume": initial_tank_volume,
            "insulation_eff": insulation_eff,
            "wind_factor": wind_factor
        }

        # Store in session state
        st.session_state.simulation_results = results
        st.session_state.simulation_inputs = inputs

        save_simulation(inputs, results["avg_temp"], results["energy"])

    # Use stored results if available
    results = st.session_state.simulation_results
    inputs = st.session_state.simulation_inputs

    # ---------------- METRICS (COMPACT) ----------------
    # Calculate additional metrics
    total_energy_input = capture_rate * len(results['avg_temp']) * 300  # Total energy input in Joules
    net_energy_change = results['energy'][-1] - results['energy'][0]  # Net energy stored in Joules
    system_efficiency = (net_energy_change / total_energy_input) * 100 if total_energy_input > 0 else 0

    # Estimate heat loss (simplified - would need more sophisticated calculation in real simulation)
    avg_temp_diff = np.mean(results['avg_temp'] - ambient_temp)
    estimated_heat_loss = total_energy_input - net_energy_change  # Simplified estimate
    heat_loss_mw = estimated_heat_loss / (len(results['avg_temp']) * 300) / 1e6  # MW

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("**Final Temp**", f"{results['avg_temp'][-1]:.1f}°C")
    with col2:
        st.metric("**Energy Stored**", f"{results['energy'][-1] / 1e6:.1f} MJ")
    with col3:
        st.metric("**Efficiency**", f"{system_efficiency:.1f}%")
    with col4:
        st.metric("**Heat Loss**", f"{heat_loss_mw:.2f} MW")
    with col5:
        st.metric("**Energy Input**", f"{capture_rate / 1000:.1f} kW")
    with col6:
        st.metric("**Duration**", f"{len(results['avg_temp']) * 5 / 60:.1f}h")

    st.markdown("---")

    # ---------------- SYSTEM STATUS (TOP) ----------------
    stratification_ok = np.all(
    (results["top_temp"] - results["bottom_temp"] >= 60)
    | (results["mode"] != "DISCHARGE")
)


    discharge_hours = np.sum(
        results["discharge_power"] >= 40_000
    ) * 5 / 60
    discharge_ok = discharge_hours >= 8

    insulation_ok = insulation_eff >= 0.7
    power_input_ok = capture_rate <= 50_000
    alarm_ok = not results["alarm"]

    # Determine status level and message
    if not alarm_ok:
        status_level = "error"
        status_color = "red"
        status_message = "🚨 Alarm triggered — system shut down for safety."
    elif not stratification_ok:
        status_level = "error"
        status_color = "red"
        status_message = "Stratification requirement violated."
    elif not discharge_ok or not insulation_ok or not power_input_ok:
        status_level = "warning"
        status_color = "yellow"
        status_message = "Some system requirements not fully met."
    else:
        status_level = "success"
        status_color = "green"
        status_message = "System operating within all required parameters."

    # Create glowing status box
    glow_styles = {
        "red": "0 0 20px rgba(239, 68, 68, 0.8), 0 0 40px rgba(239, 68, 68, 0.4)",
        "yellow": "0 0 20px rgba(234, 179, 8, 0.8), 0 0 40px rgba(234, 179, 8, 0.4)",
        "green": "0 0 20px rgba(34, 197, 94, 0.8), 0 0 40px rgba(34, 197, 94, 0.4)"
    }
    
    border_colors = {
        "red": "rgba(239, 68, 68, 0.6)",
        "yellow": "rgba(234, 179, 8, 0.6)",
        "green": "rgba(34, 197, 94, 0.6)"
    }

    st.markdown(f"""
    <div style="
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid {border_colors[status_color]};
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: {glow_styles[status_color]};
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6edf3;
    ">
        🧊 System Status: {status_message}
    </div>
    """, unsafe_allow_html=True)

    # ---------------- COMPLIANCE CHECKS ---------------- 
    st.subheader("✅ System Requirements Compliance")

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Stratification ≥ 60 K", "PASS" if stratification_ok else "FAIL")
    colB.metric("40 kW for 8h", "PASS" if discharge_ok else "FAIL")
    colC.metric("Insulation R ≥ 5", "PASS" if insulation_ok else "FAIL")
    colD.metric("Input ≤ 50 kW", "PASS" if power_input_ok else "FAIL")

    st.markdown("---")

    # ---------------- TEMPERATURE AND ENERGY PLOTS (SIDE BY SIDE) ---------------- 
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            y=results["avg_temp"],
            mode="lines",
            name="Average Tank Temperature (°C)",
            line=dict(width=3)
        ))

        fig_temp.update_layout(
            title="Tank Temperature Over Time",
            xaxis_title="Time Step (5 min)",
            yaxis_title="Temperature (°C)",
            template="plotly_dark"
        )

        st.plotly_chart(fig_temp, use_container_width=True)

    with plot_col2:
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            y=results["energy"] / 1e6,
            mode="lines",
            name="Stored Energy (MJ)",
            line=dict(width=3)
        ))

        fig_energy.update_layout(
            title="Stored Thermal Energy Over Time",
            xaxis_title="Time Step (5 min)",
            yaxis_title="Energy (MJ)",
            template="plotly_dark"
        )

        st.plotly_chart(fig_energy, use_container_width=True)

    # ---------------- ANIMATED TANK STRATIFICATION ----------------
    st.subheader("🌀 Animated Tank Stratification Over Time")
    FRAME_SKIP = 4

    # Show only temperature range
    temp_range = np.max(results['layers']) - np.min(results['layers'])
    st.caption(f"Temperature range: {np.min(results['layers']):.1f}°C to {np.max(results['layers']):.1f}°C")

    # Prepare data for contour plot: time (x) x height (y) with temperature (z)
    num_layers = len(results["layers"][0])
    num_steps = len(results["layers"])
    time_hours = np.arange(num_steps) * 5 / 60  # Convert 5-min steps to hours

    # Create height coordinates (from bottom to top of tank)
    layer_heights = np.linspace(0, 5.0, num_layers)  # Tank height = 5m

    # Create 2D temperature array: height x time (for contour plot)
    temp_data = np.array(results["layers"])  # Shape: (time_steps, layers)
    temp_data_T = temp_data.T  # Transpose to (layers, time_steps)

    # Create frames for animation - each frame adds one more time step
    frames = []
    for t in range(1, num_steps + 1, FRAME_SKIP):
        frames.append(
            go.Frame(
                data=[
                    go.Contour(
                        z=temp_data_T[:, :t],  # Show data up to current time
                        x=time_hours[:t],      # Time axis
                        y=layer_heights,       # Height axis
                        colorscale=[[0, "blue"], [0.3, "cyan"], [0.5, "yellow"], [0.7, "orange"], [1, "red"]],
                        showscale=True,
                        contours=dict(
                            showlines=False,  # Smoother appearance
                            showlabels=False
                        ),
                        colorbar=dict(title="Temperature (°C)"),
                        zmin=np.min(temp_data) - 5,
                        zmax=np.max(temp_data) + 5
                    )
                ],
                name=str(t-1)
            )
        )

    # Initial figure with first time step
    fig_anim = go.Figure(
        data=[
            go.Contour(
                z=temp_data_T[:, :1],
                x=time_hours[:1],
                y=layer_heights,
                colorscale=[[0, "blue"], [0.3, "cyan"], [0.5, "yellow"], [0.7, "orange"], [1, "red"]],
                showscale=True,
                contours=dict(
                    showlines=False,
                    showlabels=False
                ),
                colorbar=dict(title="Temperature (°C)"),
                zmin=np.min(temp_data) - 5,
                zmax=np.max(temp_data) + 5
            )
        ],
        layout=go.Layout(
            title="STES Tank Thermal Stratification Evolution",
            xaxis=dict(
                title="Time (hours)",
                range=[0, time_hours[-1]]
            ),
            yaxis=dict(
                title="Tank Height (m)",
                range=[0, 5.0]
            ),
            template="plotly_dark",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": True,
                    "direction": "down",
                    "pad": {"r": 10, "t": 10},
                    "x": 0.01,
                    "xanchor": "left",
                    "y": 0.5,
                    "yanchor": "middle",
                    "bgcolor": "rgba(0,0,0,0.8)",
                    "bordercolor": "rgba(148,163,184,0.3)",
                    "borderwidth": 1.5,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[], {"frame": {"duration": 0}, "mode": "immediate"}]
                        }
                    ]
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "method": "animate",
                            "args": [[str(k)], {"mode": "immediate"}],
                            "label": f"{time_hours[k]:.2f} h"
                        }
                        for k in range(0, num_steps, FRAME_SKIP)
                    ],
                    "active": 0,
                    "currentvalue": {"prefix": "Time: "},
                }
            ]
        ),
        frames=frames
    )

    st.plotly_chart(fig_anim, use_container_width=True)

    # ---------------- STRATIFIED TANK VIEW ----------------
    st.subheader("🛢️ Tank Stratification (Final State)")

    final_layers = results["layers"][-1]
    layer_heights = np.linspace(0, 5.0, len(final_layers))
    height_labels = [f"{h:.1f} m" for h in layer_heights]

    fig_layers = go.Figure()
    fig_layers.add_trace(go.Bar(
        x=final_layers,
        y=height_labels,
        orientation="h",
        marker=dict(
            color=final_layers,
            colorscale=[[0, "blue"], [0.3, "cyan"], [0.5, "yellow"], [0.7, "orange"], [1, "red"]],
            showscale=True,
            colorbar=dict(title="Temperature (°C)")
        )
    ))

    fig_layers.update_layout(
        title="Vertical Temperature Distribution (Final State)",
        xaxis_title="Temperature (°C)",
        yaxis_title="Tank Height (m)",
        template="plotly_dark"
    )

    st.plotly_chart(fig_layers, use_container_width=True)

    # ---------------- PDF REPORT DOWNLOAD ----------------
    if st.button("📊 Generate PDF Report", use_container_width=False, type="secondary"):
        try:
            with st.spinner("Generating PDF report..."):
                pdf_buffer = create_simulation_report(
                    inputs, results, stratification_ok, discharge_ok,
                    insulation_ok, power_input_ok, status_message
                )

                # Convert to base64 for download
                pdf_data = pdf_buffer.getvalue()
                pdf_size_kb = len(pdf_data) / 1024  # Convert to KB

                b64_pdf = base64.b64encode(pdf_data).decode()

                # Create download link
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="arctic_heat_simulation_report.pdf" style="text-decoration: none;"><button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">📥 Download PDF Report ({pdf_size_kb:.1f} KB)</button></a>'

                st.success("PDF Report Generated Successfully!")
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")


# ---------------- RESET MESSAGE ----------------
if st.session_state.clear_results:
    st.info("Simulation reset. Adjust parameters and run again.")
    st.session_state.clear_results = False
