import streamlit as st
import plotly.express as px

st.set_page_config(page_title="CyberSecurity ML", layout="wide", page_icon="🛡️")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<div style='padding: 10px 0px;'><h2 style='color:white; margin:0; font-size:22px;'>🛡️ CyberSecurity ML</h2><p style='color:#94a3b8; font-size:12px; margin-top:5px;'>Intrusion Detection & Threat Prevention</p><hr style='border-color:#1f2937; margin: 20px 0;'><div class='sidebar-btn active'>🏠 Home</div><div class='sidebar-btn'>⚡ Make Prediction</div><div class='sidebar-btn'>📊 Model Performance</div><div class='sidebar-btn'>⏱️ Class Distribution</div><div class='sidebar-btn'>ℹ️ About Project</div><br><br><br><br><br><br><div style='position: absolute; bottom: 20px;'><div style='color:#10b981; font-weight:bold; font-size:14px;'>🟢 System Ready</div><div style='color:#64748b; font-size:12px;'>Streamlit App | NSL-KDD Dataset</div></div></div>", unsafe_allow_html=True)

# ==========================================
# HEADER ROW
# ==========================================
st.markdown("<div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 20px;'><div style='display:flex; gap:15px;'><div style='font-size: 45px;'>🛡️</div><div><h1 style='color:white; margin:0; font-size:32px;'>Network Intrusion Detection System</h1><h3 style='color:#cbd5e1; margin:0; font-weight:500;'>Machine Learning Based Cybersecurity</h3></div></div><div style='background-color: #064e3b; border: 1px solid #10b981; padding: 15px 20px; border-radius: 10px; display:flex; align-items:center; gap:15px;'><div style='background-color:#10b981; color:white; border-radius:50%; width:35px; height:35px; display:flex; justify-content:center; align-items:center; font-size:18px;'>🛡️</div><div><div style='color:#d1fae5; font-size:12px;'>Model Used</div><div style='color:white; font-weight:bold; font-size:16px;'>Original XGBoost</div><div style='color:#a7f3d0; font-size:11px;'>Highest accuracy | Lowest false alarms</div></div></div></div><p style='color:#94a3b8; font-size:15px; margin-bottom: 30px;'>This application uses a trained XGBoost model to classify network connections as normal or an attack and identify the type of attack. Enter the network features below to get a prediction.</p>", unsafe_allow_html=True)

# ==========================================
# MAIN CONTENT SPLIT
# ==========================================
col_left, col_right = st.columns([1.5, 1])

with col_left:
    # Prediction Form Card (Single line HTML)
    st.markdown("<div class='dark-card'><h3 style='color:white; margin-top:0;'>Make a Prediction</h3><p style='color:#94a3b8; font-size:14px;'>Network Connection Features (41 features)</p><form><div style='display:grid; grid-template-columns: repeat(4, 1fr); gap: 15px;'><div><label class='custom-label'>duration</label><input type='number' class='custom-input' value='0.0'></div><div><label class='custom-label'>protocol_type</label><select class='custom-input'><option>tcp</option><option>udp</option></select></div><div><label class='custom-label'>service</label><select class='custom-input'><option>http</option><option>ftp</option></select></div><div><label class='custom-label'>flag</label><select class='custom-input'><option>SF</option><option>S0</option></select></div><div><label class='custom-label'>src_bytes</label><input type='number' class='custom-input' value='181'></div><div><label class='custom-label'>dst_bytes</label><input type='number' class='custom-input' value='5450'></div><div><label class='custom-label'>land</label><input type='number' class='custom-input' value='0'></div><div><label class='custom-label'>wrong_fragment</label><input type='number' class='custom-input' value='0'></div><div><label class='custom-label'>urgent</label><input type='number' class='custom-input' value='0'></div><div><label class='custom-label'>hot</label><input type='number' class='custom-input' value='0'></div><div><label class='custom-label'>num_failed_logins</label><input type='number' class='custom-input' value='0'></div><div><label class='custom-label'>logged_in</label><input type='number' class='custom-input' value='1'></div></div><p style='color:#94a3b8; font-size:12px; margin-top:15px;'>ℹ️ All 41 features are required. You can use the dataset or enter values manually.</p><div style='display:grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;'><button type='button' class='btn-predict'>▶ Predict</button><button type='button' class='btn-reset'>🔄 Reset</button></div></form></div>", unsafe_allow_html=True)

    # Metrics Row (Single line HTML)
    st.markdown("<div style='display:grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px;'><div class='dark-card' style='padding:15px;'><span style='font-size:20px;'>🗄️</span><br><span style='color:#94a3b8; font-size:12px;'>Total Classes</span><br><span style='color:#a78bfa; font-size:24px; font-weight:bold;'>28</span></div><div class='dark-card' style='padding:15px;'><span style='font-size:20px;'>⚙️</span><br><span style='color:#94a3b8; font-size:12px;'>Total Features</span><br><span style='color:#60a5fa; font-size:24px; font-weight:bold;'>41</span></div><div class='dark-card' style='padding:15px;'><span style='font-size:20px;'>🎯</span><br><span style='color:#94a3b8; font-size:12px;'>Model Accuracy</span><br><span style='color:#34d399; font-size:24px; font-weight:bold;'>99.73%</span></div><div class='dark-card' style='padding:15px;'><span style='font-size:20px;'>🏆</span><br><span style='color:#94a3b8; font-size:12px;'>Avg. Macro F1</span><br><span style='color:#fbbf24; font-size:24px; font-weight:bold;'>0.867</span></div></div>", unsafe_allow_html=True)


with col_right:
    # Green Result Box (Single line HTML)
    st.markdown("<div style='background-color: #064e3b; border: 1px solid #10b981; border-radius: 10px; padding: 0; margin-bottom: 20px;'><div style='background-color: #047857; padding: 12px 20px; border-top-left-radius: 10px; border-top-right-radius: 10px; color:white; font-weight:bold;'>🛡️ Prediction Result</div><div style='padding: 20px;'><div style='display:flex; align-items:center; gap:15px;'><div style='background-color:#10b981; color:white; border-radius:50%; width:40px; height:40px; display:flex; justify-content:center; align-items:center; font-size:20px;'>✓</div><div><h2 style='color:white; margin:0; font-size:22px;'>Normal Traffic</h2><p style='color:#d1fae5; margin:0; font-size:13px;'>This connection is classified as normal network traffic.</p></div></div><div style='display:flex; justify-content:space-between; margin-top:20px; color:white; font-size:14px; font-weight:600;'><span>Confidence</span><span>94.2%</span></div><div class='progress-bg'><div class='progress-fill' style='background-color:#10b981; width:94.2%;'></div></div><hr style='border-color: #065f46; margin: 15px 0;'><div style='display:flex; justify-content:space-between; color:#a7f3d0; font-size:12px;'><div>⏱️ Prediction Time<br><span style='color:white; font-size:13px;'>0.023s</span></div><div style='text-align:right;'>Model<br><span style='color:white; font-size:13px;'>Original XGBoost</span></div></div></div></div>", unsafe_allow_html=True)

    # Class Probabilities Box (Single line HTML)
    st.markdown("<div class='dark-card'><h4 style='color:white; margin-top:0; margin-bottom:15px;'>Class Probabilities</h4><div style='display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1;'><span>normal</span><span>0.942</span></div><div class='progress-bg'><div class='progress-fill' style='background-color:#10b981; width:94.2%;'></div></div><div style='display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1;'><span>neptune</span><span>0.021</span></div><div class='progress-bg'><div class='progress-fill' style='background-color:#3b82f6; width:2.1%;'></div></div><div style='display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1;'><span>satan</span><span>0.012</span></div><div class='progress-bg'><div class='progress-fill' style='background-color:#3b82f6; width:1.2%;'></div></div><div style='display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1;'><span>ipsweep</span><span>0.008</span></div><div class='progress-bg'><div class='progress-fill' style='background-color:#8b5cf6; width:0.8%;'></div></div><div style='display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1;'><span>portsweep</span><span>0.006</span></div><div class='progress-bg'><div class='progress-fill' style='background-color:#ef4444; width:0.6%;'></div></div></div>", unsafe_allow_html=True)

# ==========================================
# BOTTOM CHARTS ROW
# ==========================================
st.markdown("<hr style='border-color:#1f2937; margin: 20px 0;'>", unsafe_allow_html=True)
ch1, ch2, ch3 = st.columns([1, 1, 1.2])

with ch1:
    st.markdown("<h4 style='color:white; margin-bottom:0;'>Traffic Distribution</h4>", unsafe_allow_html=True)
    fig_pie = px.pie(values=[53, 32, 15], names=['Normal', 'Neptune', 'Other'], hole=0.75, color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b'])
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0), margin=dict(t=10, b=10, l=10, r=10), height=220)
    fig_pie.add_annotation(text="148,517<br><span style='font-size:10px; color:#94a3b8'>Total Samples</span>", x=0.5, y=0.5, font=dict(size=14, color="white"), showarrow=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with ch2:
    st.markdown("<h4 style='color:white; margin-bottom:0;'>Top Attack Types</h4>", unsafe_allow_html=True)
    fig_bar = px.bar(x=[9574, 866, 660], y=['neptune', 'satan', 'portsweep'], orientation='h', color_discrete_sequence=['#8b5cf6'])
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', xaxis=dict(showgrid=False, title=""), yaxis=dict(showgrid=False, title=""), margin=dict(t=10, b=10, l=10, r=10), height=220)
    st.plotly_chart(fig_bar, use_container_width=True)

with ch3:
    # Table (Single line HTML)
    st.markdown("<h4 style='color:white; margin-bottom:10px;'>Recent Predictions</h4><table style='width: 100%; border-collapse: collapse; font-size: 13px;'><tr><th style='color: #94a3b8; text-align: left; padding: 8px; border-bottom: 1px solid #1f2937;'>#</th><th style='color: #94a3b8; text-align: left; padding: 8px; border-bottom: 1px solid #1f2937;'>Time</th><th style='color: #94a3b8; text-align: left; padding: 8px; border-bottom: 1px solid #1f2937;'>Result</th></tr><tr><td style='color:white; padding:10px 8px; border-bottom:1px solid #1f2937;'>1</td><td style='color:white; padding:10px 8px; border-bottom:1px solid #1f2937;'>14:32:17</td><td style='color:#10b981; font-weight:bold; padding:10px 8px; border-bottom:1px solid #1f2937;'>Normal</td></tr><tr><td style='color:white; padding:10px 8px; border-bottom:1px solid #1f2937;'>2</td><td style='color:white; padding:10px 8px; border-bottom:1px solid #1f2937;'>14:31:03</td><td style='color:#10b981; font-weight:bold; padding:10px 8px; border-bottom:1px solid #1f2937;'>Normal</td></tr><tr><td style='color:white; padding:10px 8px; border-bottom:1px solid #1f2937;'>3</td><td style='color:white; padding:10px 8px; border-bottom:1px solid #1f2937;'>14:29:45</td><td style='color:#ef4444; font-weight:bold; padding:10px 8px; border-bottom:1px solid #1f2937;'>Attack</td></tr></table>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:#64748b; font-size:12px; margin-top:40px; padding:20px 0; border-top:1px solid #1f2937;'>Smart Cybersecurity ML | Built with Streamlit</div>", unsafe_allow_html=True)