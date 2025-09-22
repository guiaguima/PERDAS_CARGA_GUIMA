import streamlit as st
import numpy as np
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------
# Funções hidráulicas
# ----------------------
g = 9.80665  # gravidade [m/s²]

def reynolds_number(rho, v, D, mu):
    return rho * v * D / mu

def swamee_jain(Re, eps, D):
    if Re <= 0:
        return np.nan
    return 0.25 / (np.log10(eps / (3.7 * D) + 5.74 / (Re**0.9))**2)

def colebrook_friction(Re, eps, D, tol=1e-6, maxiter=200):
    if Re <= 2300:
        return 64.0 / Re
    f = swamee_jain(Re, eps, D)
    if not np.isfinite(f) or f <= 0:
        f = 0.02
    for _ in range(maxiter):
        lhs = 1.0 / np.sqrt(f)
        rhs = -2.0 * np.log10(eps / (3.7 * D) + 2.51 / (Re * np.sqrt(f)))
        new_f = 1.0 / (rhs**2)
        if abs(lhs - rhs) < tol:
            return f
        f = 0.5 * f + 0.5 * new_f
    return swamee_jain(Re, eps, D)

def darcy_weisbach_headloss(f, L, D, v):
    return f * (L / D) * (v**2) / (2.0 * g)

def pressure_drop_from_head(rho, head):
    return rho * g * head

def calcular_perda(rho, mu, D, L, Q, eps, method="Swamee-Jain", K_total=0):
    A = np.pi * D**2 / 4
    v = Q / A
    Re = reynolds_number(rho, v, D, mu)
    f = colebrook_friction(Re, eps, D) if method == "Colebrook" else swamee_jain(Re, eps, D)
    h_major = darcy_weisbach_headloss(f, L, D, v)
    h_minor = K_total * (v**2) / (2*g)
    h_total = h_major + h_minor
    delta_p = pressure_drop_from_head(rho, h_total)
    return {"Re": Re, "f": f, "v": v, "h_total": h_total, "delta_p": delta_p}

# ----------------------
# Função PDF segura
# ----------------------
def gerar_pdf(trechos):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Relatório de Perda de Carga", styles['Title']))
    story.append(Spacer(1, 12))

    total_dp = 0.0
    total_h = 0.0

    for i, t in enumerate(trechos, 1):
        story.append(Paragraph(f"<b>Trecho {i}</b>", styles['Heading2']))
        story.append(Paragraph(f"Diâmetro: {t.get('D', 'N/A')} m", styles['Normal']))
        story.append(Paragraph(f"Comprimento: {t.get('L', 'N/A')} m", styles['Normal']))
        story.append(Paragraph(f"Vazão: {t.get('Q', 'N/A')} m³/s", styles['Normal']))
        story.append(Paragraph(f"Rugosidade: {t.get('eps', 'N/A')} m", styles['Normal']))
        story.append(Paragraph(f"Método: {t.get('method', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"K total: {t.get('K_total', 'N/A')}", styles['Normal']))

        res = t.get("resultado")
        if res:
            story.append(Paragraph(f"Reynolds: {res.get('Re',0):.2f}", styles['Normal']))
            story.append(Paragraph(f"Fator de atrito: {res.get('f',0):.4f}", styles['Normal']))
            story.append(Paragraph(f"Velocidade: {res.get('v',0):.3f} m/s", styles['Normal']))
            story.append(Paragraph(f"Perda de carga: {res.get('h_total',0):.3f} m", styles['Normal']))
            story.append(Paragraph(f"Δp: {res.get('delta_p',0):.2f} Pa", styles['Normal']))
            total_dp += res.get('delta_p',0)
            total_h += res.get('h_total',0)

        story.append(Spacer(1, 12))

    # Adiciona somatório total no final do relatório
    story.append(Spacer(1, 12))
    story.append(Paragraph("📌 Totais de todos os trechos", styles['Heading2']))
    story.append(Paragraph(f"Δp total: {total_dp:.2f} Pa", styles['Normal']))
    story.append(Paragraph(f"h_total total: {total_h:.3f} m", styles['Normal']))

    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value
# ----------------------
# Streamlit app
# ----------------------
st.set_page_config(page_title="Perda de Carga em Tubulações", layout="centered")
st.title("💧 Cálculo de Perda de Carga em Tubulações Industriais")

# Inicializa a sessão ou reseta para evitar dados antigos
if "trechos" not in st.session_state:
    st.session_state.trechos = []

# Botão de reset
if st.button("🗑️ Limpar todos os trechos"):
    st.session_state.trechos = []
    st.success("Todos os trechos foram apagados.")
    st.experimental_rerun()

# ----------------------
# Formulário
# ----------------------
with st.form("form_trecho"):
    st.subheader("Adicionar novo trecho")
    rho = st.number_input("Densidade [kg/m³]", value=1000.0)
    mu = st.number_input("Viscosidade dinâmica [Pa.s]", value=0.001002)
    D = st.number_input("Diâmetro interno [m]", value=0.1, format="%.4f")
    L = st.number_input("Comprimento [m]", value=10.0)
    Q = st.number_input("Vazão [m³/s]", value=0.01, format="%.5f")
    st.image("https://github.com/guiaguima/PERDAS_CARGA_GUIMA/blob/main/RUGOSIDADE.png?raw=true")
    eps = st.number_input("Rugosidade absoluta [m]", value=1e-4, format="%.1e")
    method = st.selectbox("Método de atrito", ["Swamee-Jain", "Colebrook"])
    st.image("https://github.com/guiaguima/PERDAS_CARGA_GUIMA/blob/main/PERDAS_LOCALIZADAS.png?raw=true")
    K_total = st.number_input("Somatório de coeficientes K (perdas locais)", value=0.0)

    submitted = st.form_submit_button("➕ Adicionar trecho")
    if submitted:
        resultado = calcular_perda(rho, mu, D, L, Q, eps, method, K_total)
        st.session_state.trechos.append({
            "rho": rho, "mu": mu, "D": D, "L": L, "Q": Q,
            "eps": eps, "method": method, "K_total": K_total,
            "resultado": resultado
        })
        st.success("Trecho adicionado com sucesso!")

# ----------------------
# Mostrar trechos, somatório e botão PDF
# ----------------------
if st.session_state.trechos:
    st.subheader("📊 Trechos cadastrados")
    total_dp = 0.0
    total_h = 0.0
    for i, t in enumerate(st.session_state.trechos, 1):
        res = t.get("resultado")
        if res:
            st.write(f"**Trecho {i}:** Δp = {res.get('delta_p',0):.2f} Pa, "
                     f"h = {res.get('h_total',0):.3f} m, "
                     f"v = {res.get('v',0):.3f} m/s")
            total_dp += res.get('delta_p',0)
            total_h += res.get('h_total',0)
        else:
            st.warning(f"Trecho {i} não possui cálculos salvos.")

    # Somatório total
    st.markdown(f"**Δp total (todos os trechos):** {total_dp:.2f} Pa")
    st.markdown(f"**h_total total (todos os trechos):** {total_h:.3f} m")

    # Download do relatório PDF
    pdf_bytes = gerar_pdf(st.session_state.trechos)
    st.download_button(
        "📥 Baixar relatório PDF",
        data=pdf_bytes,
        file_name="relatorio_perda_carga.pdf",
        mime="application/pdf"
    )
else:
    st.info("Adicione pelo menos um trecho para gerar relatório.")
