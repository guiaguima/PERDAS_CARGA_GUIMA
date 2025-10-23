import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import io # Para manipulação de dados para download
from scipy.optimize import fsolve # Importar fsolve aqui para evitar erro de escopo

# --- Constantes e Conversões ---
g = 9.81  # Aceleração da gravidade (m/s²)
NU_WATER = 1.004e-6 # Viscosidade cinemática da água a 20°C (m²/s)

# Fatores K de Perda Menor para Acessórios Comuns (Valores Típicos)
FITTING_K_FACTORS = {
    "Cotovelo 90° (Raio Curto)": 0.9,
    "Cotovelo 90° (Raio Longo)": 0.4,
    "Cotovelo 45°": 0.2,
    "Válvula de Gaveta (Totalmente Aberta)": 0.15,
    "Válvula Globo (Totalmente Aberta)": 10.0,
    "Válvula de Retenção (Tipo Porta)": 2.0,
    "Válvula de Retenção (Tipo Disco)": 4.0,
    "Entrada de Tubo (Borda Viva)": 0.5,
    "Entrada de Tubo (Arredondada)": 0.1,
    "Saída de Tubo (Perda de Saída)": 1.0,
    "Tê (Linha Principal)": 0.4,
    "Tê (Ramificação 90°)": 1.5,
}

# --- Funções de Cálculo de Curva do Sistema (Colebrook/Swamee-Jain) ---

def swamee_jain_f(Re, e_D):
    """
    Calcula o fator de atrito 'f' de Darcy-Weisbach usando a equação explícita de Swamee-Jain,
    uma excelente aproximação de Colebrook-White.
    """
    if Re < 2000:
        # Fluxo Laminar (Equação de Poiseuille)
        return 64 / Re
    elif Re < 4000:
        # Fluxo de Transição: Simplificação ou interpolação (manter a simplicidade)
        return 0.035 # Valor conservador para transição
    else:
        # Fluxo Turbulento (Swamee-Jain)
        term1 = e_D / 3.7
        term2 = 5.74 / (Re**0.9)
        try:
            return 0.25 / (math.log10(term1 + term2))**2
        except (ValueError, ZeroDivisionError):
            return 0.02 # Valor padrão de segurança

def calculate_head_loss(Q_m3h, segments, nu_kinematic):
    """
    Calcula a perda de carga total (hf) para um dado Q (vazão) e segmentos de tubulação.
    """
    if not segments:
        return 0.0

    # Conversão de Q: m³/h -> m³/s
    Q_m3s = Q_m3h / 3600

    total_h_loss = 0.0

    for seg in segments:
        L = seg['L']
        D = seg['D']
        e = seg['e']
        K_minor = seg['K_minor']

        if D <= 0 or Q_m3s <= 1e-6:
            continue

        # Área da seção transversal
        A = math.pi * (D**2) / 4
        # Velocidade
        V = Q_m3s / A
        # Número de Reynolds
        Re = V * D / nu_kinematic
        # Rugosidade relativa
        e_D = e / D

        # Fator de atrito 'f' (Darcy)
        f = swamee_jain_f(Re, e_D)

        # Carga de Velocidade
        H_vel = V**2 / (2 * g)

        # Perda de Carga
        h_major = f * (L / D) * H_vel
        h_minor = K_minor * H_vel

        total_h_loss += (h_major + h_minor)

    return total_h_loss

# --- Streamlit UI Setup ---

st.set_page_config(layout="wide", page_title="Curvas Bomba vs. Sistema")
st.title("🔀 Curva da Bomba vs. Curva do Sistema")
st.markdown("---")

# --- Inicialização do State ---
if 'pump_points' not in st.session_state:
    st.session_state.pump_points = pd.DataFrame(columns=['Q (m³/h)', 'H (m)'])
if 'system_segments' not in st.session_state:
    st.session_state.system_segments = []

# --- Colunas para Layout ---
col_input, col_plot = st.columns([1, 2])

# Variável para armazenar o DataFrame de segmentos para uso posterior
df_segments_display = pd.DataFrame()

with col_input:
    st.header("1. Parâmetros do Sistema")

    # --- INPUTS GERAIS DO SISTEMA ---
    with st.expander("Gerais (Fluido e Estática)", expanded=True):
        st.subheader("Carga Estática e Fluido")
        # Carga Estática
        static_head = st.number_input(
            "Carga Estática (H estático, m):",
            value=10.0,
            min_value=0.0,
            step=1.0,
            key='static_head'
        )

        # Viscosidade Cinemática
        kinematic_viscosity = st.number_input(
            "Viscosidade Cinemática (ν, m²/s) - Água 20°C: 1e-6",
            value=NU_WATER,
            min_value=1e-8,
            format="%.10e",
            key='kinematic_viscosity'
        )

        # Vazão Máxima para o Gráfico
        max_flow = st.number_input(
            "Vazão Máxima do Gráfico (Q máx, m³/h):",
            value=100.0,
            min_value=1.0,
            step=10.0,
            key='max_flow'
        )

    # --- INPUT DE SEGMENTOS DE PERDA DE CARGA ---
    st.header("2. Segmentos de Perda de Carga")
    with st.expander("Adicionar Segmento (Colebrook/Swamee-Jain)", expanded=True):
        
        # --- INPUTS GEOMÉTRICOS ---
        col_L, col_D = st.columns(2)
        with col_L:
            L_in = st.number_input("Comprimento (L, m):", value=50.0, min_value=0.1, step=5.0)
        with col_D:
            D_in = st.number_input("Diâmetro (D, m):", value=0.15, min_value=0.01, format="%.3f")

        e_in = st.number_input("Rugosidade (ε, m) - Aço: 0.000045", value=0.000045, min_value=1e-7, format="%.7e")
        
        st.subheader("Perdas Menores (K minor)")
        
        # --- NOVO: Cálculo de Perdas Menores ---
        k_method = st.radio(
            "Método de Cálculo K minor:",
            ("Cálculo por Acessórios", "Valor Manual Direto"),
            key='k_method'
        )
        
        K_in = 0.0 # Inicializa
        
        if k_method == "Cálculo por Acessórios":
            
            total_k_calc = 0.0
            st.markdown("Selecione os acessórios incluídos neste segmento:")
            
            # Divide os acessórios em duas colunas para melhor visualização
            fitting_names = list(FITTING_K_FACTORS.keys())
            mid_point = len(fitting_names) // 2
            col_k1, col_k2 = st.columns(2)
            
            # Usa um ID único para o state dos inputs de quantidade
            segment_id_for_state = len(st.session_state.system_segments)
            
            for i, name in enumerate(fitting_names):
                k_value = FITTING_K_FACTORS[name]
                
                # Define a chave única para cada input de quantidade
                key_qty = f"qty_{segment_id_for_state}_{name}"
                
                # Certifica-se de que o estado existe
                if key_qty not in st.session_state:
                    st.session_state[key_qty] = 0

                # Renderiza nas colunas
                target_col = col_k1 if i < mid_point else col_k2
                
                with target_col:
                    qty = st.number_input(
                        f"{name} (K={k_value})", 
                        min_value=0, 
                        value=st.session_state[key_qty], 
                        step=1, 
                        key=key_qty,
                        label_visibility="collapsed"
                    )
                
                # Calcula a contribuição
                total_k_calc += qty * k_value
            
            K_in = total_k_calc
            st.success(f"K minor Total Calculado: **{K_in:.2f}**")
            
        else: # Valor Manual Direto
             K_in = st.number_input("Coef. de Perda Menor (K minor):", value=2.0, min_value=0.0, step=0.1)


        # --- BOTÃO DE ADICIONAR ---
        if st.button("Adicionar Segmento", key="add_segment_button"):
            # Acessórios usados para fins de relatório (se aplicável)
            accessories_used = {}
            if k_method == "Cálculo por Acessórios":
                for name in fitting_names:
                    key_qty = f"qty_{segment_id_for_state}_{name}"
                    qty = st.session_state[key_qty]
                    if qty > 0:
                        accessories_used[name] = qty

            st.session_state.system_segments.append({
                'L': L_in,
                'D': D_in,
                'e': e_in,
                'K_minor': K_in,
                'Acessórios': accessories_used, # Adiciona detalhes para o relatório
                'id': len(st.session_state.system_segments) + 1 
            })
            st.success(f"Segmento {len(st.session_state.system_segments)} adicionado com K minor = {K_in:.2f}!")
            
            # Limpa o estado dos inputs de acessórios após adicionar (para o próximo segmento)
            for name in fitting_names:
                st.session_state[f"qty_{segment_id_for_state}_{name}"] = 0
                
            st.rerun()

    # --- Visualização e Remoção de Segmentos ---
    st.subheader("Segmentos Atuais:")
    if st.session_state.system_segments:
        df_segments = pd.DataFrame(st.session_state.system_segments)
        df_segments_display = df_segments.rename(columns={
            'L': 'L (m)', 'D': 'D (m)', 'e': 'ε (m)', 'K_minor': 'K Menor', 'id': 'ID', 'Acessórios': 'Detalhes K'
        })
        # Remove a coluna 'Acessórios' para o display principal, ou formata de forma concisa
        df_display_clean = df_segments_display.drop(columns=['Detalhes K']).set_index('ID')
        st.dataframe(df_display_clean, use_container_width=True)

        if st.button("Remover Todos os Segmentos"):
            st.session_state.system_segments = []
            st.rerun()
    else:
        st.info("Nenhum segmento de tubulação adicionado.")

    st.markdown("---")

    # --- INPUT DE PONTOS DA BOMBA ---
    st.header("3. Curva da Bomba (Ajuste)")
    with st.expander("Adicionar Ponto de Pressão vs. Vazão", expanded=True):
        col_Q_pump, col_H_pump = st.columns(2)
        with col_Q_pump:
            Q_pump_in = st.number_input("Vazão (Q, m³/h):", value=0.0, step=5.0, key='Q_pump_input')
        with col_H_pump:
            H_pump_in = st.number_input("Altura Manométrica (H, m):", value=40.0, step=5.0, key='H_pump_input')

        if st.button("Adicionar Ponto da Bomba"):
            new_point = pd.DataFrame({'Q (m³/h)': [Q_pump_in], 'H (m)': [H_pump_in]})
            st.session_state.pump_points = pd.concat([st.session_state.pump_points, new_point], ignore_index=True)
            st.session_state.pump_points = st.session_state.pump_points.sort_values('Q (m³/h)').reset_index(drop=True)
            st.rerun()

    st.subheader("Pontos Atuais da Bomba:")
    if not st.session_state.pump_points.empty:
        st.dataframe(st.session_state.pump_points, use_container_width=True)
        if st.button("Limpar Pontos da Bomba"):
            st.session_state.pump_points = pd.DataFrame(columns=['Q (m³/h)', 'H (m)'])
            st.rerun()
    else:
        st.info("Adicione pelo menos 3 pontos para um bom ajuste quadrático.")


# --- CÁLCULOS PRINCIPAIS E PLOTAGEM ---
with col_plot:
    st.header("Curvas Geradas e Ponto de Operação")

    # 1. Vazões para Plotagem
    Q_plot = np.linspace(0, max_flow, 100)
    
    # 2. Cálculo da Curva do Sistema
    H_sys_values = [
        static_head + calculate_head_loss(q, st.session_state.system_segments, kinematic_viscosity)
        for q in Q_plot
    ]
    df_system = pd.DataFrame({'Q (m³/h)': Q_plot, 'H (m)': H_sys_values, 'Curva': 'Sistema'})
    
    # 3. Cálculo da Curva da Bomba
    curve_A = 0
    curve_B = 0
    curve_C = 0
    pump_ready = False
    
    df_pump = pd.DataFrame(columns=['Q (m³/h)', 'H (m)', 'Curva'])
    df_plot = df_system # Inicializa com o sistema

    if len(st.session_state.pump_points) >= 3:
        try:
            # Ajuste Polinomial de 2º Grau (Parábola: H = A + BQ + CQ²)
            Q_data = st.session_state.pump_points['Q (m³/h)'].values
            H_data = st.session_state.pump_points['H (m)'].values
            
            # Coeficientes: [C, B, A] -> numpy.polyfit retorna os coeficientes da ordem mais alta para a mais baixa
            coeffs = np.polyfit(Q_data, H_data, 2)
            curve_C, curve_B, curve_A = coeffs
            pump_ready = True
            
            H_pump_values = curve_A + curve_B * Q_plot + curve_C * (Q_plot**2)
            df_pump = pd.DataFrame({'Q (m³/h)': Q_plot, 'H (m)': H_pump_values, 'Curva': 'Bomba (Ajuste)'})
            
            df_plot = pd.concat([df_system, df_pump], ignore_index=True)

        except np.linalg.LinAlgError:
            st.error("Erro no ajuste: Verifique se os pontos da bomba são válidos e não colineares.")
            
    # --- PONTO DE INTERSEÇÃO (PONTO DE OPERAÇÃO) ---
    
    op_Q, op_H = None, None
    h_loss_max = 0.0 # Inicializa para relatório
    
    if st.session_state.system_segments:
        h_loss_max = calculate_head_loss(max_flow, st.session_state.system_segments, kinematic_viscosity)
        
        if pump_ready:
            
            def difference_function(Q):
                # Q está em m³/h
                H_pump = curve_A + curve_B * Q + curve_C * (Q**2)
                H_system = static_head + calculate_head_loss(Q, st.session_state.system_segments, kinematic_viscosity)
                return H_pump - H_system

            
            # Chute inicial para a vazão de operação
            Q_guess = max_flow / 2
            try:
                # Usa fsolve para encontrar a raiz (interseção)
                Q_operation = fsolve(difference_function, Q_guess)[0]
                
                # Garante que a solução está dentro do intervalo de vazão plotado
                if 0 <= Q_operation <= max_flow:
                    op_Q = Q_operation
                    op_H = curve_A + curve_B * op_Q + curve_C * (op_Q**2)
            except Exception:
                # Caso a curva da bomba e do sistema não se cruzem no intervalo
                pass
    
    # --- PLOTAGEM INTERATIVA (PLOTLY) ---

    # Gráfico Base
    fig = px.line(
        df_plot,
        x='Q (m³/h)',
        y='H (m)',
        color='Curva',
        title="Curva da Bomba vs. Curva do Sistema",
        color_discrete_map={'Sistema': 'blue', 'Bomba (Ajuste)': 'red'}
    )
    
    # Adicionar pontos de entrada da bomba
    if not st.session_state.pump_points.empty:
        fig.add_trace(go.Scatter(
            x=st.session_state.pump_points['Q (m³/h)'],
            y=st.session_state.pump_points['H (m)'],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle-open', line=dict(width=2, color='red')),
            name='Pontos Fornecidos'
        ))

    # Adicionar ponto de operação (se encontrado)
    if op_Q is not None and op_H is not None:
        fig.add_trace(go.Scatter(
            x=[op_Q],
            y=[op_H],
            mode='markers+text',
            marker=dict(size=12, color='green', symbol='star'),
            name='Ponto de Operação',
            text=[f"Q={op_Q:.2f} m³/h<br>H={op_H:.2f} m"],
            textposition="top right"
        ))
        
        st.success(f"**Ponto de Operação Encontrado (Interseção):**")
        st.metric("Vazão de Operação (Q)", f"{op_Q:.2f} m³/h")
        st.metric("Altura Manométrica (H)", f"{op_H:.2f} m")


    # Configurações de layout
    fig.update_layout(
        height=600,
        xaxis_title="Vazão Q (m³/h)",
        yaxis_title="Altura Manométrica H (m)",
        hovermode="x unified",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(range=[0, max_flow * 1.05])
    fig.update_yaxes(rangemode="tozero")
    
    st.plotly_chart(fig, use_container_width=True)

    # Coeficientes da Bomba
    pump_eq_str = "Não disponível (Pontos insuficientes)"
    if pump_ready:
        st.subheader("Equação da Curva da Bomba (H = A + BQ + CQ²)")
        pump_eq_str = f"H = {curve_A:.2f} + ({curve_B:.4f})Q + ({curve_C:.4e})Q²"
        st.code(pump_eq_str)

    # Informações da Curva do Sistema
    st.subheader("Composição da Curva do Sistema")
    st.write(f"Carga Estática (Intercepto): {static_head:.2f} m")
    st.write(f"Perda de Carga Total Calculada (@ {max_flow:.0f} m³/h): {h_loss_max:.2f} m")

    # --- EXPORTAÇÃO DE RELATÓRIO ---
    st.markdown("---")
    st.header("4. Exportação de Relatório e Dados")
    
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
    
    # Função para formatar os detalhes dos acessórios em texto
    def format_accessories_for_report(segments):
        text = ""
        for seg in segments:
            text += f"\n--- Segmento {seg['id']} ---\n"
            text += f"Comprimento (L): {seg['L']} m\n"
            text += f"Diâmetro (D): {seg['D']} m\n"
            text += f"Rugosidade (ε): {seg['e']:.2e} m\n"
            text += f"K minor Final: {seg['K_minor']:.2f}\n"
            
            if seg.get('Acessórios'):
                text += "Acessórios Detalhados:\n"
                for acc, qty in seg['Acessórios'].items():
                    text += f"  - {acc}: {qty} unidades (K={FITTING_K_FACTORS.get(acc, 'N/A')})\n"
            else:
                text += "Acessórios Detalhados: (Valor K Manual)\n"
        return text

    # 1. Gerar Resumo Textual
    report_text = f"""
# Relatório de Análise de Curvas Bomba vs. Sistema

## 1. Parâmetros Gerais do Sistema
- Carga Estática (H estático): {static_head:.2f} m
- Viscosidade Cinemática (ν): {kinematic_viscosity:.2e} m²/s
- Vazão Máxima de Análise: {max_flow:.0f} m³/h

## 2. Ponto de Operação (Interseção)
"""
    if op_Q is not None and op_H is not None:
        report_text += f"""
- Vazão de Operação (Q_op): {op_Q:.2f} m³/h
- Altura Manométrica (H_op): {op_H:.2f} m
"""
    else:
        report_text += "\n- Ponto de Operação: Não encontrado ou fora do limite de vazão."
        
    report_text += f"""
## 3. Curva da Bomba (Ajuste Polinomial de 2º Grau)
- Equação: {pump_eq_str}

## 4. Detalhes da Tubulação e Perdas Menores
{format_accessories_for_report(st.session_state.system_segments)}
"""
    
    # Botão de Download do Resumo
    with col_dl1:
        st.download_button(
            label="📄 Baixar Resumo (TXT)",
            data=report_text,
            file_name="Relatorio_Bomba_Sistema.txt",
            mime="text/plain",
            help="Baixa um resumo textual (Markdown/TXT) dos resultados principais, incluindo os detalhes dos acessórios."
        )

    # 2. Download dos Pontos da Bomba
    if not st.session_state.pump_points.empty:
        with col_dl2:
            csv_pump = st.session_state.pump_points.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Dados Bomba (CSV)",
                data=csv_pump,
                file_name='PumpPoints.csv',
                mime='text/csv',
                help="Baixa os pontos de pressão vs. vazão inseridos para a bomba."
            )
            
    # 3. Download dos Segmentos do Sistema
    if st.session_state.system_segments:
        with col_dl3:
            # Prepara um DF para download, garantindo que a coluna 'Acessórios' seja serializável
            df_export = df_segments_display.copy()
            df_export['Detalhes K'] = df_export['Detalhes K'].apply(lambda x: str(x))
            csv_segments = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="⬇️ Dados Sistema (CSV)",
                data=csv_segments,
                file_name='SystemSegments.csv',
                mime='text/csv',
                help="Baixa os dados detalhados, incluindo o K minor calculado e os acessórios."
            )

    # 4. Download dos Dados da Curva para Plotagem
    with col_dl4:
        csv_plot = df_plot.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Dados das Curvas (CSV)",
            data=csv_plot,
            file_name='PlotData.csv',
            mime='text/csv',
            help="Baixa os 100 pontos gerados para plotar ambas as curvas (Bomba e Sistema)."
        )
    
    st.markdown("---")
    st.info("Dica: Use a barra de ferramentas do gráfico (canto superior direito) para exportar a imagem do gráfico (PNG/SVG).")
