"""
Dashboard de Política Monetaria Mexicana
Series de Tiempo con Python
Autor: Chris
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from prophet import Prophet
import pmdarima as pm
import warnings
warnings.filterwarnings("ignore")

# ── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Política Monetaria MX",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS personalizados ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Fondo general */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Título principal */
    .titulo-principal {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #e6edf3;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }

    .subtitulo {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Tarjetas de métricas */
    .metrica-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metrica-card:hover { border-color: #58a6ff; }
    .metrica-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metrica-valor {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #e6edf3;
        line-height: 1;
    }
    .metrica-delta-up   { font-size: 0.8rem; color: #f85149; margin-top: 0.3rem; }
    .metrica-delta-down { font-size: 0.8rem; color: #3fb950; margin-top: 0.3rem; }
    .metrica-delta-neutral { font-size: 0.8rem; color: #8b949e; margin-top: 0.3rem; }

    /* Secciones */
    .seccion-titulo {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #e6edf3;
        border-left: 3px solid #58a6ff;
        padding-left: 0.8rem;
        margin: 2rem 0 1rem 0;
    }

    /* Caja de hallazgo */
    .hallazgo {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #3fb950;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: #c9d1d9;
    }
    .hallazgo strong { color: #e6edf3; }

    /* Botón */
    .stButton > button {
        background: #1f6feb;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        width: 100%;
        padding: 0.6rem;
        font-size: 0.9rem;
    }
    .stButton > button:hover { background: #388bfd; }

    /* Ocultar elementos de Streamlit */
    #MainMenu, footer, header { visibility: hidden; }

    #/* Oculta el botón de colapsar la sidebar — múltiples versiones de Streamlit */
    #[data-testid="collapsedControl"] { display: none !important; }
    #button[kind="header"] { display: none !important; }
    #[data-testid="stSidebarCollapseButton"] { display: none !important; }
    #section[data-testid="stSidebar"] button { display: none !important; }

</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SIDEBAR — Configuración
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏦 Configuración")
    st.markdown("---")

    token = st.text_input(
        "Token SIE Banxico",
        type="password",
        placeholder="Tu token de 64 caracteres",
        help="Obtén tu token en https://www.banxico.org.mx/SieAPIRest/service/v1/"
    )

    st.markdown("**Periodo de análisis**")
    col_a, col_b = st.columns(2)
    with col_a:
        anio_inicio = st.selectbox("Desde", list(range(2008, 2025)), index=0)
    with col_b:
        anio_fin = st.selectbox("Hasta", list(range(2020, 2027)), index=6)

    n_proyeccion = st.slider("Meses a proyectar", 6, 24, 12)

    st.markdown("---")
    correr = st.button("▶  Correr análisis", type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#8b949e; line-height:1.6'>
    <strong style='color:#c9d1d9'>Datos:</strong> API SIE Banxico<br>
    <strong style='color:#c9d1d9'>Series:</strong> SP30578 · SP74662 · SF61745 · SF43718<br>
    <strong style='color:#c9d1d9'>Modelos:</strong> ARIMA · Prophet · VAR(3)<br>
    <strong style='color:#c9d1d9'>Versión:</strong> 1.0
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# ENCABEZADO
# ════════════════════════════════════════════════════════════════
st.markdown('<p class="titulo-principal">Política Monetaria Mexicana</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Análisis de series de tiempo con datos del Banco de México</p>', unsafe_allow_html=True)

# Estado inicial — sin datos
if not correr or not token:
    st.info("👈 Ingresa tu token en el panel izquierdo y presiona **Correr análisis** para comenzar.")
    st.markdown('<p class="seccion-titulo">¿Qué analiza este dashboard?</p>', unsafe_allow_html=True)

    for hallazgo in [
        ("<strong>Causalidad:</strong> La inflación Granger-causa a la tasa Banxico (lags 1-2 meses). El Banxico <em>reacciona</em>, no anticipa."),
        ("<strong>Modelos:</strong> Comparamos ARIMA, Prophet y VAR. El VAR(3) es el más adecuado por capturar interdependencias entre variables."),
        ("<strong>Shocks exógenos:</strong> Crisis 2008, pandemia 2020 y ciclo inflacionario 2022 no son predecibles por ningún modelo estadístico."),
        ("<strong>Proyección:</strong> El VAR anticipa inflación cerca de 4.25% y tasa bajando a ~6% para finales de 2026."),
    ]:
        st.markdown(f'<div class="hallazgo">{hallazgo}</div>', unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)  # Cachea los datos 1 hora para no llamar la API en cada recarga
def descargar_datos(token, fecha_inicio, fecha_fin):
    """
    Descarga y limpia los datos de la API del Banxico.
    Retorna un DataFrame mensual con las 4 series.
    """
    from sie_banxico import SIEBanxico

    NOMBRES = {
        'SP30578': 'Inflación general anual',
        'SP74662': 'Inflación subyacente anual',
        'SF61745': 'Tasa objetivo Banxico',
        'SF43718': 'Tipo de cambio USD/MXN (FIX)'
    }

    api = SIEBanxico(token=token, id_series=list(NOMBRES.keys()))
    respuesta = api.get_timeseries_range(init_date=fecha_inicio, end_date=fecha_fin)
    series_raw = respuesta['bmx']['series']

    dfs = []
    for serie in series_raw:
        nombre   = NOMBRES[serie['idSerie']]
        df_serie = pd.DataFrame(serie['datos'])
        df_serie['fecha'] = pd.to_datetime(df_serie['fecha'], format='%d/%m/%Y')
        df_serie['dato']  = pd.to_numeric(df_serie['dato'], errors='coerce')
        df_serie = df_serie.rename(columns={'dato': nombre}).set_index('fecha')
        dfs.append(df_serie)

    df = dfs[0].join(dfs[1:], how='outer').sort_index()
    return df.resample('ME').mean().dropna(how='all')


with st.spinner("Conectando con la API del Banxico y descargando datos..."):
    try:
        fecha_inicio = f"{anio_inicio}-01-01"
        fecha_fin    = f"{anio_fin}-12-31"
        df = descargar_datos(token, fecha_inicio, fecha_fin)
        st.success(f"✅ {len(df)} meses de datos descargados ({fecha_inicio[:4]}–{fecha_fin[:4]})")
    except Exception as e:
        st.error(f"❌ Error al conectar con la API del Banxico: {e}")
        st.stop()


# ════════════════════════════════════════════════════════════════
# MÉTRICAS — Indicadores más recientes
# ════════════════════════════════════════════════════════════════
st.markdown('<p class="seccion-titulo">Indicadores más recientes</p>', unsafe_allow_html=True)

ultimo     = df.dropna().iloc[-1]
penultimo  = df.dropna().iloc[-2]

def delta_html(actual, anterior, invertir=False):
    """Genera el HTML del cambio mes a mes."""
    diff = actual - anterior
    if invertir:
        clase = "metrica-delta-down" if diff > 0 else "metrica-delta-up"
    else:
        clase = "metrica-delta-up" if diff > 0 else "metrica-delta-down"
    signo = "▲" if diff > 0 else "▼"
    return f'<div class="{clase}">{signo} {abs(diff):.2f} vs mes anterior</div>'

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metrica-card">
        <div class="metrica-label">Inflación general</div>
        <div class="metrica-valor">{ultimo['Inflación general anual']:.2f}%</div>
        {delta_html(ultimo['Inflación general anual'], penultimo['Inflación general anual'])}
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metrica-card">
        <div class="metrica-label">Inflación subyacente</div>
        <div class="metrica-valor">{ultimo['Inflación subyacente anual']:.2f}%</div>
        {delta_html(ultimo['Inflación subyacente anual'], penultimo['Inflación subyacente anual'])}
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metrica-card">
        <div class="metrica-label">Tasa Banxico</div>
        <div class="metrica-valor">{ultimo['Tasa objetivo Banxico']:.2f}%</div>
        {delta_html(ultimo['Tasa objetivo Banxico'], penultimo['Tasa objetivo Banxico'], invertir=True)}
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metrica-card">
        <div class="metrica-label">USD / MXN</div>
        <div class="metrica-valor">${ultimo['Tipo de cambio USD/MXN (FIX)']:.2f}</div>
        {delta_html(ultimo['Tipo de cambio USD/MXN (FIX)'], penultimo['Tipo de cambio USD/MXN (FIX)'])}
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Visualizaciones históricas
# ════════════════════════════════════════════════════════════════
st.markdown('<p class="seccion-titulo">Análisis histórico</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Inflación", "⚖️ Inflación vs Tasa", "🌐 Panel completo"])

COLORES = {
    'inflacion':  '#f85149',
    'subyacente': '#ffa657',
    'tasa':       '#58a6ff',
    'usdmxn':     '#3fb950',
}
TEMPLATE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c9d1d9', family='DM Sans'),
    xaxis=dict(gridcolor='#21262d', showline=False),
    yaxis=dict(gridcolor='#21262d', showline=False),
    hovermode='x unified',
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#30363d', borderwidth=1)
)

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Inflación general anual'],
        name='General', line=dict(color=COLORES['inflacion'], width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Inflación subyacente anual'],
        name='Subyacente', line=dict(color=COLORES['subyacente'], width=2, dash='dot')))
    fig.add_hline(y=3, line_dash='dash', line_color='#3fb950',
        annotation_text='Meta Banxico 3%', annotation_font_color='#3fb950')
    fig.add_hrect(y0=2, y1=4, fillcolor='#3fb950', opacity=0.05)
    fig.update_layout(**TEMPLATE,
        title=dict(text='Inflación en México — Variación anual del INPC', font=dict(size=15)),
        yaxis_title='%')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = make_subplots(specs=[[{'secondary_y': True}]])
    fig2.add_trace(go.Scatter(x=df.index, y=df['Inflación general anual'],
        name='Inflación general', line=dict(color=COLORES['inflacion'], width=2.5)),
        secondary_y=False)
    fig2.add_trace(go.Scatter(x=df.index, y=df['Tasa objetivo Banxico'],
        name='Tasa Banxico', line=dict(color=COLORES['tasa'], width=2.5)),
        secondary_y=True)
    fig2.add_hline(y=3, line_dash='dash', line_color='#3fb950',
        annotation_text='Meta 3%', annotation_font_color='#3fb950')
    fig2.update_layout(**TEMPLATE,
        title=dict(text='Inflación vs Tasa de Política Monetaria', font=dict(size=15)))
    fig2.update_yaxes(title_text='Inflación (%)', secondary_y=False,
        gridcolor='#21262d', showline=False)
    fig2.update_yaxes(title_text='Tasa (%)', secondary_y=True,
        gridcolor='#21262d', showline=False)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = make_subplots(rows=3, cols=1,
        subplot_titles=('Inflación General (%)', 'Tasa Objetivo Banxico (%)', 'Tipo de Cambio USD/MXN'),
        shared_xaxes=True, vertical_spacing=0.08)
    datos_panel = [
        ('Inflación general anual',      COLORES['inflacion'], 1),
        ('Tasa objetivo Banxico',         COLORES['tasa'],      2),
        ('Tipo de cambio USD/MXN (FIX)', COLORES['usdmxn'],   3),
    ]
    for col, color, row in datos_panel:
        fig3.add_trace(go.Scatter(x=df.index, y=df[col],
            line=dict(color=color, width=2), showlegend=False), row=row, col=1)
    fig3.update_layout(**TEMPLATE, height=700,
        title=dict(text='Panel de Indicadores Macroeconómicos', font=dict(size=15)))
    for i in range(1, 4):
        fig3.update_xaxes(gridcolor='#21262d', row=i, col=1)
        fig3.update_yaxes(gridcolor='#21262d', row=i, col=1)
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Causalidad de Granger
# ════════════════════════════════════════════════════════════════
st.markdown('<p class="seccion-titulo">Causalidad de Granger</p>', unsafe_allow_html=True)
st.markdown("""
<div style='font-size:0.88rem; color:#8b949e; margin-bottom:1rem'>
"X Granger-causa a Y" significa que el historial de X mejora la predicción de Y.
No es causalidad filosófica — es causalidad <em>predictiva</em>.
</div>
""", unsafe_allow_html=True)

df_granger = df[['Inflación general anual', 'Tasa objetivo Banxico']].dropna()

col_g1, col_g2 = st.columns(2)

def tabla_granger(df_input, titulo):
    res = grangercausalitytests(df_input, maxlag=6, verbose=False)
    filas = []
    for lag, r in res.items():
        pval = r[0]['ssr_ftest'][1]
        filas.append({
            'Lag': lag,
            'p-valor': round(pval, 4),
            'Significativo (p<0.05)': '✅ Sí' if pval < 0.05 else '❌ No'
        })
    return pd.DataFrame(filas)

with col_g1:
    st.markdown("**¿Inflación → Tasa?**")
    df_g1 = tabla_granger(df_granger[['Tasa objetivo Banxico', 'Inflación general anual']],
                           'Inflación → Tasa')
    st.dataframe(df_g1, hide_index=True, use_container_width=True)

with col_g2:
    st.markdown("**¿Tasa → Inflación?**")
    df_g2 = tabla_granger(df_granger[['Inflación general anual', 'Tasa objetivo Banxico']],
                           'Tasa → Inflación')
    st.dataframe(df_g2, hide_index=True, use_container_width=True)

st.markdown("""
<div class="hallazgo">
<strong>Hallazgo:</strong> La inflación Granger-causa a la tasa (lags 1-2, p&lt;0.05), pero la tasa no causa a la inflación.
El Banco de México opera una <strong>regla de reacción</strong> — sigue a la inflación pasada con 1-2 meses de rezago.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Proyección VAR
# ════════════════════════════════════════════════════════════════
st.markdown('<p class="seccion-titulo">Proyección con modelo VAR</p>', unsafe_allow_html=True)
st.markdown("""
<div style='font-size:0.88rem; color:#8b949e; margin-bottom:1rem'>
El VAR (Vector AutoRegresivo) es multivariado: cada variable se explica con el pasado de todas las demás simultáneamente.
Es el modelo más adecuado para este análisis porque captura las interdependencias reales de la economía.
</div>
""", unsafe_allow_html=True)

with st.spinner("Entrenando modelo VAR(3)..."):
    df_var = df[['Inflación general anual', 'Tasa objetivo Banxico',
                  'Tipo de cambio USD/MXN (FIX)']].dropna()
    var_model  = VAR(df_var).fit(3)
    proy_var   = var_model.forecast(df_var.values[-3:], steps=n_proyeccion)
    fechas_fut = pd.date_range(
        start=df_var.index[-1] + pd.DateOffset(months=1),
        periods=n_proyeccion, freq='ME'
    )
    df_proy = pd.DataFrame(proy_var, index=fechas_fut, columns=df_var.columns)

    mse           = var_model.mse(steps=n_proyeccion)
    std_inflacion = np.sqrt([mse[i][0,0] for i in range(n_proyeccion)])
    std_tasa      = np.sqrt([mse[i][1,1] for i in range(n_proyeccion)])

# Gráfica de proyección
fig_var = make_subplots(rows=2, cols=1,
    subplot_titles=('Inflación General — Proyección VAR', 'Tasa Banxico — Proyección VAR'),
    shared_xaxes=True, vertical_spacing=0.1)

for row, col, std, color in [
    (1, 'Inflación general anual', std_inflacion, COLORES['inflacion']),
    (2, 'Tasa objetivo Banxico',   std_tasa,      COLORES['tasa'])
]:
    # Histórico
    fig_var.add_trace(go.Scatter(x=df_var.index, y=df_var[col],
        name=f'{col} (real)', line=dict(color=color, width=2)), row=row, col=1)
    # Proyección
    fig_var.add_trace(go.Scatter(x=fechas_fut, y=df_proy[col],
        name=f'{col} (VAR)', line=dict(color=color, width=2.5, dash='dash')), row=row, col=1)
    # Banda de incertidumbre
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    fig_var.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=list(df_proy[col] + 1.96*std) + list((df_proy[col] - 1.96*std)[::-1]),
        fill='toself', fillcolor=f'rgba({r},{g},{b},0.12)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=row, col=1)

fig_var.add_hline(y=3, line_dash='dash', line_color='#3fb950',
    annotation_text='Meta 3%', annotation_font_color='#3fb950', row=1, col=1)

fig_var.update_layout(**TEMPLATE, height=600,
    title=dict(text=f'Proyección VAR(3) — {n_proyeccion} meses', font=dict(size=15)))
for i in range(1, 3):
    fig_var.update_xaxes(gridcolor='#21262d', row=i, col=1)
    fig_var.update_yaxes(gridcolor='#21262d', row=i, col=1)

# Ocultamos cualquier trace sin nombre que Plotly agregue automáticamente
for trace in fig_var.data:
    if trace.name is None or trace.name.startswith('trace'):
        trace.update(showlegend=False)

st.plotly_chart(fig_var, use_container_width=True)

# Tabla de proyecciones
st.markdown("**Tabla de proyecciones**")
df_tabla = df_proy[['Inflación general anual', 'Tasa objetivo Banxico',
                      'Tipo de cambio USD/MXN (FIX)']].round(2).copy()
df_tabla.index = df_tabla.index.strftime('%B %Y')
st.dataframe(df_tabla, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PIE DE PÁGINA
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.78rem; color:#484f58; padding:0.5rem'>
    Datos: API SIE Banco de México · Modelo: VAR(3) · 
    Series: SP30578 · SP74662 · SF61745 · SF43718
</div>
""", unsafe_allow_html=True)
