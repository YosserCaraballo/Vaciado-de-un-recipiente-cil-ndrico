# Interfaz
import streamlit as st

# Manejo de datos
import numpy as np
import pandas as pd

# Gráficas
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Regresiones y estadísticos
from scipy.optimize import curve_fit          # regresión exponencial no lineal
from scipy.stats import pearsonr              # coeficiente de correlación

# Matemáticas base
from math import sqrt

# Configuración de la página 
st.set_page_config(
    page_title="Vaciado de recipiente",
    page_icon="💧",
    layout="wide",
    
)

# Título principal 
st.title("Vaciado de un recipiente cilíndrico")

st.divider()

# Sidebar con los datos experimentales 
# Sidebar con carga de datos 
st.sidebar.header("Datos experimentales")

archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo, sep=";", decimal=",")
    t = df["tiempo"].values
    h = df["altura"].values
    st.sidebar.success(f"{len(t)} datos cargados correctamente")
    st.sidebar.dataframe(df)
else:
    st.sidebar.info("Sube un archivo CSV con columnas: tiempo, altura")
    # Datos de prueba mientras no hay experimento
    t = np.array([0,10,20,30,40,50,60,70,80,90,100,
                  110,120,130,140,150,160,170,180,190,200], dtype=float)
    h = np.array([30.0,27.8,25.8,23.9,22.1,20.4,18.8,17.3,15.9,
                  14.6,13.3,12.2,11.1,10.1,9.1,8.2,7.4,6.6,5.9,5.3,4.7])
    st.sidebar.warning("Usando datos de prueba")


st.header("Procesamiento de datos")

# Gráfica de datos crudos
st.subheader("Datos experimentales")
fig_datos = go.Figure()
fig_datos.add_trace(go.Scatter(
    x=t, y=h,
    mode="markers",
    name="Datos experimentales",
    marker=dict(color="cyan", size=8)
))
fig_datos.update_layout(
    title="Altura vs Tiempo — Datos experimentales",
    xaxis_title="Tiempo (s)",
    yaxis_title="Altura (cm)",
    template="plotly_dark"
)
st.plotly_chart(fig_datos, use_container_width=True)

st.subheader("Ajuste de curvas")
tab1, tab2, tab3 = st.tabs(["Regresión lineal", "Regresión polinomial", "Regresión exponencial"])

with tab1:
    st.subheader("Regresión lineal")

    # Cálculo
    coef = np.polyfit(t, h, 1)
    h_lin = np.polyval(coef, t)

    # Estadísticos
    n = len(t)
    h_mean = np.mean(h)
    St = np.sum((h - h_mean)**2)
    Sr = np.sum((h - h_lin)**2)
    r2 = 1 - Sr/St

    # r con signo correcto usando pearsonr
    r, _ = pearsonr(t, h)

    Sy = np.sqrt(St/(n-1))
    Syx = np.sqrt(Sr/(n-2))

    # Gráfica
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=h, mode="markers", name="Datos", marker=dict(color="cyan", size=8)))
    fig1.add_trace(go.Scatter(x=t, y=h_lin, mode="lines", name="Regresión lineal", line=dict(color="red")))
    fig1.update_layout(
        title=f"Regresión lineal: h = {coef[0]:.4f}t + {coef[1]:.4f}",
        xaxis_title="Tiempo (s)", yaxis_title="Altura (cm)", template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Estadísticos en columnas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.4f}")
    col2.metric("r (correlación)", f"{r:.4f}")
    col3.metric("Sy (desv. est. total)", f"{Sy:.4f}")
    col4.metric("Syx (error estándar)", f"{Syx:.4f}")
    st.info(f"Ecuación: h(t) = {coef[0]:.4f} · t + {coef[1]:.4f}")

with tab2:
    st.subheader("Regresión polinomial grado 2")

    # Cálculo
    coef2 = np.polyfit(t, h, 2)
    h_pol = np.polyval(coef2, t)

    Sr2 = np.sum((h - h_pol)**2)
    r2_pol = 1 - Sr2/St
    r_pol, _ = pearsonr(h, h_pol)   # r entre valores reales y ajustados
    Syx2 = np.sqrt(Sr2/(n-3))

    

    # Gráfica
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=h, mode="markers", name="Datos", marker=dict(color="cyan", size=8)))
    fig2.add_trace(go.Scatter(x=t, y=h_pol, mode="lines", name="Regresión polinomial", line=dict(color="orange")))
    fig2.update_layout(
        title=f"Regresión polinomial: h = {coef2[0]:.6f}t² + {coef2[1]:.4f}t + {coef2[2]:.4f}",
        xaxis_title="Tiempo (s)", yaxis_title="Altura (cm)", template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2_pol:.4f}")
    col2.metric("r (correlación)", f"{r_pol:.4f}")
    col3.metric("Sy (desv. est. total)", f"{Sy:.4f}")
    col4.metric("Syx (error estándar)", f"{Syx2:.4f}")

    
with tab3:
    st.subheader("Regresión exponencial no lineal")

    # Modelo exponencial
    def modelo_exp(t, a, b):
        return a * np.exp(b * t)

    popt, _ = curve_fit(modelo_exp, t, h, p0=[30, -0.01], maxfev=5000)
    h_exp = modelo_exp(t, *popt)

    Sr3 = np.sum((h - h_exp)**2)
    r2_exp = 1 - Sr3/St
    r_exp, _ = pearsonr(h, h_exp)   # r entre valores reales y ajustados
    Syx3 = np.sqrt(Sr3/(n-2))

    

    # Gráfica
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t, y=h, mode="markers", name="Datos", marker=dict(color="cyan", size=8)))
    fig3.add_trace(go.Scatter(x=t, y=h_exp, mode="lines", name="Regresión exponencial", line=dict(color="lime")))
    fig3.update_layout(
        title=f"Regresión exponencial: h = {popt[0]:.4f} · e^({popt[1]:.6f}·t)",
        xaxis_title="Tiempo (s)", yaxis_title="Altura (cm)", template="plotly_dark"
    )
    st.plotly_chart(fig3, use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2_exp:.4f}")
    col2.metric("r (correlación)", f"{r_exp:.4f}")
    col3.metric("Sy (desv. est. total)", f"{Sy:.4f}")
    col4.metric("Syx (error estándar)", f"{Syx3:.4f}")

    

st.divider()

# Seccion 4: Interpolacion de Lagrange 
# Seccion 4: Interpolacion de Lagrange
st.header("Interpolación de Lagrange")

# Función de Lagrange
def lagrange_interp(x_data, y_data, x, grado):
    # Seleccionar puntos equiespaciados según el grado
    indices = np.linspace(0, len(x_data)-1, grado+1, dtype=int)
    xp = x_data[indices]
    yp = y_data[indices]
    
    result = 0.0
    for i in range(len(xp)):
        term = yp[i]
        for j in range(len(xp)):
            if i != j:
                term *= (x - xp[j]) / (xp[i] - xp[j])
        result += term
    return result

# Calcular interpolaciones para todos los grados
t_suave = np.linspace(t[0], t[-1], 300)
colores = ["yellow", "orange", "lime", "violet"]

def calcular_grado(grado):
    h_interp = np.array([lagrange_interp(t, h, ti, grado) for ti in t])
    h_curva  = np.array([lagrange_interp(t, h, ti, grado) for ti in t_suave])
    
    # Error relativo porcentual
    with np.errstate(divide='ignore', invalid='ignore'):
        et = np.where(h != 0, np.abs((h - h_interp) / h) * 100, 0)
    
    et_prom = np.mean(et)
    et_std  = np.std(et)
    return h_interp, h_curva, et, et_prom, et_std

g1 = calcular_grado(1)
g2 = calcular_grado(2)
g3 = calcular_grado(3)
g4 = calcular_grado(4)
grados = [g1, g2, g3, g4]

tab4, tab5, tab6, tab7 = st.tabs(["Lineal", "Cuadrática", "Cúbica", "Cuártica"])
tabs_interp = [tab4, tab5, tab6, tab7]
nombres = ["Lineal", "Cuadrática", "Cúbica", "Cuártica"]

formulas = [
    r"P_1(x) = \frac{x - x_1}{x_0 - x_1} f(x_0) + \frac{x - x_0}{x_1 - x_0} f(x_1)",
    r"P_2(x) = \sum_{i=0}^{2} f(x_i) \prod_{j=0, j \neq i}^{2} \frac{x - x_j}{x_i - x_j}",
    r"P_3(x) = \sum_{i=0}^{3} f(x_i) \prod_{j=0, j \neq i}^{3} \frac{x - x_j}{x_i - x_j}",
    r"P_4(x) = \sum_{i=0}^{4} f(x_i) \prod_{j=0, j \neq i}^{4} \frac{x - x_j}{x_i - x_j}",
]

for i, (tab, (h_interp, h_curva, et, et_prom, et_std)) in enumerate(zip(tabs_interp, grados)):
    with tab:
        st.subheader(f"Interpolación {nombres[i]}")

        with st.expander("Ver fórmulas"):
            st.markdown("**Polinomio de Lagrange:**")
            st.latex(formulas[i])
            st.markdown("**Error relativo porcentual:**")
            st.latex(r"\varepsilon_t \% = \left| \frac{f(x_i) - P_n(x_i)}{f(x_i)} \right| \times 100")
            st.markdown("**Error promedio:**")
            st.latex(r"\varepsilon_t \%_{promedio} = \frac{1}{n} \sum_{i=1}^{n} \varepsilon_t \%_i")

        fig_interp = go.Figure()
        fig_interp.add_trace(go.Scatter(
            x=t, y=h, mode="markers", name="Datos experimentales",
            marker=dict(color="cyan", size=8)
        ))
        fig_interp.add_trace(go.Scatter(
            x=t_suave, y=h_curva, mode="lines",
            name="Interpolación",
            line=dict(color=colores[i], width=2)
        ))
        fig_interp.update_layout(
            title=f"Interpolación {nombres[i]} de Lagrange",
            xaxis_title="Tiempo (s)", yaxis_title="Altura (cm)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_interp, use_container_width=True)

        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(
            x=t, y=et, mode="lines+markers",
            name="εt%",
            line=dict(color=colores[i], width=2),
            marker=dict(size=6)
        ))
        fig_error.update_layout(
            title=f"Error relativo porcentual εt% — Interpolación {nombres[i]}",
            xaxis_title="Tiempo (s)", yaxis_title="εt%",
            template="plotly_dark"
        )
        st.plotly_chart(fig_error, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("εt% promedio", f"{et_prom:.4f}%")
        col2.metric("Desv. estándar εt%", f"{et_std:.4f}%")

st.divider()

# Comparacion de todos los grados
st.subheader("Comparación de interpolaciones")

fig_comp = go.Figure()
fig_comp.add_trace(go.Scatter(
    x=t, y=h, mode="markers", name="Datos experimentales",
    marker=dict(color="cyan", size=8)
))
for i, (_, h_curva, _, _, _) in enumerate(grados):
    fig_comp.add_trace(go.Scatter(
        x=t_suave, y=h_curva, mode="lines",
        name=nombres[i], line=dict(color=colores[i], width=2)
    ))
fig_comp.update_layout(
    title="Comparación de interpolaciones — Lineal, Cuadrática, Cúbica y Cuártica",
    xaxis_title="Tiempo (s)", yaxis_title="Altura (cm)",
    template="plotly_dark"
)
st.plotly_chart(fig_comp, use_container_width=True)

fig_err_comp = go.Figure()
for i, (_, _, et, _, _) in enumerate(grados):
    fig_err_comp.add_trace(go.Scatter(
        x=t, y=et, mode="lines+markers",
        name=f"εt% {nombres[i]}",
        line=dict(color=colores[i], width=2), marker=dict(size=5)
    ))
fig_err_comp.update_layout(
    title="Comparación εt% vs t en todos los grados",
    xaxis_title="Tiempo (s)", yaxis_title="εt%",
    template="plotly_dark"
)
st.plotly_chart(fig_err_comp, use_container_width=True)

# Tabla comparativa
st.subheader("Tabla comparativa de estadísticos")
tabla = pd.DataFrame({
    "Interpolación": nombres,
    "εt% promedio": [f"{g[3]:.4f}%" for g in grados],
    "Desv. estándar εt%": [f"{g[4]:.4f}%" for g in grados]
})
st.dataframe(tabla, use_container_width=True)

# Ocultar menú y footer de Streamlit y asignnando estilos
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
