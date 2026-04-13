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
    t = df["Tiempo"].values  # convierte en columna en un array de  numpy para trabajar con ella
    h = df["Altura"].values
    st.sidebar.success(f"{len(t)} datos cargados correctamente")
    st.sidebar.dataframe(df) # muestra los datos de la tabla de exel 
else:
    st.sidebar.info("Sube un archivo CSV con columnas: tiempo, altura")
    # Datos de prueba mientras no hay experimento
    # creamos un array de numpy y con dtype forzamos que sean valores de ese tipo en este caso flotante
    t = np.array([0,10,20,30,40,50,60,70,80,90,100,
                  110,120,130,140,150,160,170,180,190,200], dtype=float)
    h = np.array([30.0,27.8,25.8,23.9,22.1,20.4,18.8,17.3,15.9,
                  14.6,13.3,12.2,11.1,10.1,9.1,8.2,7.4,6.6,5.9,5.3,4.7])
    st.sidebar.warning("Usando datos de prueba") # aviso


st.header("Procesamiento de datos")

# Gráfica de datos 
st.subheader("Datos experimentales")
fig_datos = go.Figure() # crea figura vacia una hoja en blanco
# trace es una capa de datos y se puede repsentar de muchas maneras
# en este caso se usa para aagregar puntos
fig_datos.add_trace(go.Scatter(
    x=t, y=h,
    mode="markers", # solo puntos
    name="Datos experimentales",
    marker=dict(color="cyan", size=8)
))
#aqui agregamos el titulo y los estilos de la figura
fig_datos.update_layout( 
    title="Altura vs Tiempo — Datos experimentales",
    xaxis_title="Tiempo (s)",
    yaxis_title="Altura (cm)",
    template="plotly_dark" # fondo negro
)
st.plotly_chart(fig_datos, use_container_width=True) # renderiza o muestra el fig_datos

st.subheader("Ajuste de curvas")
tab1, tab2, tab3,tab4 = st.tabs(["Regresión lineal", "Regresión polinomial", "Regresión exponencial","Comparación"])

with tab1:
    st.subheader("Regresión lineal")
    with st.expander("Ver fórmulas"):
        st.markdown("**Modelo:**")
        st.latex(r"h(t) = a_1 \cdot t + a_0")
        st.markdown("**Coeficientes por mínimos cuadrados:**")
        st.latex(r"a_1 = \frac{n\sum t_i h_i - \sum t_i \sum h_i}{n\sum t_i^2 - (\sum t_i)^2}")
        st.latex(r"a_0 = \bar{h} - a_1 \bar{t}")
        st.markdown("**Estadísticos:**")
        st.latex(r"R^2 = 1 - \frac{S_r}{S_t} \quad S_t = \sum(h_i - \bar{h})^2 \quad S_r = \sum(h_i - \hat{h}_i)^2")
        st.latex(r"S_y = \sqrt{\frac{S_t}{n-1}} \qquad S_{y/x} = \sqrt{\frac{S_r}{n-2}}")
        st.latex(r"S_r = \sum_{i=1}^{n}(h_i - \hat{h}_i)^2")

    # Cálculo
    coef = np.polyfit(t, h, 1) # funcion de coeficientes
    h_lin = np.polyval(coef, t)  # evaluamos el polimonio en cada punto

    # Estadísticos
    n = len(t)
    h_mean = np.mean(h)
    St = np.sum((h - h_mean)**2) # desviacion estandar
    Sr = np.sum((h - h_lin)**2)
    r2 = 1 - Sr/St  # coeficiente de determinacion

    # r con signo correcto usando pearsonr
    r, _ = pearsonr(t, h) # -1 a 1

    Sy = np.sqrt(St/(n-1)) # desvaicion de los datos
    Syx = np.sqrt(Sr/(n-2)) # error estandar

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
    st.metric("Sr (suma de residuos)", f"{Sr:.4f}") 
    col3.metric("Sy (desv. est. total)", f"{Sy:.4f}")
    col4.metric("Syx (error estándar)", f"{Syx:.4f}")
    st.info(f"Ecuación: h(t) = {coef[0]:.4f} · t + {coef[1]:.4f}")

with tab2:
    
    st.subheader("Regresión polinomial grado 2")
    with st.expander("Ver fórmulas"):
        st.markdown("**Modelo:**")
        st.latex(r"h(t) = a_2 t^2 + a_1 t + a_0")
        st.markdown("**Coeficientes por mínimos cuadrados:**")
        st.latex(r"\begin{bmatrix} a_0 \\ a_1 \\ a_2 \end{bmatrix} = (X^T X)^{-1} X^T h")
        st.markdown("**Estadísticos:**")
        st.latex(r"R^2 = 1 - \frac{S_r}{S_t} \qquad S_{y/x} = \sqrt{\frac{S_r}{n-3}}")
        st.latex(r"S_r = \sum_{i=1}^{n}(h_i - \hat{h}_i)^2")

    # Cálculo
    coef2 = np.polyfit(t, h, 2) # funcion de coeficientes
    h_pol = np.polyval(coef2, t) # evaluamos el polimonio en cada punto

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
    st.metric("Sr (suma de residuos)", f"{Sr2:.4f}")
    col3.metric("Sy (desv. est. total)", f"{Sy:.4f}")
    col4.metric("Syx (error estándar)", f"{Syx2:.4f}")

    
with tab3:
    st.subheader("Regresión exponencial no lineal")
    
    with st.expander("Ver fórmulas"):
        st.markdown("**Modelo:**")
        st.latex(r"h(t) = a \cdot e^{b \cdot t}")
        st.markdown("**Linealización:**")
        st.latex(r"\ln(h) = \ln(a) + b \cdot t")
        st.markdown("**Ajuste por curve_fit (Levenberg-Marquardt):**")
        st.latex(r"\min_{a,b} \sum_{i=1}^{n} \left( h_i - a \cdot e^{b t_i} \right)^2")
        st.markdown("**Estadísticos:**")
        st.latex(r"R^2 = 1 - \frac{S_r}{S_t} \qquad S_{y/x} = \sqrt{\frac{S_r}{n-2}}")

    # Modelo exponencial
    def modelo_exp(t, a, b):
        return a * np.exp(b * t)

    # para encontrar los vamlores de a y b que mas se ajusten a los datos
    #p0 = valores iniciales
    # max fev = valor maximo de iteraciones
    # *popt es el array de los valores a y b encontrados
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
    st.metric("Sr (suma de residuos)", f"{Sr3:.4f}")
    col3.metric("Sy (desv. est. total)", f"{Sy:.4f}")
    col4.metric("Syx (error estándar)", f"{Syx3:.4f}")
with tab4:
    st.subheader("Tabla comparativa de regresiones")

    tabla_reg = pd.DataFrame({
        "Modelo": ["Lineal", "Polinomial grado 2", "Exponencial"],
        "Ecuación": [
            f"h = {coef[0]:.4f}t + {coef[1]:.4f}",
            f"h = {coef2[0]:.6f}t² + {coef2[1]:.4f}t + {coef2[2]:.4f}",
            f"h = {popt[0]:.4f}·e^({popt[1]:.6f}·t)"
        ],
        "R²":      [f"{r2:.4f}",     f"{r2_pol:.4f}",  f"{r2_exp:.4f}"],
        "r":       [f"{r:.4f}",      f"{r_pol:.4f}",   f"{r_exp:.4f}"],
        "Sy":      [f"{Sy:.4f}",     f"{Sy:.4f}",      f"{Sy:.4f}"],
        "Syx":     [f"{Syx:.4f}",    f"{Syx2:.4f}",    f"{Syx3:.4f}"],
        "Sr":      [f"{Sr:.4f}",     f"{Sr2:.4f}",     f"{Sr3:.4f}"],
    })
    st.dataframe(tabla_reg, use_container_width=True, hide_index=True)

    # Gráfica comparativa superpuesta
    st.subheader("Gráfica comparativa")
    fig_comp_reg = go.Figure()
    fig_comp_reg.add_trace(go.Scatter(
        x=t, y=h, mode="markers",
        name="Datos experimentales",
        marker=dict(color="cyan", size=8)
    ))
    fig_comp_reg.add_trace(go.Scatter(
        x=t, y=h_lin, mode="lines",
        name="Lineal",
        line=dict(color="red", width=2)
    ))
    fig_comp_reg.add_trace(go.Scatter(
        x=t, y=h_pol, mode="lines",
        name="Polinomial grado 2",
        line=dict(color="orange", width=2)
    ))
    fig_comp_reg.add_trace(go.Scatter(
        x=t, y=h_exp, mode="lines",
        name="Exponencial",
        line=dict(color="lime", width=2)
    ))
    fig_comp_reg.update_layout(
        title="Comparación de regresiones — Lineal, Polinomial y Exponencial",
        xaxis=dict(title=dict(text="Tiempo (s)")),
        yaxis=dict(title=dict(text="Altura (cm)")),
        template="plotly_dark"
    )
    st.plotly_chart(fig_comp_reg, use_container_width=True)

    

st.divider()



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


itab1, itab2, itab3, itab4 = st.tabs(["Lineal", "Cuadrática", "Cúbica", "Cuártica"])
tabs_interp = [itab1, itab2, itab3, itab4]
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
st.dataframe(tabla, use_container_width=True, hide_index=True)


st.subheader("Modelo teórico de Torricelli")

with st.expander("Ver fórmula del modelo teórico"):
    st.latex(r"h(t) = \left(\sqrt{h_0} - k \cdot t\right)^2")
    st.latex(r"k = \frac{C_d \cdot A_o \sqrt{2g}}{2 A_T}")
    st.latex(r"A_T = \frac{\pi d_T^2}{4} \qquad A_o = \frac{\pi d_o^2}{4}")

# Parámetros físicos
Cd = 0.457
g  = 9.8
h0 = 9 / 100

AT = np.pi * (0.06/2)**2 
Ao = np.pi * (0.0012/2)**2

# Constante k
k = (Cd * Ao * np.sqrt(2 * g)) / (2 * AT)

# Tiempo hasta vaciado completo
t_max = np.sqrt(h0) / k
t_teorico = np.linspace(0, t_max, 300)

# Modelo 
h_teorico = (np.sqrt(h0) - k * t_teorico)**2
h_teorico_cm = h_teorico * 100

# Grafica solo del modelo teórico
fig_teorico = go.Figure()
fig_teorico.add_trace(go.Scatter(
    x=t_teorico, y=h_teorico_cm,
    mode="lines",
    name="Modelo teórico Torricelli",
    line=dict(color="cyan", width=2)
))
fig_teorico.update_layout(
    title="Modelo teórico: h(t) = (√h₀ − k·t)²",
    xaxis=dict(title=dict(text="Tiempo (s)")),
    yaxis=dict(title=dict(text="Altura (cm)")),
    template="plotly_dark"
)
st.plotly_chart(fig_teorico, use_container_width=True)

# Parámetros calculados
col1, col2, col3 = st.columns(3)
col1.metric("k", f"{k:.6f}")
col2.metric("AT (m²)", f"{AT:.6f}")
col3.metric("Ao (m²)", f"{Ao:.8f}")

st.divider()

# Comparación modelo teórico vs interpolación cubica
st.subheader("Comparación: Modelo teórico vs Interpolación Cúbica")

# Interpolación cubica en t_teorico 
h_cubica_curva = np.array([lagrange_interp(t, h, ti, 3) for ti in t_teorico])


# Grafica comparativa
fig_comp_teorico = go.Figure()
fig_comp_teorico.add_trace(go.Scatter(
    x=t_teorico, y=h_teorico_cm,
    mode="lines",
    name="Modelo teórico Torricelli",
    line=dict(color="cyan", width=2)
))
fig_comp_teorico.add_trace(go.Scatter(
    x=t_teorico, y=h_cubica_curva,
    mode="lines",
    name="Interpolación Cúbica",
    line=dict(color="lime", width=2, dash="dash")
))
fig_comp_teorico.add_trace(go.Scatter(
    x=t, y=h,
    mode="markers",
    name="Datos experimentales",
    marker=dict(color="yellow", size=8)
))
fig_comp_teorico.update_layout(
    title="Comparación: Modelo teórico de Torricelli vs Interpolación Cúbica",
    xaxis=dict(title=dict(text="Tiempo (s)")),
    yaxis=dict(title=dict(text="Altura (cm)")),
    template="plotly_dark"
)
st.plotly_chart(fig_comp_teorico, use_container_width=True)







# Ocultar menu y footer de Streamlit y asignnando estilos
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
