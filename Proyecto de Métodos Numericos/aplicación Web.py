import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import base64
import re
import os

# Configuración de la página
st.set_page_config(page_title="Métodos Numéricos Educativos", layout="wide")

# --- IMAGEN DE FONDO ---
def poner_fondo_local(nombre_archivo):
    # Buscar el archivo en la misma carpeta del script
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), nombre_archivo)
    try:
        with open(ruta, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            /* Hacemos transparentes las capas por defecto de Streamlit */
            .stApp, .stAppHeader, [data-testid="stAppViewContainer"] {{
                background: transparent !important;
            }}
            
            /* Capa fija del fondo detrás de todo */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: linear-gradient(rgba(0, 0, 0, 0.60), rgba(0, 0, 0, 0.60)), 
                                  url("data:image/avif;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                filter: blur(6px);
                transform: scale(1.03);
                z-index: -1;
            }}
            
            /* Forzamos todos los textos explicativos y títulos a color blanco */
            .stMarkdown, .stSubheader, .stHeader, h1, h2, h3, p, span, h4, h5 {{
                color: #ffffff !important;
            }}
            
            /* Etiquetas de parámetros superiores en blanco */
            label, 
            .stTextInput label, 
            .stNumberInput label, 
            [data-testid="stWidgetLabel"] p,
            div[data-testid="stWidgetLabel"] {{
                color: #ffffff !important;
                font-weight: bold !important;
            }}
            
            /* Fondo oscuro semitransparente en los contenedores de los inputs */
            div[data-baseweb="input"] {{
                background-color: rgba(255, 255, 255, 0.10) !important;
                border-radius: 4px !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
            }}
            
            /* Texto blanco dentro de los inputs */
            div[data-baseweb="input"] input {{
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                font-weight: 500 !important;
                background-color: transparent !important;
            }}

            /* Placeholder en gris claro */
            div[data-baseweb="input"] input::placeholder {{
                color: rgba(255, 255, 255, 0.5) !important;
                -webkit-text-fill-color: rgba(255, 255, 255, 0.5) !important;
            }}
            
            /* Botones de más (+) y menos (-) en blanco */
            div[data-baseweb="input"] button {{
                color: #ffffff !important;
            }}

            /* Input de texto (st.text_input) también en blanco */
            .stTextInput input {{
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                background-color: rgba(255, 255, 255, 0.10) !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
            }}
            
            /* Contenedores de tablas y formularios */
            .stTable, .stForm, div[data-testid="stMetricValue"], .stDataFrame {{
                background-color: rgba(255, 255, 255, 0.08) !important;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass  # Si no hay imagen de fondo, continuar sin error visible


# Llamada a la función (una sola vez)
poner_fondo_local("Tabla de fórmulas.avif")


st.title("Plataforma Educativa: Métodos Numéricos")
st.write("Bienvenido. Selecciona un método en la barra lateral para explorar su teoría, gráfica y calculadora interactiva.")

# Selector de método en la barra lateral
metodo = st.sidebar.selectbox(
    "Selecciona el Método:",
    ["Newton-Raphson", "Newton-Raphson Modificado"]
)


# --- FUNCIÓN DE PREPROCESAMIENTO MATEMÁTICO ---
def preprocesar_expresion(expr_str):
    """
    Convierte notación matemática informal a sintaxis válida de SymPy.
    Ejemplos:
        x^3*2x   →  x**3*2*x
        2x^2     →  2*x**2
        (x-2)(x+1) →  (x-2)*(x+1)
        3x(x+1)  →  3*x*(x+1)
    """
    # 1. Reemplazar ^ por ** (potenciación)
    expr_str = expr_str.replace("^", "**")

    # 2. Multiplicación implícita: número seguido de letra o paréntesis → 2x, 3(x+1)
    expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)

    # 3. Multiplicación implícita: letra/cierre de paréntesis seguido de paréntesis abierto → x(, )(
    expr_str = re.sub(r'([a-zA-Z)])(\()', r'\1*\2', expr_str)

    return expr_str


# --- FUNCIONES DE CÁLCULO NUMÉRICO ---
def analizar_funcion(expr_str):
    """Convierte un string a funciones de Python usando Sympy."""
    x = sp.symbols('x')
    try:
        # Preprocesar antes de pasar a SymPy
        expr_str = preprocesar_expresion(expr_str)

        expr = sp.sympify(expr_str, locals={"x": x})
        f_num = sp.lambdify(x, expr, "numpy")

        # Derivada primera
        f_der = sp.diff(expr, x)
        f_der_num = sp.lambdify(x, f_der, "numpy")

        # Derivada segunda
        f_der2 = sp.diff(f_der, x)
        f_der2_num = sp.lambdify(x, f_der2, "numpy")

        return expr, f_num, f_der_num, f_der2_num, None
    except Exception as e:
        return None, None, None, None, str(e)


# --- CONTENIDO SEGÚN EL MÉTODO ---

if metodo == "Newton-Raphson":
    st.header("Método de Newton-Raphson")
    
    # 1. Definición y Fundamento Teórico
    st.subheader("1. Definición y Fundamento Teórico")
    st.write("""
    El método de **Newton-Raphson** es un algoritmo iterativo abierto utilizado para encontrar aproximaciones de los ceros o raíces de una función real $f(x) = 0$. 
    Se basa en la linealización de la función mediante su aproximación por la recta tangente en un punto estimado inicial $x_0$.
    """)
    st.latex(r"X_{i+1} = X_i - \frac{f(X_i)}{f'(X_i)}")
    st.info("💡 **Condición de convergencia:** La derivada $f'(x_i)$ no debe ser igual a cero en ningún punto de la iteración.")

    # 2. Procedimiento Paso a Paso
    st.subheader("2. Procedimiento Paso a Paso")
    st.markdown("""
    1. **Definir** la función $f(x)$ y obtener su primera derivada $f'(x)$.
    2. **Establecer** un valor inicial aproximado $x_0$.
    3. **Calcular** la siguiente aproximación utilizando la fórmula iterativa.
    4. **Evaluar el error** absoluto o relativo $|X_{i+1} - X_i|$.
    5. **Repetir** el proceso hasta que el error sea menor que la tolerancia predefinida ($\\epsilon$) o se alcance el límite de iteraciones.
    """)

    # 3. Calculadora Interactiva
    st.subheader("3. Calculadora Interactiva")
    st.caption("Puedes escribir funciones como: `x^3 - 2x + 1`, `x^3*2x`, `(x-1)(x+2)`, `3x^2 - 5`")
    
    with st.form("formulario_newton_raphson"):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Parámetros de Entrada")
            func_input = st.text_input("Función f(x):", "x**3 - x - 1")
            x0 = st.number_input("Valor inicial (x0):", value=1.5)
            tol = st.number_input("Tolerancia (ε):", value=0.001, format="%.5f")
            max_iter = st.number_input("Máximo de iteraciones:", value=10, min_value=1)
            
            botón_calcular = st.form_submit_button("Calcular Raíz", use_container_width=True)
            
    if botón_calcular:
        expr, f, df, d2f, error_msg = analizar_funcion(func_input)
        
        if error_msg:
            st.error(f"Error en la expresión matemática: {error_msg}")
        else:
            pasos = []
            xi = x0
            convergio = False
            
            for i in range(int(max_iter)):
                try:
                    f_val = f(xi)
                    df_val = df(xi)
                    
                    if abs(df_val) < 1e-12:
                        st.warning(f"La derivada es cercana a cero en x = {xi}. El método puede fallar.")
                        break
                        
                    xi_next = xi - (f_val / df_val)
                    err = abs(xi_next - xi)
                    
                    pasos.append({"Iteración": i+1, "X_i": xi, "f(X_i)": f_val, "f'(X_i)": df_val, "X_{i+1}": xi_next, "Error": err})
                    
                    if err < tol:
                        convergio = True
                        xi = xi_next
                        break
                    xi = xi_next
                except Exception as e:
                    st.error(f"Error numérico durante el cálculo: {e}")
                    break

            # Mostrar Resultados Geométricos en la columna de la derecha
            with col2:
                st.write("### 4. Representación Gráfica")
                if pasos:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('none')
                    
                    valores_x = [p["X_i"] for p in pasos] + [xi]
                    xmin, xmax = min(valores_x) - 1, max(valores_x) + 1
                    x_vals = np.linspace(xmin, xmax, 400)
                    
                    ax.plot(x_vals, f(x_vals), label='$f(x)$', color='#00d2ff', lw=2)
                    ax.axhline(0, color='white', linestyle='--', linewidth=0.8)
                    
                    for idx, p in enumerate(pasos[:3]):
                        ax.plot(p["X_i"], p["f(X_i)"], 'ro')
                        t_vals = df(p["X_i"]) * (x_vals - p["X_i"]) + f(p["X_i"])
                        ax.plot(x_vals, t_vals, linestyle=':', alpha=0.7, label=f'Tangente Iter {p["Iteración"]}')
                    
                    ax.set_title("Visualización de las aproximaciones sucesivas", color='white')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.legend()
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)
            
            # Tabla de iteraciones
            st.write("### Historial de Convergencia")
            if pasos:
                st.table(pasos)
                st.success(f"**Resultado obtenido:** Raíz aproximada en **{xi:.6f}** en la iteración {len(pasos)}.")

                # BOTÓN PASO A PASO
                st.write("---")
                with st.expander("Ver Paso a Paso detallado"):
                    st.write("## Desarrollo Paso a Paso")
                    for p in pasos:
                        with st.expander(f"Iteración {p['Iteración']}"):
                            st.markdown(f"""
**Valor actual:** $x_{{{p['Iteración']-1}}} = {p['X_i']:.6f}$

**Paso 1 — Evaluar la función:**
$$f({p['X_i']:.6f}) = {p['f(X_i)']:.6f}$$

**Paso 2 — Evaluar la derivada:**
$$f'({p['X_i']:.6f}) = {p["f'(X_i)"]:.6f}$$

**Paso 3 — Aplicar la fórmula:**
$$x_{{{p['Iteración']}}} = {p['X_i']:.6f} - \\frac{{{p['f(X_i)']:.6f}}}{{{p["f'(X_i)"]:.6f}}} = {p['X_{i+1}']:.6f}$$

**Paso 4 — Error:**
$$|{p['X_{i+1}']:.6f} - {p['X_i']:.6f}| = {p['Error']:.6f}$$
""")
            else:
                st.error("No se pudieron generar iteraciones válidas.")


elif metodo == "Newton-Raphson Modificado":
    st.header("Método de Newton-Raphson Modificado")
    
    # 1. Definición y Fundamento Teórico
    st.subheader("1. Definición y Fundamento Teórico")
    st.write("""
    El método de **Newton-Raphson Modificado** está diseñado específicamente para resolver problemas donde existen **raíces múltiples** (puntos donde la función y sus derivadas se anulan simultáneamente). 
    El método tradicional de Newton-Raphson pierde su velocidad de convergencia cuadrática ante raíces múltiples; esta modificación introduce la segunda derivada $f''(x)$ para restaurar la eficiencia del algoritmo.
    """)
    st.latex(r"X_{i+1} = X_i - \frac{f(X_i) \cdot f'(X_i)}{[f'(X_i)]^2 - f(X_i) \cdot f''(X_i)}")

    # 2. Procedimiento Paso a Paso
    st.subheader("2. Procedimiento Paso a Paso")
    st.markdown("""
    1. **Determinar** analíticamente la primera derivada $f'(x)$ y la segunda derivada $f''(x)$ de la función original.
    2. **Ingresar** una estimación inicial $x_0$.
    3. **Aplicar** la fórmula modificada usando el valor actual de la función y sus dos primeras derivadas.
    4. **Verificar** si la diferencia o error relativo cumple con el criterio de parada.
    5. **Iterar** consecutivamente hasta alcanzar la precisión requerida.
    """)

    # 3. Calculadora Interactiva
    st.subheader("3. Calculadora Interactiva")
    st.caption("Puedes escribir funciones como: `(x-2)^2*(x+1)`, `x^3 - 6x^2 + 12x - 8`, `2x^2 - 4x + 2`")
    
    with st.form("formulario_newton_modificado"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Parámetros de Entrada")
            func_input = st.text_input("Función f(x) (ejemplo con raíz múltiple):", "(x-2)**2 * (x+1)")
            x0 = st.number_input("Valor inicial (x0):", value=1.7)
            tol = st.number_input("Tolerancia (ε):", value=0.001, format="%.5f")
            max_iter = st.number_input("Máximo de iteraciones:", value=10, min_value=1)
            
            botón_calcular_mod = st.form_submit_button("Calcular Raíz Múltiple", use_container_width=True)
            
    if botón_calcular_mod:
        expr, f, df, d2f, error_msg = analizar_funcion(func_input)
        
        if error_msg:
            st.error(f"Error en la expresión matemática: {error_msg}")
        else:
            pasos = []
            xi = x0
            convergio = False
            
            for i in range(int(max_iter)):
                try:
                    f_val = f(xi)
                    df_val = df(xi)
                    d2f_val = d2f(xi)
                    
                    denominador = (df_val**2) - (f_val * d2f_val)
                    
                    if abs(denominador) < 1e-12:
                        st.warning("El denominador se volvió cero o cercano a cero. El método se detuvo por seguridad.")
                        break
                    
                    xi_next = xi - (f_val * df_val) / denominador
                    err = abs(xi_next - xi)
                    
                    pasos.append({
                        "Iteración": i+1, 
                        "X_i": xi, 
                        "f(X_i)": f_val, 
                        "f'(X_i)": df_val, 
                        "f''(X_i)": d2f_val,
                        "X_{i+1}": xi_next, 
                        "Error": err
                    })
                    
                    if err < tol:
                        convergio = True
                        xi = xi_next
                        break
                    xi = xi_next
                except Exception as e:
                    st.error(f"Error numérico durante el cálculo: {e}")
                    break

            # 4. Representación Gráfica
            with col2:
                st.write("### 4. Representación Gráfica")
                if pasos:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('none')
                    
                    valores_x = [p["X_i"] for p in pasos] + [xi]
                    xmin, xmax = min(valores_x) - 1.5, max(valores_x) + 1.5
                    x_vals = np.linspace(xmin, xmax, 400)
                    
                    try:
                        ax.plot(x_vals, f(x_vals), label='$f(x)$ (Función)', color='#ff007f', lw=2)
                        ax.axhline(0, color='white', linestyle='--', linewidth=0.8)
                        
                        for p in pasos:
                            ax.plot(p["X_i"], f(p["X_i"]), 'mo', markersize=6)
                    except Exception as graph_err:
                        st.warning(f"No se pudo graficar la función en este rango: {graph_err}")
                        
                    ax.set_title("Convergencia en raíces múltiples (Newton Modificado)", color='white')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.legend()
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)
            
            # Historial de iteraciones
            st.write("### Historial de Convergencia")
            if pasos:
                st.table(pasos)
                st.success(f"**Resultado obtenido:** Raíz múltiple aproximada en **{xi:.6f}** en la iteración {len(pasos)}.")

                # BOTÓN PASO A PASO
                st.write("---")
                with st.expander("Ver Paso a Paso detallado"):
                    st.write("## Desarrollo Paso a Paso")
                    for p in pasos:
                        with st.expander(f"Iteración {p['Iteración']}"):
                            st.markdown(f"""
**Valor actual:** $x_{{{p['Iteración']-1}}} = {p['X_i']:.6f}$

**Paso 1 — Evaluar la función y sus derivadas:**
$$f({p['X_i']:.6f}) = {p['f(X_i)']:.6f}$$
$$f'({p['X_i']:.6f}) = {p["f'(X_i)"]:.6f}$$
$$f''({p['X_i']:.6f}) = {p["f''(X_i)"]:.6f}$$

**Paso 2 — Calcular el denominador:**
$$[f'(x)]^2 - f(x) \\cdot f''(x) = ({p["f'(X_i)"]:.6f})^2 - ({p['f(X_i)']:.6f})({p["f''(X_i)"]:.6f}) = {p["f'(X_i)"]**2 - p['f(X_i)']*p["f''(X_i)"]:.6f}$$

**Paso 3 — Aplicar la fórmula modificada:**
$$x_{{{p['Iteración']}}} = {p['X_i']:.6f} - \\frac{{({p['f(X_i)']:.6f})({p["f'(X_i)"]:.6f})}}{{{p["f'(X_i)"]**2 - p['f(X_i)']*p["f''(X_i)"]:.6f}}} = {p['X_{i+1}']:.6f}$$

**Paso 4 — Error:**
$$|{p['X_{i+1}']:.6f} - {p['X_i']:.6f}| = {p['Error']:.6f}$$
""")
            else:
                st.error("No se pudieron generar iteraciones válidas.")