# Plataforma Educativa de Métodos Numéricos

Aplicación web interactiva desarrollada con Streamlit para el aprendizaje y análisis de métodos numéricos de búsqueda de raíces.

## Descripción

Esta plataforma permite a los estudiantes comprender y aplicar métodos numéricos mediante una interfaz intuitiva que combina teoría, cálculos automáticos y visualización gráfica.

La aplicación incluye:

- Explicación teórica de cada método.
- Procedimiento paso a paso.
- Cálculo automático de raíces.
- Historial de iteraciones.
- Visualización gráfica de la convergencia.
- Desarrollo detallado de cada iteración.

## Métodos Implementados

### Newton-Raphson

Método iterativo utilizado para aproximar raíces de ecuaciones no lineales.

Fórmula utilizada:

\[
x_{i+1}=x_i-\frac{f(x_i)}{f'(x_i)}
\]

### Newton-Raphson Modificado

Versión mejorada para funciones con raíces múltiples.

Fórmula utilizada:

\[
x_{i+1}=x_i-\frac{f(x_i)f'(x_i)}
{[f'(x_i)]^2-f(x_i)f''(x_i)}
\]

## Tecnologías Utilizadas

- Python
- Streamlit
- NumPy
- SymPy
- Matplotlib

## Características Principales

- Ingreso de funciones matemáticas personalizadas.
- Procesamiento automático de expresiones algebraicas.
- Cálculo iterativo de raíces.
- Control de tolerancia y número máximo de iteraciones.
- Visualización gráfica del comportamiento de la función.
- Explicación detallada del procedimiento matemático.
- Interfaz amigable orientada al aprendizaje.

## Estructura del Proyecto

```text
Proyecto/
│
├── aplicación Web.py
├── Tabla de fórmulas.avif
└── README.md
```

## Ejecución

Para ejecutar la aplicación:

```bash
streamlit run "aplicación Web.py"
```

## Ejemplos de Funciones

### Newton-Raphson

```text
x^3 - x - 1
x^2 - 4
x^3 - 2x + 1
```

### Newton-Raphson Modificado

```text
(x-2)^2*(x+1)
x^3 - 6x^2 + 12x - 8
2x^2 - 4x + 2
```

## Objetivo

Desarrollar una herramienta educativa que facilite la comprensión de los métodos numéricos mediante simulaciones interactivas, cálculos automatizados y representaciones gráficas del proceso de convergencia.

## Autor

**Adrián Cifuentes**

Universidad Mariano Gálvez de Guatemala
