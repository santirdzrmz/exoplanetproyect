# Guardar este código como app_gradio.py

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PREPARACIÓN DE DATOS (El Backend) ---
# Se ejecuta una sola vez al iniciar la app.
print("Cargando y preparando los datos de Kepler...")
try:
    df = pd.read_csv('KEPLER.csv', comment='#')
    # Obtenemos las listas de columnas para los menús desplegables
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    columnas_categoricas = ['koi_pdisposition', 'koi_disposition', 'koi_vet_stat']
    print("Datos listos.")
except FileNotFoundError:
    print("Error: Asegúrate de que 'KEPLER.csv' está en el mismo directorio.")
    df = pd.DataFrame() # Crear un DataFrame vacío si hay error

# --- 2. LA FUNCIÓN PRINCIPAL (El Corazón de la App) ---
# Esta función es la que se ejecutará cada vez que el usuario interactúe.
def generar_grafica(col_categorica, col_numerica):
    """
    Toma los nombres de dos columnas y devuelve un diagrama de cajas de Matplotlib.
    """
    if df.empty or col_categorica not in df or col_numerica not in df:
        return None # Devuelve nada si hay un error o no hay datos

    # Filtrar datos nulos para evitar errores en la gráfica
    df_plot = df[[col_categorica, col_numerica]].dropna()
    
    # Crear la figura y los ejes para la gráfica
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generar el diagrama de cajas
    sns.boxplot(data=df_plot, x=col_categorica, y=col_numerica, ax=ax)
    
    # Mejorar la visualización
    ax.set_title(f'Relación entre {col_categorica} y {col_numerica}', fontsize=14)
    ax.set_yscale('log') # Escala logarítmica es útil aquí
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # ¡Importante! Gradio necesita que la función devuelva el objeto de la figura.
    return fig

# --- 3. CREAR Y LANZAR LA INTERFAZ ---
# Aquí es donde definimos la interfaz y conectamos todo.
demo = gr.Interface(
    fn=generar_grafica,  # La función que se ejecutará
    inputs=[
        gr.Dropdown(choices=columnas_categoricas, label="Elige una Columna Categórica (Eje X)"),
        gr.Dropdown(choices=columnas_numericas, label="Elige una Columna Numérica (Eje Y)")
    ],
    outputs=gr.Plot(label="Gráfica de Relación"), # El componente para mostrar el resultado
    title="Explorador de Datos Kepler con Gradio",
    description="Selecciona dos columnas para visualizar su relación. La gráfica muestra la distribución de la columna numérica para cada categoría de la columna categórica."
)

# Este comando lanza la aplicación web
if __name__ == "__main__":
    demo.launch()
