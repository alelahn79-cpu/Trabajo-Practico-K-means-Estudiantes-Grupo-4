import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================================
# 5. Trabajar con el set de datos de estudiantes (Dataset de entrada)
# =========================================================================
estudiantes = [
    {"edad": 42, "horas_estudio": 2, "promedio_academico": 8.87},
    {"edad": 26, "horas_estudio": 8, "promedio_academico": 9.11},
    {"edad": 56, "horas_estudio": 5, "promedio_academico": 7.29},
    {"edad": 29, "horas_estudio": 14, "promedio_academico": 7.27},
    {"edad": 43, "horas_estudio": 8, "promedio_academico": 8.5},
    {"edad": 34, "horas_estudio": 10, "promedio_academico": 4.45},
    {"edad": 51, "horas_estudio": 3, "promedio_academico": 6.12},
    {"edad": 22, "horas_estudio": 12, "promedio_academico": 4.33},
    {"edad": 47, "horas_estudio": 6, "promedio_academico": 7.0},
    {"edad": 38, "horas_estudio": 9, "promedio_academico": 4.75},
    {"edad": 60, "horas_estudio": 2, "promedio_academico": 9.0},
    {"edad": 31, "horas_estudio": 11, "promedio_academico": 2.0},
    {"edad": 45, "horas_estudio": 4, "promedio_academico": 8.0},
    {"edad": 27, "horas_estudio": 13, "promedio_academico": 3.2},
    {"edad": 50, "horas_estudio": 5, "promedio_academico": 7.0},
    {"edad": 36, "horas_estudio": 7, "promedio_academico": 5.0},
    {"edad": 40, "horas_estudio": 6, "promedio_academico": 6.0},
    {"edad": 24, "horas_estudio": 15, "promedio_academico": 3.0},
    {"edad": 55, "horas_estudio": 3, "promedio_academico": 9.0},
    {"edad": 33, "horas_estudio": 10, "promedio_academico": 3.0},
    {"edad": 48, "horas_estudio": 4, "promedio_academico": 8.0},
    {"edad": 28, "horas_estudio": 12, "promedio_academico": 2.0},
    {"edad": 52, "horas_estudio": 5, "promedio_academico": 7.0},
    {"edad": 30, "horas_estudio": 11, "promedio_academico": 2.0},
    {"edad": 46, "horas_estudio": 6, "promedio_academico": 6.0},
    {"edad": 35, "horas_estudio": 9, "promedio_academico": 4.0},
    {"edad": 41, "horas_estudio": 7, "promedio_academico": 5.0},
    {"edad": 23, "horas_estudio": 14, "promedio_academico": 1.0},
    {"edad": 53, "horas_estudio": 3, "promedio_academico": 8.0},
    {"edad": 32, "horas_estudio": 10, "promedio_academico": 3.0},
    {"edad": 49, "horas_estudio": 4, "promedio_academico": 8.0},
    {"edad": 25, "horas_estudio": 13, "promedio_academico": 2.0},
    {"edad": 54, "horas_estudio": 5, "promedio_academico": 7.0},
    {"edad": 37, "horas_estudio": 8, "promedio_academico": 4.0},
    {"edad": 39, "horas_estudio": 7, "promedio_academico": 5.0},
    {"edad": 21, "horas_estudio": 15, "promedio_academico": 9.0},
    {"edad": 58, "horas_estudio": 2, "promedio_academico": 9.0},
    {"edad": 44, "horas_estudio": 6, "promedio_academico": 6.0},
    {"edad": 29, "horas_estudio": 12, "promedio_academico": 2.0},
    {"edad": 50, "horas_estudio": 5, "promedio_academico": 7.0},
    {"edad": 34, "horas_estudio": 9, "promedio_academico": 3.0},
    {"edad": 42, "horas_estudio": 7, "promedio_academico": 6.0},
    {"edad": 26, "horas_estudio": 14, "promedio_academico": 3.0},
    {"edad": 57, "horas_estudio": 3, "promedio_academico": 9.0},
    {"edad": 30, "horas_estudio": 11, "promedio_academico": 2.0},
    {"edad": 47, "horas_estudio": 4, "promedio_academico": 7.0},
    {"edad": 31, "horas_estudio": 10, "promedio_academico": 3.0},
    {"edad": 45, "horas_estudio": 6, "promedio_academico": 6.0},
    {"edad": 27, "horas_estudio": 13, "promedio_academico": 1.0},
    {"edad": 55, "horas_estudio": 5, "promedio_academico": 8.0},
]

# Convertimos la lista en un DataFrame de Pandas
df = pd.DataFrame(estudiantes)


# ============================================================
# 1. Estandarizar los datos para aplicar K-Means correctamente
# ============================================================
features = ["edad", "horas_estudio", "promedio_academico"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])


# =============================================
# 2. Agrupar a los estudiantes en 4 segmentos
# =============================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)


# ===================================================================
# 4. Realizar un informe con el análisis de los resultados obtenidos 
# ===================================================================
print("=========================================================")
print(" INFORME: CENTROS PROMEDIO DE CADA CLÚSTER (VALORES REALES)")
print("=========================================================")
centros_originales = scaler.inverse_transform(kmeans.cluster_centers_)
df_centros = pd.DataFrame(centros_originales, columns=features)
print(df_centros.to_string())
print("---------------------------------------------------------\n")

# ================================================================
# 3. Visualizar los resultados en un gráfico 2D (edad vs promedio)
# ================================================================
plt.figure(figsize=(10, 6))

# Dibujamos los puntos con la barra lateral de colores (estilo limpio)
scatter = plt.scatter(
    df["edad"],
    df["promedio_academico"],
    c=df["cluster"],
    cmap="viridis",
    s=80,
    edgecolor="k",
)

# Graficamos los centroides proyectados en el plano 2D (Edad vs Promedio)
plt.scatter(
    centros_originales[:, 0],  # Columna 0: Edad
    centros_originales[:, 2],  # Columna 2: Promedio Académico
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    linewidths=1.5,
    label="Centroides",
)

# Configuraciones visuales del gráfico
plt.colorbar(scatter, label="Cluster")
plt.title("Segmentación de Estudiantes Universitarios (K-Means)", fontsize=14, fontweight="bold")
plt.xlabel("Edad (Años)", fontsize=12)
plt.ylabel("Promedio Académico", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()