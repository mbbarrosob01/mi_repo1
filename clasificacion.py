import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import streamlit as st

# Configurar criterio inicial
criterion = "gini"

# Cargar el conjunto de datos
df = pd.read_csv("atletas_1.csv")

# Eliminar valores faltantes
df.dropna(inplace=True)

# Convertir variables categóricas a numéricas
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Identificar valores atípicos (opcional: podrías eliminarlos)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# Separar características y etiquetas
X = df.drop("Clasificación", axis=1)
y = df["Clasificación"]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelos
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

dec_tree = DecisionTreeClassifier(criterion=criterion)
dec_tree.fit(X_train, y_train)

# Predicciones
y_pred_log = log_reg.predict(X_test)
y_pred_tree = dec_tree.predict(X_test)

# Curvas ROC
fpr_log, tpr_log, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fpr_tree, tpr_tree, _ = roc_curve(y_test, dec_tree.predict_proba(X_test)[:, 1])

# Funciones de gráficos dinámicos
def plot_outliers():
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, ax=ax)
    ax.set_title("Distribución de Datos y Valores Atípicos")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def plot_histogram():
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=20, color='lightblue', edgecolor='black')
        axes[i].set_title(f'Histograma de {col}')

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])  # Eliminar subgráficos vacíos

    plt.tight_layout()
    st.pyplot(fig)



def plot_correlation():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de Correlación")
    st.pyplot(fig)

def plot_roc_curve():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_log, tpr_log, label=f"Regresión Logística (AUC = {auc(fpr_log, tpr_log):.2f})")
    ax.plot(fpr_tree, tpr_tree, label=f"Árbol de Decisión (AUC = {auc(fpr_tree, tpr_tree):.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.legend()
    ax.set_title("Curva ROC")
    st.pyplot(fig)

# Interfaz Streamlit
st.title("Análisis de Datos y Modelado")

# Menú lateral
menu = ["Preprocesamiento de Datos", "Métricas del Modelo", "Realizar Predicciones"]
choice = st.sidebar.selectbox("Selecciona una opción", menu)

# Selector de criterio para árbol (aunque se entrena antes del cambio)
criterion = st.sidebar.selectbox("Criterio del árbol", ["entropy", "gini"])

# Variables de entrada
st.sidebar.subheader("Variables de entrada")
feature_columns = X.columns.tolist()
input_values = {}

for feature in feature_columns:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    input_values[feature] = st.sidebar.slider(
        f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=float(X[feature].mean()),
        step=(max_val - min_val) / 100
    )

input_values_df = pd.DataFrame([input_values])

# Contenido según menú
if choice == "Preprocesamiento de Datos":
    st.subheader("Datos después del preprocesamiento")
    st.dataframe(df.head())
    plot_outliers()
    plot_histogram()
    plot_correlation()

elif choice == "Métricas del Modelo":
    st.subheader("Resultados de Regresión Logística")
    st.text("Matriz de Confusión:")
    st.text(confusion_matrix(y_test, y_pred_log))
    st.text("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred_log))

    st.subheader("Resultados del Árbol de Decisión")
    st.text("Matriz de Confusión:")
    st.text(confusion_matrix(y_test, y_pred_tree))
    st.text("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred_tree))

    plot_roc_curve()

elif choice == "Realizar Predicciones":
    st.subheader("Predicciones con los valores seleccionados")
    st.dataframe(input_values_df)

    log_pred = log_reg.predict(input_values_df)[0]
    tree_pred = dec_tree.predict(input_values_df)[0]

    st.subheader("Predicción con Regresión Logística")
    st.write(f"Resultado: {log_pred}")
    st.write("Fondista" if log_pred == 0 else "Velocista")

    st.subheader("Predicción con Árbol de Decisión")
    st.write(f"Resultado: {tree_pred}")
    st.write("Fondista" if tree_pred == 0 else "Velocista")
