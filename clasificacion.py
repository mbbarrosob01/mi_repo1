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
import os.path


# Definimos criterion antes de usarlo
criterion = "gini"


# Cargar el conjunto de datos
df = pd.read_csv("D:/Usuarios/mbbarrosob01/Desktop/3eval/clasificacion/atletas_1.csv")


# Eliminar valores faltantes
df.dropna(inplace=True)


# Convertir variables categóricas a números
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])


# Identificar valores atípicos utilizando el método del Rango Intercuartílico (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))


# Función para visualizar valores atípicos con diagramas de caja (boxplots)
def plot_outliers():
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.title("Distribución de Datos y Valores Atípicos")
    plt.show()


# Mostrar los valores atípicos
plot_outliers()


# Visualizar distribuciones de datos con histogramas
df.hist(figsize=(10, 8))
plt.show()


# Mostrar correlación entre variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()


# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X = df.drop("Clasificación", axis=1)
y = df["Clasificación"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Crear y entrenar modelos
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


dec_tree = DecisionTreeClassifier(criterion=criterion)
dec_tree.fit(X_train, y_train)


# Hacer predicciones
y_pred_log = log_reg.predict(X_test)
y_pred_tree = dec_tree.predict(X_test)


# Evaluar modelos
print("Matriz de Confusión - Regresión Logística:")
print(confusion_matrix(y_test, y_pred_log))
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_log))


print("Matriz de Confusión - Árbol de Decisión:")
print(confusion_matrix(y_test, y_pred_tree))
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_tree))


# Curva ROC
# Calcular la Tasa de Falsos Positivos y la Tasa de Verdaderos Positivos para ambos modelos
fpr_log, tpr_log, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fpr_tree, tpr_tree, _ = roc_curve(y_test, dec_tree.predict_proba(X_test)[:, 1])


# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Regresión Logística (AUC = {auc(fpr_log, tpr_log):.2f})")
plt.plot(fpr_tree, tpr_tree, label=f"Árbol de Decisión (AUC = {auc(fpr_tree, tpr_tree):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.legend()
plt.title("Curva ROC")
plt.show()


# Aplicación web con Streamlit
st.title("Análisis de Datos y Modelado")


# Crear un menú lateral
menu = ["Preprocesamiento de Datos", "Métricas del Modelo", "Realizar Predicciones"]
choice = st.sidebar.selectbox("Selecciona una opción", menu)


# Ahora esto actualizará el valor de criterion que ya tiene un valor predeterminado
criterion = st.sidebar.selectbox("Criterio del árbol", ["entropy", "gini"])


# Hago un sidebar donde colocar las variables
# Crear sliders para las variables de entrada
st.sidebar.subheader("Variables de entrada")


# Obtener nombres de columnas excluyendo la variable objetivo
feature_columns = X.columns.tolist()


# Diccionario para almacenar valores de los sliders
input_values = {}


# Crear un slider para cada característica
for feature in feature_columns:
    # Obtener valores mínimo y máximo de la característica
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
   
    # Crear slider con rango apropiado
    input_values[feature] = st.sidebar.slider(
        f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=float(X[feature].mean()),  # Valor predeterminado es la media
        step=(max_val - min_val) / 100  # Crear 100 pasos entre min y max
    )


# Crear un DataFrame con los valores de entrada
input_values_df = pd.DataFrame([input_values])

# Mostrar contenido según la selección del menú
if choice == "Preprocesamiento de Datos":
    st.write("Datos después del preprocesamiento:")
    st.dataframe(df.head())
    st.image('D:/Usuarios/mbbarrosob01/Desktop/3eval/clasificacion/Figure_1.png')
    st.image('D:/Usuarios/mbbarrosob01/Desktop/3eval/clasificacion/Figure_2.png')
elif choice == "Métricas del Modelo":
    st.write("Métricas de Evaluación del Modelo")
   
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
    st.image('D:/Usuarios/mbbarrosob01/Desktop/3eval/clasificacion/Figure_3.png')
    st.image('D:/Usuarios/mbbarrosob01/Desktop/3eval/clasificacion/Figure_4.png')


elif choice == "Realizar Predicciones":
    st.write("Predicciones con los valores seleccionados:")
    st.dataframe(input_values_df)
   
    # Realizar predicciones
    log_pred = log_reg.predict(input_values_df)[0]
    tree_pred = dec_tree.predict(input_values_df)[0]
   
    st.subheader("Predicción con Regresión Logística")
    st.write(f"Resultado: {log_pred}")
    if log_pred == 0:
        st.write("Fondista")
    else:
        st.write("Velocista")
   
    st.subheader("Predicción con Árbol de Decisión")
    st.write(f"Resultado: {tree_pred}")
    if tree_pred == 0:
        st.write("Fondista")
    else:
        st.write("Velocista")
   

