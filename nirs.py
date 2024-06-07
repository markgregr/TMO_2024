import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Функция для загрузки данных
@st.cache
def load_data():
    data = pd.read_csv('Admission_Predict.csv')
    return data

# Функция для предобработки данных
@st.cache
def preprocess_data(data_in):
    data_out = data_in.copy()
    # Масштабирование признаков
    scale_cols = ['University Rating']
    sc = MinMaxScaler()
    data_out[scale_cols] = sc.fit_transform(data_out[scale_cols])
    data_out['GRE Score'] = np.log1p(data_out['GRE Score'])
    return data_out

# Загрузка данных
data = load_data()

# Предобработка данных
data = preprocess_data(data)

# Интерфейс пользователя
st.sidebar.header('Метод ближайших соседей')
cv_slider = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)
step_slider = st.sidebar.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1)

# Подбор гиперпараметра
n_range_list = list(range(1, 100, step_slider))
n_range = np.array(n_range_list)
tuned_parameters = [{'n_neighbors': n_range}]

clf_gs = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=cv_slider, scoring='neg_mean_squared_error')
clf_gs.fit(data.drop(columns=['Chance of Admit ']), data['Chance of Admit '])

best_params = clf_gs.best_params_
best_mse = clf_gs.cv_results_['mean_test_score'][clf_gs.best_index_]

# Получение предсказанных значений для лучшей модели
best_model = clf_gs.best_estimator_
predictions = best_model.predict(data.drop(columns=['Chance of Admit ']))

# Вычисление MAE, RMSE и R2
mae = mean_absolute_error(data['Chance of Admit '], predictions)
rmse = np.sqrt(mean_squared_error(data['Chance of Admit '], predictions))
r2 = r2_score(data['Chance of Admit '], predictions)

# Отображение оценок на интерфейсе
st.subheader('Оценка модели')
st.write('Лучшее значение гиперпараметра (Количество соседей):', best_params['n_neighbors'])
st.write('MSE (Mean Squared Error):', abs(best_mse))
st.write('MAE (Mean Absolute Error):', mae)
st.write('RMSE (Root Mean Squared Error):', rmse)
st.write('R2 Score (Coefficient of Determination):', r2)

# Визуализация метрик на графиках
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# График MSE
axes[0].plot(n_range, clf_gs.cv_results_['mean_test_score'])
axes[0].set_title('Mean Squared Error')
axes[0].set_xlabel('Количество соседей')
axes[0].set_ylabel('MSE')

# График MAE
axes[1].scatter(data['Chance of Admit '], predictions)
axes[1].plot([data['Chance of Admit '].min(), data['Chance of Admit '].max()], 
             [data['Chance of Admit '].min(), data['Chance of Admit '].max()], 'k--')
axes[1].set_title('Mean Absolute Error')
axes[1].set_xlabel('Реальное значение')
axes[1].set_ylabel('Предсказанное значение')

# График RMSE
sns.histplot(data['Chance of Admit '] - predictions, ax=axes[2], kde=True)
axes[2].set_title('Residuals Distribution (RMSE)')
axes[2].set_xlabel('Residuals')

st.pyplot(fig)
