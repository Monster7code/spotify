<h1>Гушшамов Кирилл spotify</h1>


import numpy as np <br>
import pandas as pd<br>
import matplotlib.pyplot as plt<br>


import os<br>

df = pd.read_csv('/content/dataset.csv')<br>
df.head()<br>
![image](https://github.com/user-attachments/assets/9bafe9a3-dcd9-4d05-9e18-9baf33e97a0c)


<h2><b>Удаляю track id так как он не особо нужен</b></h2><br>

del df["track_id"]<br>
df.head()<br>
![image](https://github.com/user-attachments/assets/b93002e0-4cfc-4fff-b772-b436f30286da)


<h2><b>Графики:</b></h2>

Понять, как распределена популярность треков в нашем наборе данных. <br>
Это поможет нам увидеть, есть ли треки с высокой популярностью и как они соотносятся с менее популярными.

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title('Распределение популярности треков')
plt.xlabel('Популярность')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

![image](https://github.com/user-attachments/assets/b3e7e71f-08fd-4095-a170-ff6e1f267f9a)

Нужно выяснить, какие признаки имеют сильную корреляцию с целевой переменной (популярностью)
![image](https://github.com/user-attachments/assets/66398478-ca77-4379-9e45-4a33cdcbe73c)


numerical_data = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()
correlation_matrix.corr()['popularity'].sort_values(ascending=False)

![image](https://github.com/user-attachments/assets/d8ab1657-feae-40e8-bcfc-77cfb53b3fc1)

<h2>Feature Engineering. Корреляция новых колонок с таргетом. Feature Importances. Простая модель.</h2>
Смотрим как жанр влияает на популярность

genre_popularity = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)

# Строим график
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_popularity.index, y=genre_popularity.values, palette='viridis')
plt.title('Средняя популярность по жанрам')
plt.xlabel('Жанр')
plt.ylabel('Средняя популярность')
plt.xticks(rotation=45)
plt.show()

![image](https://github.com/user-attachments/assets/0156edd8-b65c-476f-be5d-4a4f2cf14aee)



Заменяю признак на категориальный

from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
df['track_genre_encoded'] = label_encoder.fit_transform(df['track_genre'])

# Проверим результат
print(df[['track_genre', 'track_genre_encoded']])

![image](https://github.com/user-attachments/assets/aba907be-21a6-460f-85fb-00f641f8bd51)

Дополнительно проверю весь датасет

![image](https://github.com/user-attachments/assets/ffd6f07f-997c-424b-a3a8-aea9fec2e585)

Удаляю лишний столбец

del df["track_genre"]
# again check
df

![image](https://github.com/user-attachments/assets/f9990569-08b5-434f-ba33-d44602ef1c0c)

Предполагаю, что при поиске определенные названия будут попадаться чаще 
<b>upd:</b> совсем забыл что есть nan значения) получал очень долго ошиюку не мог понять

df.isnull().sum()

![image](https://github.com/user-attachments/assets/9b66ff8b-a412-40e6-be12-993fd5ad384f)

df = df.dropna()
df.isnull().sum()

![image](https://github.com/user-attachments/assets/0aab0ddb-4425-42b2-8700-953d5a3c0ae9)

Возвращаюсь к теории
keywords = ['love', 'night', 'dance', 'party', 'dream',
            "comedy", "fun", "funny","sad","melancholy","sorrow","lonely",
            "mournful","heartbroken","despair","regret","nostalgia","wistful",
            "forlorn","dismal","doleful","grief","pensive", "hold", "destiny"]

# Функция для подсчета веса на основе совпадений
def calculate_weight(track_name):
    weight = 0
    for word in keywords:
        if word in track_name.lower():  # Приводим к нижнему регистру для точного сравнения
            weight += 1  # Увеличиваем вес за каждое совпадение
    return weight

# Применяем функцию к столбцу track_name
df['track_weight'] = df['track_name'].apply(calculate_weight)

# Проверим результат
df[['track_name', 'track_weight']].head()

![image](https://github.com/user-attachments/assets/9b251369-8d17-45e7-b6e9-67edebce5d2a)

Теперь нужно удалить лишнее

del df["track_name"]

Теперь тоже самое хочу сделать и с названием альбома

df['album_weight'] = df['album_name'].apply(calculate_weight)

# Проверим результат
df[['album_name', 'album_weight']].head()


del df["album_name"]
df

Также удаляю лишнее
![image](https://github.com/user-attachments/assets/e49d71cb-3995-4962-9f9b-db14bc67cdab)

резко пришла мысль, что есть корреляция от длителности и популярности


plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration_ms', y='popularity', data=df)

# Добавление заголовка и меток осей
plt.title('Корреляция между длительностью трека и популярностью')
plt.xlabel('Длительность (мс)')
plt.ylabel('Популярность')

# Отображение графика
plt.show()

![image](https://github.com/user-attachments/assets/9ccce083-9cfc-40ff-aaaf-da58b38af2e5)

Как видно если песня меньше опр значения, то она популярнее поэтому добавляю новую фичу

df['short_duration'] = df['duration_ms'] < 200000  # 10000 мс = 1 секунда

# Преобразование булевой переменной в целочисленный формат (0 и 1)
df['short_duration'] = df['short_duration'].astype(int)

# Просмотр первых нескольких строк данных с новой фичей
df[['duration_ms', 'popularity', 'short_duration']].head()

![image](https://github.com/user-attachments/assets/0eb4d1fc-6a4a-4eaf-8980-886fa4a7b118)

Доп импорты для пострения моделей

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# подготовка данных
X = df.drop(columns=['popularity'])  # Используем все признаки, кроме 'popularity'
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Вновь получаю ошибку что данные не соответсвуют
удаляю имя 

del df["artists"]

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Прогноз и оценка
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Линейная регрессия - MSE: {mse_lr}')

# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Линейная регрессия: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()

![image](https://github.com/user-attachments/assets/5bb9f6e7-b6ad-4791-92c6-37eb1cbaf267)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Создание модели
lin_reg = LinearRegression()

# Кросс-валидация
scores = cross_val_score(lin_reg, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

# Визуализация ошибок
plt.figure(figsize=(10, 5))
plt.plot(mse_scores, marker='o', label='Линейная регрессия')
plt.title('Ошибки кросс-валидации для линейной регрессии')
plt.xlabel('Фолд')
plt.ylabel('MSE')
plt.legend()
plt.show()

![image](https://github.com/user-attachments/assets/a844b412-c238-4aed-880d-9ec1b8dcb8a4)

<h2>Регрессия дерева решений</h2>
# Обучение модели
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# Прогноз и оценка
y_pred_dt = model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f'Дерево решений - MSE: {mse_dt}')

# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Дерево решений: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()

![image](https://github.com/user-attachments/assets/9f66ea44-9602-431b-a22c-bf97417f3309)


tree_reg = DecisionTreeRegressor()

# Кросс-валидация
scores = cross_val_score(tree_reg, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

# Визуализация ошибок
plt.figure(figsize=(10, 5))
plt.plot(mse_scores, marker='o', label='Дерево решений')
plt.title('Ошибки кросс-валидации для дерева решений')
plt.xlabel('Фолд')
plt.ylabel('MSE')
plt.legend()
plt.show()

![image](https://github.com/user-attachments/assets/7421bb3f-b7ea-456b-8135-0308ff412086)

<h2>Градиентный бустинг</h2>

# Обучение модели
model_gb = GradientBoostingRegressor(random_state=42)
model_gb.fit(X_train, y_train)

# Прогноз и оценка
y_pred_gb = model_gb.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
print(f'Градиентный бустинг - MSE: {mse_gb}')

# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Градиентный бустинг: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()

![image](https://github.com/user-attachments/assets/d3d139d3-4c4e-48dc-b28b-a9f4878692cf)


gb_reg = GradientBoostingRegressor()

# Кросс-валидация
scores = cross_val_score(gb_reg, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

# Визуализация ошибок
plt.figure(figsize=(10, 5))
plt.plot(mse_scores, marker='o', label='Градиентный бустинг')
plt.title('Ошибки кросс-валидации для градиентного бустинга')
plt.xlabel('Фолд')
plt.ylabel('MSE')
plt.legend()
plt.show()

![image](https://github.com/user-attachments/assets/5c9670f1-bb4d-46e1-aab7-0aa8619c9d2a)

<h2>Нейронная сеть</h2>
# Обучение модели
model_nn = MLPRegressor(random_state=42, max_iter=1000)
model_nn.fit(X_train, y_train)

# Прогноз и оценка
y_pred_nn = model_nn.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Нейронная сеть - MSE: {mse_nn}')

# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Нейронная сеть: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()

![image](https://github.com/user-attachments/assets/591f4a2a-cc15-4f38-84e5-3cc861bdf5a2)

# Создание модели
nn_reg = MLPRegressor(max_iter=1000)

# Кросс-валидация
scores = cross_val_score(nn_reg, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

# Визуализация ошибок
plt.figure(figsize=(10, 5))
plt.plot(mse_scores, marker='o', label='Нейронная сеть')
plt.title('Ошибки кросс-валидации для нейронной сети')
plt.xlabel('Фолд')
plt.ylabel('MSE')
plt.legend()
plt.show()

![image](https://github.com/user-attachments/assets/bb6fbcc4-8349-498f-9f68-fc6d27576b97)

<h2>Кросс-валидация</h2>
# Функция для кросс-валидации
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Изменяем знак, так как scores отрицательные
    print(f'Кросс-валидация - MSE: {mse_scores.mean()}')

# Применение кросс-валидации для всех моделей
models = [model_lr, model_dt, model_gb, model_nn]
for model in models:
    cross_validate_model(model, X, y)
![image](https://github.com/user-attachments/assets/63b7e156-18a9-40d8-a822-c60cf573c7ea)

