<h1>Гушшамов Кирилл spotify</h1>
<h2>Импортирую нужные библиотеки</h2>

```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
```

<h2>Читаю файл</h2>

```
df = pd.read_csv('/content/dataset.csv')
df.head()
```

![image](https://github.com/user-attachments/assets/600f3aba-668e-4668-85b6-4a79dd9ff3e5)

<h2>Удаляю track id так как он не особо нужен</h2>

```
del df["track_id"]
df.head()
```

![image](https://github.com/user-attachments/assets/17c00ca7-f2bb-4a7e-bc54-788292b06076)

<h2>Удаляю ещё один не нужный признак</h2>

```
del df["Unnamed: 0"]
```

<h1>Графики:</h1>


<h2>Понять, как распределена популярность треков в нашем наборе данных. Это поможет нам увидеть, есть ли треки с высокой популярностью и как они соотносятся с менее популярными.</h2>

```
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title('Распределение популярности треков')
plt.xlabel('Популярность')
plt.ylabel('Частота')
plt.grid(True)
plt.show()
```

![image](https://github.com/user-attachments/assets/124521ea-d3bb-43aa-8ae6-c651e5df96ea)


<h2>Нужно выяснить, какие признаки имеют сильную корреляцию с целевой переменной (популярностью)</h2>

```
plt.figure(figsize=(12, 8))
numerical_data = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()
```

![image](https://github.com/user-attachments/assets/8f958431-83ba-4b7f-b9e5-a6a59b597825)


<h2>Теперь посмотрим корреляцию в числах</h2>

```
numerical_data = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()
correlation_matrix.corr()['popularity'].sort_values(ascending=False)

```

![image](https://github.com/user-attachments/assets/58513407-40c7-4dd0-8cce-7d2db92ab04d)


<h2><b>Feature Engineering.</b> Корреляция новых колонок с таргетом. Feature Importances. Простая модель.</h2>

```
genre_popularity = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)

# Строим график
plt.figure(figsize=(17, 9))
sns.barplot(x=genre_popularity.index, y=genre_popularity.values, palette='viridis')
plt.title('Средняя популярность по жанрам')
plt.xlabel('Жанр')
plt.ylabel('Средняя популярность')
plt.xticks(rotation=85)
plt.show()
```

![image](https://github.com/user-attachments/assets/75b8b8f4-50d9-4285-acd8-3751f0f48112)


<h2>Делаю категориальный признак</h2>

```
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
df['track_genre_encoded'] = label_encoder.fit_transform(df['track_genre'])

# Проверим результат
print(df[['track_genre', 'track_genre_encoded']])
```

![image](https://github.com/user-attachments/assets/26ba2b73-2a2f-428d-a8a0-144bb84edc6b)

<h3>Дополнительно проверю весь датасет</h3>

```
df
```

![image](https://github.com/user-attachments/assets/d222ae3c-5d2f-4356-b46e-db2071145c79)

<h2>Заметил ещё один некорректный признак, также меняю на категориальный</h2>

```
label_encoder = LabelEncoder()
df['explicit'] = label_encoder.fit_transform(df['explicit'])
df.head()
```

![image](https://github.com/user-attachments/assets/dc60895e-5367-40e5-bb3e-9ba268d1cabc)


<h3>Удаляю лишний столбецб который появился при замене на категориальный</h3>

```
del df["track_genre"]
# again check
df
```

<h2>Предполагаю, что при поиске определенные названия будут попадаться чаще <b>upd: совсем забыл что есть nan значения) получал очень долго ошиюку не мог понять</b></h2>

```
df.isnull().sum()
```

![image](https://github.com/user-attachments/assets/8008ff3b-4565-4424-ba3e-3a756b5f3dd8)

```
# Удаляю их
df = df.dropna()
df.isnull().sum()
```

![image](https://github.com/user-attachments/assets/bb570252-d59e-470d-9972-a7fb2d60a01f)


<h2>Вернёмся к гипотезе о ключевых словах</h2>
<p>Выдляю список слов триггеров, которые люди могли бы чаще искать в определенных ситуациях</p>

```
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
```

![image](https://github.com/user-attachments/assets/86a44ad6-fffc-4b07-a887-bb9b5d144c02)


<h2>Теперь нужно удалить лишний столбец которы получился при создании нового признака</h2>

```
del df["track_name"]
```

<h2>Теперь тоже самое хочу сделать и с названием альбома</h2>

```
df['album_weight'] = df['album_name'].apply(calculate_weight)

# Проверим результат
df[['album_name', 'album_weight']].head()
```

![image](https://github.com/user-attachments/assets/7815d930-ce06-477e-8c71-21152d948888)


<h2>Также удаляю лишнее</h2>

```
del df["album_name"]
```

<h2>Настало время проверить гипотезу, насколько она была корректна</h2>

```
plt.figure(figsize=(10, 6))
sns.scatterplot(x='album_weight', y='popularity', data=df)

# Добавление заголовка и меток осей
plt.title('Корреляция между метками и популярностью')
plt.xlabel('Кол-во найденныйх слов')
plt.ylabel('Популярность')

# Отображение графика
plt.show()

```
![image](https://github.com/user-attachments/assets/aab64ee0-c295-479b-aa76-34bda24f3887)

<h3> И для название трека такой же график</h3>

```
plt.figure(figsize=(10, 6))
sns.scatterplot(x='track_weight', y='popularity', data=df)

# Добавление заголовка и меток осей
plt.title('Корреляция между метками и популярностью')
plt.xlabel('Кол-во найденныйх слов')
plt.ylabel('Популярность')

# Отображение графика
plt.show()
```

![image](https://github.com/user-attachments/assets/30fa4867-e438-4018-af80-6eb6f269bf4a)

<h2><b>Как видно из графиков выше гипотеза не подвердилась</b></h2>

<h2>Решил проверить ещё одну корреляцию между темпом и популярностью
как можно заметить из графика, ничего полезного нам здесь не достать</h2>

```
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tempo', y='popularity', data=df)

# Добавление заголовка и меток осей
plt.title('Корреляция между темпом и популярностью')
plt.xlabel('Кол-во найденныйх слов')
plt.ylabel('Популярность')

# Отображение графика
plt.show()
```

![image](https://github.com/user-attachments/assets/17df5f56-65a6-4a8c-8ed9-ea2aed97e798)


<h2>Еще одна гипотеза, что если длительность терка меньше, то его популярность больше</h2>

```
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration_ms', y='popularity', data=df)

# Добавление заголовка и меток осей
plt.title('Корреляция между длительностью трека и популярностью')
plt.xlabel('Длительность (мс)')
plt.ylabel('Популярность')

# Отображение графика
plt.show()
```

![image](https://github.com/user-attachments/assets/e853195a-9b54-4619-8a27-d206f8ef8dac)

<h2>Гипотеза подтвердилась и теперь если трек меньшей длины я ему добавляю новый признак</h2>

```
df['short_duration'] = df['duration_ms'] < 50000  # 10000 мс = 1 секунда

# Преобразование булевой переменной в целочисленный формат (0 и 1)
df['short_duration'] = df['short_duration'].astype(int)

# Просмотр первых нескольких строк данных с новой фичей
df[['duration_ms', 'popularity', 'short_duration']].head()
```

<h3>Смотрим сколько таких песен</h3>

```
df[df['short_duration'] > 0]
```

![image](https://github.com/user-attachments/assets/d920c7f9-4c50-4d39-a397-08b71cd888b3)

![image](https://github.com/user-attachments/assets/79cc37c3-0b86-42ac-8b27-24ffcff7cbd4)

<h2>Нормализум значение у длительности уж сильно оно огромное</h2>

```
# Нормализация
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Инициализация MinMaxScaler
min_max_scaler = MinMaxScaler()

# Применение MinMaxScaler к столбцу duration_ms
df["duration_ms"] = min_max_scaler.fit_transform(df[["duration_ms"]])
df.head()
```

![image](https://github.com/user-attachments/assets/954c6820-d931-4622-87f8-d95a30388cd5)


<h2>Вновь получаю ошибку что даннеы не соответсвуют, удаляю имя</h2>

```
del df["artists"]
```

<h2><b>Модели:</b></h2>

<h2>Линейная регрессия</h2>

```
# доп импорты
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
```

```
# подготовка данных
X = df.drop(columns=['popularity'])  # Используем все признаки, кроме 'popularity'
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
<h2>Постоение модели, вывод метрик и построение графиков ошибок</h2>

```
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Обучение модели
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Прогноз и оценка
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Вывод результатов
print(f'Линейная регрессия - MSE: {mse_lr}')
print(f'Линейная регрессия - MAE: {mae_lr}')
print(f'Линейная регрессия - R²: {r2_lr}')

# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Линейная регрессия: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()
```

![image](https://github.com/user-attachments/assets/20ec4b5c-47e6-4b1e-97d4-bb46b1aaa7bc)

```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error

# Создание модели

lin_reg = LinearRegression()

# Обучение модели
lin_reg.fit(X, y)

# Предсказание на данных
y_pred = lin_reg.predict(X)

# Кросс-валидация для получения потерь
# Кросс-валидация для MSE


plt.figure(figsize=(10, 6))

# Построение графика предсказанных значений против истинных значений
plt.scatter(y, y_pred, color='blue', label='Предсказанные значения', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Линия идеального соответствия')

# Настройки графика
plt.title("Сравнение предсказанных и истинных значений")
plt.xlabel("Истинные значения (y)")
plt.ylabel("Предсказанные значения (y_pred)")
plt.grid(True)
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/9a2b7c77-7286-480d-a354-d49e2a293b26)


<h2>Регрессия дерева решений</h2>

```
# Обучение модели
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# Прогноз и оценка
y_pred_dt = model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Вывод метрик
print(f'Дерево решений - MSE: {mse_dt}')
print(f'Дерево решений - MAE: {mae_dt}')
print(f'Дерево решений - R²: {r2_dt}')

# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Дерево решений: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()
```

![image](https://github.com/user-attachments/assets/df38a5a5-d894-4b02-b455-ae1b8cd40d70)


```
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
```

![image](https://github.com/user-attachments/assets/d92bc4db-82a4-4525-a426-400c584d6f8e)



<h2>Градиентный бустинг</h2>

```
# Обучение модели
model_gb = GradientBoostingRegressor(random_state=42)
model_gb.fit(X_train, y_train)

# Прогноз и оценка
y_pred_gb = model_gb.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_gb)
mae_dt = mean_absolute_error(y_test, y_pred_gb)
r2_dt = r2_score(y_test, y_pred_gb)

# Вывод метрик
print(f'Градиентный бустинг - MSE: {mse_dt}')
print(f'Градиентный бустинг - MAE: {mae_dt}')
print(f'Градиентный бустинг - R²: {r2_dt}')

# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Градиентный бустинг: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()
```


![image](https://github.com/user-attachments/assets/39fa9577-c82d-49d2-8125-e4d35f9934bf)




```
# Создание модели градиентного бустинга
gb_reg = GradientBoostingRegressor()

# Кросс-валидация
scores = cross_val_score(gb_reg, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

# Визуализация ошибок
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(mse_scores) + 1), mse_scores, marker='o', label='Градиентный бустинг', color='blue')
plt.title('Ошибки кросс-валидации для градиентного бустинга')
plt.xlabel('Фолд')
plt.ylabel('Среднеквадратичная ошибка (MSE)')
plt.xticks(range(1, len(mse_scores) + 1))
plt.grid(True)
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/e35b7737-3134-407a-9dbe-65f38d88e3c7)



<h2>Нейронная сеть</h2>

```
# Обучение модели
model_nn = MLPRegressor(random_state=42, max_iter=1000)
model_nn.fit(X_train, y_train)

# Прогноз и оценка
y_pred_nn = model_nn.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_nn)
mae_dt = mean_absolute_error(y_test, y_pred_nn)
r2_dt = r2_score(y_test, y_pred_nn)

# Вывод метрик
print(f'Нейронная - MSE: {mse_dt}')
print(f'Нейронная - MAE: {mae_dt}')
print(f'Нейронная - R²: {r2_dt}')
# График ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Нейронная сеть: Предсказанные vs. Реальные значения')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.show()
```

![image](https://github.com/user-attachments/assets/053b426d-b2f0-4975-a879-ad8ec58cc29e)


```
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
```



<h1>Итоговая гистограмма с результатами моделей</h1>

<h2>Подготовка данных</h2>

```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Результаты моделей
models = ['Linear Regression', 'Decision Tree', 'Gradient Boosting', 'Neural Network']

# Вычисляем метрики для каждой модели
mse_scores = [
    mean_squared_error(y_test, y_pred_lr),
    mean_squared_error(y_test, y_pred_dt),
    mean_squared_error(y_test, y_pred_gb),
    mean_squared_error(y_test, y_pred_nn)
]

mae_scores = [
    mean_absolute_error(y_test, y_pred_lr),
    mean_absolute_error(y_test, y_pred_dt),
    mean_absolute_error(y_test, y_pred_gb),
    mean_absolute_error(y_test, y_pred_nn)
]

r2_scores = [
    r2_score(y_test, y_pred_lr),
    r2_score(y_test, y_pred_dt),
    r2_score(y_test, y_pred_gb),
    r2_score(y_test, y_pred_nn)
]
```

<h2>График MSE</h2>

```
# График MSE
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=mse_scores, palette='viridis')
plt.title('Среднеквадратичная ошибка (MSE)')
plt.xlabel('Модель')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.show()
```
