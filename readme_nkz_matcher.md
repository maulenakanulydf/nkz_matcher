# Пошаговая инструкция по классификации текстов

Этот проект поможет вам научиться находить схожие тексты, используя два типа моделей: FastText и Sentence Transformers. Мы будем работать с текстами, например, с описаниями профессий, и искать самые похожие на введенную профессию в контексте Национального классификатора занятости (НКЗ).

Представьте, что вам нужно найти схожие записи из НКЗ. Например, если вы введете текст «программист инженер », наша система будет искать профессии из НКЗ, которые наиболее схожи с этим запросом.

## Шаг 1. Установка нужных библиотек

Для начала нам нужно установить несколько программ, которые помогут работать с текстами. Это можно сделать с помощью команд, которые нужно ввести в командной строке или Google collab. Введите следующие команды:

```bash
pip install fasttext faiss-cpu sentence-transformers nltk
```

Вот что они делают:

- **fasttext** — помогает работать с текстами и находит смысл слов.
- **faiss-cpu** — помогает быстро искать похожие тексты.
- **sentence-transformers** — помогает преобразовывать текст в числа (эмбеддинги), которые можно сравнивать.
- **nltk** — библиотека для обработки текста, например, для разделения текста на слова.

## Шаг 2. Загружаем наши данные

У нас есть таблица с текстами. В этой таблице есть два столбца: **CODE** и **NAME_RU2**. В первом столбце код вакансии, а во втором — описание вакансии.

Загрузим данные из файла, чтобы дальше работать с ними.

```python
import pandas as pd

# Загружаем таблицу
data = pd.read_csv('drive/MyDrive/data_set/filtered_nkz (3).csv')
df = data[['CODE', 'NAME_RU2']]
df['NAME_RU2'].to_list()
```

Этот код загрузит данные и оставит только два столбца: **CODE** и **NAME_RU2**. В столбце **NAME_RU2** находятся тексты вакансий.

## Шаг 3. Очищаем текст

Чтобы компьютер понял текст, нужно его немного «почистить». Мы переведем все буквы в нижний регистр, уберем лишние символы и разобьем текст на слова. Этот процесс называется **токенизацией**.

```python
import nltk
import re
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Скачиваем необходимые данные для токенизации

def preprocess_text(text):
    text = text.lower()  # Преобразуем в нижний регистр
    text = re.sub(r'[^а-яё\s]', '', text)  # Убираем все ненужные символы
    tokens = word_tokenize(text, language='russian')  # Разбиваем текст на слова
    return tokens
```

Этот код очищает текст: он делает все буквы строчными, убирает лишние символы (например, запятые и точки) и разбивает текст на слова.

## Шаг 4. Загружаем и используем модели

Теперь, когда наш текст готов, мы можем преобразовать его в векторы (множество чисел), чтобы искать похожие тексты. Для этого мы будем использовать две модели: **FastText** и **Sentence Transformers**.

### 1. **FastText** — это модель, которая позволяет преобразовать каждое слово в вектор (набор чисел).

```python
import fasttext

# Загружаем предобученную модель FastText
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz
!gunzip cc.ru.300.bin.gz  # Разархивируем файл
ft_model = fasttext.load_model('cc.ru.300.bin')  # Загружаем модель
```

### 2. **Sentence Transformers** — эта модель создает вектор для всего предложения. Это важно, потому что мы не просто хотим найти схожие слова, а именно схожие тексты.

```python
from sentence_transformers import SentenceTransformer

# Загружаем модель для преобразования предложений в векторы
st_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
```

## Шаг 5. Создаем векторы для текстов

Теперь мы можем превратить каждый текст в набор чисел (вектор), который будем использовать для поиска похожих текстов.

```python
import numpy as np

def get_sentence_embedding_fasttext(tokens):
    word_vectors = [ft_model.get_word_vector(token) for token in tokens]
    if len(word_vectors) == 0:
        return np.zeros(ft_model.get_dimension())  # Если слов нет, возвращаем вектор из нулей
    return np.mean(word_vectors, axis=0)  # Берем среднее значение для всех слов в предложении

def get_combined_embedding(text):
    st_embedding = st_model.encode([text])[0]  # Вектор для всего предложения
    tokens = preprocess_text(text)  # Разбиваем текст на слова
    ft_embedding = get_sentence_embedding_fasttext(tokens)  # Вектор для слов
    return np.concatenate((st_embedding, ft_embedding))  # Объединяем оба вектора
```

Теперь каждый текст преобразуется в длинный вектор чисел, который представляет его смысл.

## Шаг 6. Поиск похожих текстов

Мы будем использовать **FAISS**, чтобы найти похожие тексты. FAISS помогает быстро искать схожие векторы среди множества других векторов.

```python
import faiss

# Генерация векторов для всех текстов в таблице
embeddings = []
for text in df['NAME_RU2']:
    embedding = get_combined_embedding(text)
    embeddings.append(embedding)

embeddings = np.vstack(embeddings)  # Создаем матрицу из всех векторов
dimension = embeddings.shape[1]  # Определяем размерность векторов

# Нормализуем векторы, чтобы их можно было правильно сравнивать
faiss.normalize_L2(embeddings)

# Создаем индекс для поиска схожих векторов
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
```

Теперь у нас есть индекс, который можно использовать для поиска схожих текстов.

## Шаг 7. Ищем похожие тексты

Теперь мы можем искать тексты, похожие на введенный вами запрос. Например, если вы введете текст «курьер программист инженер водитель экспедитор», система найдет наиболее похожие вакансии.

```python
def find_similar_combined(text, k=5):
    embedding = get_combined_embedding(text)  # Создаем вектор для введенного текста
    faiss.normalize_L2(embedding.reshape(1, -1))  # Нормализуем вектор
    embedding = np.expand_dims(embedding, axis=0)  # Делаем его двумерным для FAISS

    # Ищем похожие векторы
    distances, indices = index.search(embedding, k)

    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            'rank': i+1,
            'NAME_RU2': df.iloc[idx]['NAME_RU2'],
            'CODE': df.iloc[idx]['CODE'],
            'similarity': distances[0][i]
        }
        results.append(result)
    return results

# Пример: поиск схожих вакансий
input_text = 'курьер программист инженер водитель экспедитор'
similar_texts = find_similar_combined(input_text, k=5)

# Выводим результаты
for res in similar_texts:
    print(f"Rank: {res['rank']}")
    print(f"NAME_RU2: {res['NAME_RU2']}")
    print(f"CODE: {res['CODE']}")
    print(f"Similarity: {res['similarity']}\n")
```

После выполнения этого кода вы получите список схожих вакансий с процентом схожести.

---

## Заключение

Теперь вы можете использовать этот процесс для поиска схожих текстов в любых данных. Этот метод позволяет эффективно искать похожие вакансии или другие тексты, что может быть полезно в разных задачах, от автоматической классификации до улучшения поиска по сайту.
