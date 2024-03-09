from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import time

# Пример текста для обучения модели Word2Vec
training_data = [
    ['gensim', 'is', 'a', 'robust', 'framework'],
    ['for', 'topic', 'modeling', 'and', 'document', 'similarity', 'analysis']
]

# Обучение модели Word2Vec
model = Word2Vec(sentences=training_data, vector_size=100, window=5, min_count=1, workers=4)

#Загружаем файл с текстом для обработки
with open('10000_lexem.txt') as file_object:
    test_text = file_object.read()

# Получение векторного представления для предложения
preprocessed_sentence = simple_preprocess(test_text)


# Выполнение векторного представление текста и подсчет среднего времени выполнения
delta_time = 0
n = 10
while(n>0):

    start_time = time.time()
    vectors = [model.wv[word] for word in preprocessed_sentence if word in model.wv]
    end_time = time.time()
    execution_time = end_time - start_time
    delta_time = delta_time + execution_time

    n = n - 1

print("Testing Time:", delta_time / 10, "seconds")

