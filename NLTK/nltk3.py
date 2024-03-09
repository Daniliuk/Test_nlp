import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import time

# Загрузка NLTK ресурсов
nltk.download('punkt')

# Обучение модели Word2Vec на некоторых текстовых данных
# Здесь предполагается, что у вас есть некоторые текстовые данные в переменной "text_data"
text_data = [
    "Natural language processing (NLP) is a field of computer science.",
    "Machine learning (ML) is the scientific study of algorithms.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines."
]

# Предварительная обработка текстов и токенизация
tokenized_data = [word_tokenize(sentence.lower()) for sentence in text_data]

# Обучение модели Word2Vec
start_time_training = time.time()
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
end_time_training = time.time()

#Загружаем файл с текстом для обработки
with open('10000_lexem.txt') as file_object:
    test_text = file_object.read()

# Токенизация тестового текста
tokenized_test_text = word_tokenize(test_text.lower())


# Выполнение векторного представление текста и подсчет среднего времени выполнения
delta_time = 0
n = 10
while(n>0):

    start_time = time.time()
    word_vectors = [model.wv[word] for word in tokenized_test_text if word in model.wv]
    end_time = time.time()
    execution_time = end_time - start_time
    delta_time = delta_time + execution_time

    n = n - 1

print("Testing Time:", delta_time / 10, "seconds")