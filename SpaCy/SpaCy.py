import time
import spacy

#Загружаем файл с текстом для обработки
with open('10000_lexem.txt') as file_object:
    test_text = file_object.read()

# Загрузка модели языка
nlp = spacy.load("en_core_web_md")  # Загрузка средней модели, включающей векторные представления слов

# Выполнение векторного представление текста и подсчет среднего времени выполнения
delta_time = 0
n = 10
while(n>0):

    start_time = time.time()
    doc = nlp(test_text)
    end_time = time.time()
    execution_time = end_time - start_time
    delta_time = delta_time + execution_time

    n = n - 1

print("Testing Time:", delta_time / 10, "seconds")
