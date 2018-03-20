import re
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes


topics, contents = [], []
for line in open('news/news_train.txt', encoding='utf8'):
    # считываем метку и содержание вместе с заголовком через символ табуляции
    topic, content = line.split('\t', maxsplit=1)

    # выкидываем все, кроме букв и цифр, и склеиваем вместе
    data = re.sub('\W', ' ', content).split()
    content = ' '.join(data)

    # добавляем в списки метку и текст
    topics.append(topic)
    contents.append(content)

train_topics = topics
train_contents = contents

# чтение тестовых данных из файла
test_contents = []
for line in open('news/news_test.txt', encoding='utf8'):
    data = re.sub('\W', ' ', line).split()
    test_content = ' '.join(data)
    test_contents.append(test_content)

# "учим" векторизатор
vectorizer = TfidfVectorizer(max_features=20000, norm='l1')
vectorizer.fit(train_contents)

# векторизация данных
train_vectorized_contents = vectorizer.transform(train_contents)
test_vectorized_contents = vectorizer.transform(test_contents)

# классификатор
clf = naive_bayes.MultinomialNB(alpha=0.0001, fit_prior=False)
#  тренируем классификатор
clf.fit(train_vectorized_contents, train_topics)

# предсказание
predicted_topics = clf.predict(test_vectorized_contents)

# вывод результата в файл
with open('news/result.txt', 'w+', encoding='utf8') as fout:
    fout.write('\n'.join(predicted_topics))
