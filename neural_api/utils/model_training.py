import json
import os
from joblib import dump
import random
import time
from neural_api.libs.json_lib import read_json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import sklearn
import sklearn.ensemble
from loguru import logger
from pathlib import Path

path = Path(__file__).resolve().parent.parent

MODEL_PATH = f"{path}\\work_data\\models"

logger.add(f"{path}\\utils\\logs\\log.log", rotation="1 MB", compression="zip")

START_TIME = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
START_TIME2 = time.ctime()

config = read_json(f'{path}\\utils\\config.json')

C_ST = config['C_ST']
MAX_ITER = config['MAX_ITER']
TARGET_SCORE = config['TARGET_SCORE']
GOOD_SCORE = config['GOOD_SCORE']
TEST_SIZE = config['TEST_SIZE']

with open(f'{path}\\work_data\\nlp_intent_data.json', 'r',
          encoding='utf-8') as f:
    BOT_CONFIG = json.load(f)

logger.success('nlp_data is loaded')
settings_view = f'C: {C_ST} | MAX_ITER: {MAX_ITER} | TARGET_SCORE: {TARGET_SCORE} | GOOD_SCORE: {GOOD_SCORE} | TEST_SIZE: {TEST_SIZE} | START TIME: {START_TIME2}'
logger.debug(settings_view)

logger.debug(f'Start training: {time.ctime()}')
time.sleep(0.05)
print('-' * 100)

corpus = []
y = []
for intent in BOT_CONFIG.keys():  # Заполнение корпуса
    for example in BOT_CONFIG[intent]['intent']:  # Цикл по примерам
        corpus.append(example)  # Добавление примера в корпус
        y.append(intent)  # Добавление ответа в корпус


def save_clsv(cls, vectorizer, path):
    try:
        with open(f'{path}\\ml_model.joblib', 'wb') as f:
            dump(cls, f)
        with open(f'{path}\\vectorizer.joblib', 'wb') as f:
            dump(vectorizer, f)
        logger.success('Model data is saved')
    except Exception as e:
        logger.error('Model data is not saved:', e)


def save_train_data(x_tests, y_tests, path):
    try:
        with open(f'{path}\\tests\\x_test.joblib',
                  'wb') as f:
            dump(x_tests, f)
        with open(f'{path}\\tests\\y_test.joblib',
                  'wb') as f:
            dump(y_tests, f)
        logger.success('Train data is saved')
    except Exception as e:
        logger.error('Train data is not saved:', e)


def save_model(name: str, cls, vectorizer, X_test, y_test, train_time) -> None:
    """
    Он сохраняет модель, векторизатор и тестовые данные в папку модели.

    :param name: название модели
    :param cls: классификатор
    :param vectorizer: векторизатор, используемый для преобразования текста в вектор
    :param X_test: тестовые данные
    :param y_test: метки набора тестов
    """

    def saving(name1, cls1, vectorizer1, X_test1, y_test1, train_time1):
        os.mkdir(f"{MODEL_PATH}\\{name1}")
        with open(f'{MODEL_PATH}\\{name1}\\model_data.txt', 'w') as model_data:
            text = f'Name: {name1}\n' \
                   f'Creation time: {time.ctime()}\n' \
                   f'Train time: {train_time1}\n' \
                   f'C: {C_ST}\n' \
                   f'MAX_ITER: {MAX_ITER}\n' \
                   f'TARGET_SCORE: {TARGET_SCORE}\n' \
                   f'GOOD_SCORE: {GOOD_SCORE}\n' \
                   f'SCORE: {cls1.score(X_test1, y_test1)}\n' \
                   f'TEST_SIZE: {TEST_SIZE}'
            model_data.write(text)
        os.mkdir(f"{MODEL_PATH}\\{name1}\\tests")
        save_clsv(cls1, vectorizer1, f"{MODEL_PATH}\\{name1}")
        save_train_data(X_test1, y_test1, f"{MODEL_PATH}\\{name1}")
        logger.success('Model is completely saved')

    try:
        saving(name, cls, vectorizer, X_test, y_test, train_time)
    except Exception as e:
        try:
            logger.error('Model is not saved')
            logger.debug('Trying to save model with another name...')
            name = f'{name}_{str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))}_{random.randint(0, 100)}'
            saving(name, cls, vectorizer, X_test, y_test, train_time)
        except Exception as e:
            logger.error('Failed to save the model:', e)


def to_fixed(num_obj: int, digits: int = 4) -> str:
    """
    Функция принимает число и возвращает представление этого числа с заданным количеством цифр после запятой.

    :param num_obj: Число, которое вы хотите преобразовать в число с фиксированной точкой
    :type num_obj: int
    :param digits: Количество цифр после запятой, defaults to 4
    :type digits: int (optional)
    :return: Число с указанным количеством цифр после запятой
    """
    return f"{num_obj:.{digits}f}"


def draw_graph(data: np.ndarray, good_models: int, scores_v: str) -> None:
    """
    Функция рисует график обучения моделей

    :param data: np.ndarray - массив оценок
    :type data: np.ndarray
    :param good_models: количество моделей, получивших оценку > GOOD_SCORE
    :type good_models: int
    :param scores_v: str - имя оптимизируемого счета
    :type scores_v: str
    """
    fig = plt.figure()
    plt.ylabel('Score')
    plt.xlabel('Iteration')
    title = f'\nGood models: {good_models} | {scores_v}'
    plt.title(settings_view + title, fontsize=7)

    plt.ylim(np.min(data), 1)
    plt.xlim(0, data.size)
    plt.plot(data)

    plt.show()
    fig.savefig(f'{path}\\work_data\\graphics_data\\scores_{START_TIME}.png')
    plt.close()


def train() -> tuple:
    """
    Он принимает корпус и список меток, разбивает корпус на обучающий и тестовый наборы, создает векторизатор, обучает
    векторизатор, создает классификатор, обучает классификатор и возвращает классификатор, оценку, тестовый набор, тестовые
    метки и векторизатор
    :return: clf, score, X_test, y_test, vectorizer
    """
    t1 = time.time()
    corpus_train, corpus_test, y_train, y_test = sklearn.model_selection.train_test_split(corpus, y,
                                                                                          test_size=TEST_SIZE,
                                                                                          )
    # Разделение корпуса на обучающий и тестовый
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(2, 4),
                                                                 analyzer='char_wb')  # Создание векторизатора
    X_train = vectorizer.fit_transform(corpus_train)  # Обучение векторизатора
    X_test = vectorizer.transform(corpus_test)  # Тестирование векторизатора
    clf = sklearn.linear_model.LogisticRegression(C=C_ST, max_iter=MAX_ITER, n_jobs=-1,
                                                  class_weight='balanced')  # Создание модели

    clf.fit(X_train, y_train)  # Обучение модели

    score = clf.score(X_test, y_test)  # Тестирование модели
    logger.info(f'model is trained: {clf.score(X_test, y_test)}')
    training_time = time.time() - t1

    return clf, score, X_test, y_test, vectorizer, training_time


def main() -> None:
    """
    Он обучает модель, проверяет, достаточно ли она хороша, и если да, то сохраняет ее.
    """
    scores = []
    good_models = 0
    scores = np.array(scores)
    while True:
        calc_time1 = time.time()
        clf, score, X_test, y_test, vectorizer, train_time = train()
        if score >= TARGET_SCORE:
            save_model(str(score)[2:], clf, vectorizer, X_test, y_test, train_time)
            good_models += 1
        elif score > GOOD_SCORE:
            good_models += 1
            logger.debug('Model is good, but not enough')
        scores = np.append(scores, score)
        score_view = f'Average score: {to_fixed(np.mean(scores))} | Max score: {to_fixed(np.max(scores))} | Min score: {to_fixed(np.min(scores))}'
        logger.info(score_view)
        logger.info(f'Good models: {good_models}')
        draw_graph(scores, good_models, score_view)
        logger.info(f'Training time(sec): {train_time} | Calc time: {(time.time() - calc_time1) - train_time}')
        logger.info('-' * 44)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)

