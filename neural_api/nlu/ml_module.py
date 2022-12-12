import json
import os
from joblib import load
import random
from pathlib import Path
from loguru import logger

path = Path(__file__).resolve().parent.parent

PATH_TO_MODELS = f'{path}\\work_data\\models'

models_weights = []

for dir_name in list(os.walk(PATH_TO_MODELS))[0][1]:  # Получение списка папок с моделями
    try:
        models_weights.append(int(dir_name))
    except:
        continue

if models_weights:
    MODEL_NAME = max(models_weights)
else:
    logger.error('No available models found.')
    exit(1)

MODEL_PATH = f'{PATH_TO_MODELS}\\{MODEL_NAME}'

with open(f'{path}\\work_data\\nlp_intent_data.json', 'r',
          encoding='utf-8') as f:  # Загрузка конфига
    BOT_CONFIG = json.load(f)
logger.success('NLP Data is loaded')
logger.debug(f'Model path: {MODEL_PATH}')


def clean(text: str) -> str:  # Очистка текста
    clean_text = ''
    for char in text.lower():
        if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
            clean_text = clean_text + char
    return clean_text


with open(f'{MODEL_PATH}\\ml_model.joblib', 'rb') as f, \
        open(f'{MODEL_PATH}\\vectorizer.joblib', 'rb') as f2:  # Открытие модели
    clf = load(f)
    vectorizer = load(f2)

with open(f'{MODEL_PATH}\\tests\\x_test.joblib', 'rb') as f, \
        open(f'{MODEL_PATH}\\tests\\y_test.joblib', 'rb') as f2:
    X_test = load(f)
    y_test = load(f2)

logger.warning(f"Model score: {clf.score(X_test, y_test)}")


def nlu_handler(input_text: str) -> str:
    """
    Принимает текст и возвращает ответ бота

    :param input_text: Текст, который ввел пользователь
    :type input_text: str
    """

    def get_intent_by_model(text):  # Получение ответа по модели
        text = clean(text)  # Очистка текста
        if text != '' and len(text) > 1:  # Если текст не пустой
            return clf.predict(vectorizer.transform([text]))[0]  # Предсказание ответа
        return 'null'

    def nlp(text):
        try:
            intent = get_intent_by_model(text)
            if intent != 'null':
                return random.choice(BOT_CONFIG[intent]['responses'])
            else:
                return intent
        except Exception as e:
            logger.error(f'Error in nlp: {e}')
            return 'null'

    return nlp(input_text)
