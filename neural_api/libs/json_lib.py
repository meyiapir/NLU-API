import json

def read_json(file_name: str) -> any:
    """
    :param file_name: Имя файла для чтения
    :type file_name: str
    :return: Словарь
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(file_name: str, data: dict, mode: str = 'a') -> None:
    """
    :param file_name: Имя файла для записи
    :type file_name: str
    :param data: Данные для записи в файл
    :type data: dict
    :param mode: «a» означает «добавлять», «w» означает «записывать», defaults to a
    :type mode: str (optional)
    """
    with open(file_name, mode, encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
