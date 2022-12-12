from neural_api.nlu.ml_module import nlu_handler
from time import sleep
from loguru import logger

def main():
    print('-'*50)
    while KeyboardInterrupt:
        mess = input("Enter your message: ")
        answer = nlu_handler(mess)
        if answer == 'null':
            continue
        else:
            print(answer)


if __name__ == "__main__":
    sleep(0.1)
    main()