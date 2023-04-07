from neural_api.nlu.ml_module import nlu_handler
from time import sleep
from dotenv import load_dotenv
import os

load_dotenv()
MODE = os.getenv("MODE")

def main():
    print('-'*50)
    while KeyboardInterrupt:
        mess = input("Enter your message: ")
        if mess == 'exit':
            break
        answer = nlu_handler(mess)
        if answer == 'null':
            continue
        else:
            print(answer)


if __name__ == "__main__":
    try:
        if MODE == "dev":
            sleep(0.1)
            main()
            input("Press Enter to exit...")
        elif MODE == "api":
            import neural_api.api.web_api
    except KeyboardInterrupt:
        exit(0)
