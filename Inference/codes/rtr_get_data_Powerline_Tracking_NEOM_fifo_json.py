import json, os, sys
from stat import *

if __name__ == "__main__":
    # Make the named pipe and poll for new messages.
    FIFO = "Powerline_Tracking.json"
    try:
        os.mkfifo(FIFO)
    except:
        pass
    if S_ISFIFO(os.stat(FIFO).st_mode):
        print('Exist Powerline_Tracking.json')
        pass
    else:
        print('Created Powerline_Tracking.json')
        os.mkfifo(FIFO)

    try:
        while True:
            with open(FIFO) as fifo:
                json_data = json.load(fifo)
                print(json_data)
    except KeyboardInterrupt:
        sys.exit()