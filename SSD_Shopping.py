from Main import Main
import time
import os
import signal


def main():
    _main_obj = Main()

    while True:
        t = time.time()
        _main_obj.execute_and_update_state()
        # _main_obj.render()
        #print('fps=%f' % (1/(time.time()-t)))
    pass

def signal_handler(signal, frame):
    print('kill')
    os.system('kill -9 %d' % os.getpid())

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
