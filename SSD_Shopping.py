from Main import Main
import time


def main():
    _main_obj = Main()

    while True:
        t = time.time()
        _main_obj.execute_and_update_state()
        # _main_obj.render()
        #print('fps=%f' % (1/(time.time()-t)))
    pass


if __name__ == '__main__':
    main()
