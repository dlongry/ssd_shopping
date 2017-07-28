import Main
import time

class QueueChecker:
    def __init__(self, context):
    # def __init__(self, context=Main()):
        self.context = context

    def is_go(self):
        if len(self.context.redis_connection_img_queue.keys()) is not 0:
            return True
        else:
            print("waiting for queue")
            time.sleep(1)
