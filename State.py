from enum import Enum

State = Enum('State', ('WaitForQueue',
                       'Start',
                       'EdgeDetection',
                       'BehaviourDetection',
                       'BehaviourDoubleCheck'
                       ))