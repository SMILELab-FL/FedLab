
import threading

import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Queue, Manager
torch.multiprocessing.set_sharing_strategy("file_system")



def upper(read, write):
    msg = (torch.randn(size=(10,)), torch.randn(size=(5,)))
    print("write", msg)
    write.put(msg)
    
def lower(read, write):
    msg = read.get()
    print("recv")
    print(msg[0])
    print(msg[1])
    

if __name__ == "__main__":
    MQs = [Queue(), Queue()]
    
    p1 = Process(target=upper, args=(MQs[0], MQs[1]))
    p2 = Process(target=lower, args=(MQs[1], MQs[0]))

    p1.start()
    p2.start()

    p1.join()
    p2.join()



