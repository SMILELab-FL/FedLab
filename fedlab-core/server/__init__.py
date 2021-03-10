import logging
import torch
import torch.distributed as dist
from abc import ABC 


from torch.multiprocessing import Process
from FLTB_core.utils.messaging import MessageCode, send_message

_LOGGER = logging.getLogger(__name__)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
_LOGGER.addHandler(handler)


class PramsServer(Process):
    """PramsServer

    base class for message listeners, extends pythons threading Thread
    """

    def __init__(self, model):
        """__init__
        :param model: nn.Module to be defined by the user
        """
        #self.model = model
        #self.buff = torch.zeros(model.numel() + 2).cpu()
        super(PramsServer, self).__init__()

    def receive(self, sender, message_code, parameter):
        """receive
        :param sender: rank id of the sender
        :param message_code: Enum code
        :param parameter: the data payload
        """
        raise NotImplementedError()

    def run(self):
        """
        _LOGGER.info("Started Running!")
        print("Started Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for message...")
            print("Polling for message...")
            dist.recv(tensor=self.buff)
            self.receive(int(self.buff[0].item()),
                         MessageCode(self.buff[1].item()),
                         self.buff[2:])
        """

