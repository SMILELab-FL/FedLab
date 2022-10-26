# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .client.trainer import SerialClientTrainer
from .server.handler import ServerHandler

class StandalonePipeline(object):
    def __init__(self, handler: ServerHandler, trainer: SerialClientTrainer):
        """Perform standalone simulation process.

        Args:
            handler (ServerHandler): _description_
            trainer (SerialClientTrainer): _description_
        """
        self.handler = handler
        self.trainer = trainer

        # initialization
        self.handler.num_clients = self.trainer.num_clients

    def main(self):
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()
            # self.handler.evaluate()

    def evaluate(self):
        print("This is a example implementation. Please read the source code at fedlab.core.standalone.")
