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



from .server import SyncServerHandler


class qFedAvgServerHandler(SyncServerHandler):
    """_summary_

    Args:
        SyncServerHandler (_type_): _description_
    """
    def global_update(self, buffer):
        deltas = [ele[0] for ele in buffer]
        hks = [ele[1] for ele in buffer]

        hk = sum(hks)
        updates = sum([delta/hk for delta in deltas])
        model_parameters = self.model_parameters - updates

        self.set_model(model_parameters)