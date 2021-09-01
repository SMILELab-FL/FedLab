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


class Coordinator(object):
    """Deal with the mapping relation between client id in FL system and process rank in communication.

    Note
        Server Manager creates a Coordinator following:
        1. init network connection.
        2. client actively send local group info to server.
        4. server receive all info and init Coordinator.

    Args:
        setup_dict (dict): A dict like {rank:client_num ...}, representing the map relation between process rank and client id.
    """

    def __init__(self, setup_dict) -> None:
        self.map = setup_dict

    def map_id_list(self, id_list):
        map_dict = {}
        for id in id_list:
            for rank, num in self.map.items():
                if id >= num:
                    id -= num
                else:
                    if rank in map_dict.keys():
                        map_dict[rank].append(id)
                    else:
                        map_dict[rank] = [id]
                    break
        return map_dict

    def __str__(self) -> str:
        return str(self.map)
