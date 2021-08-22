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


class coordinator(object):
    """Deal with the map relation between client id in FL system and rank in communication.

    通过SetUp(), 全局同步client id -> rank的映射
    生成coordinator数据结构

    Procedure:
        1. init group
        2. server -> quest basic info.
        3. rank (1-world_size-1) answer basic info(client num)
        4. server receive info, init Coordinator.

    Args:
        object ([type]): [description]
    """
    def __init__(self, setup_dict) -> None:
        self.map = setup_dict

    def map_id_list(self, id_list):
        map_dict = {}
        for id in id_list:
            for rank, num in self.map:
                if id >= num:
                    id -= num
                else:
                    if rank in map_dict.keys():
                        map_dict[rank].append(id)
                    else:
                        map_dict[rank] = [id]
                    break
        return map_dict
