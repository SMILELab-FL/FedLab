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
        2. client send local group info (the number of client simulating in local) to server.
        4. server receive all info and init a server Coordinator.

    Args:
        setup_dict (dict): A dict like {rank:client_num ...}, representing the map relation between process rank and client id.
        mode (str, optional): “GLOBAL” and "LOCAL". Coordinator will map client id to (rank, global id) or (rank, local id) according to mode. For example, client id 51 is in a machine which has 1 manager and serial trainer simulating 10 clients. LOCAL id means the index of its 10 clients. Therefore, global id 51 will be mapped into local id 1 (depending on setting).
    """
    def __init__(self, setup_dict, mode='LOCAL') -> None:
        self.map = setup_dict
        self.mode = mode

    def map_id(self, id):
        """a map function from client id to (rank,local id)
        
        Args:
            id (int): client id

        Returns:
            rank, id : rank in distributed group and local id.
        """
        m_id = id
        for rank, num in self.map.items():
            if m_id >= num:
                m_id -= num
            else:
                local_id = m_id
                global_id = id
                ret_id = local_id if self.mode == 'LOCAL' else global_id
                return rank, ret_id

    def map_id_list(self, id_list):
        """a map function from id_list to dict{rank:local id}

            This can be very useful in Scale modules.

        Args:
            id_list (list(int)): a list of client id.

        Returns:
            map_dict (dict): contains process rank and its relative local client ids.
        """
        map_dict = {}
        for id in id_list:
            rank, id = self.map_id(id)
            if rank in map_dict.keys():
                map_dict[rank].append(id)
            else:
                map_dict[rank] = [id]
        return map_dict

    def switch(self):
        if self.mode == 'GLOBAL':
            self.mode = 'LOCAL'
        elif self.mode == 'LOCAL':
            self.mode = 'GLOBAL'
        else:
            raise ValueError("Invalid Map Mode {}".format(self.mode))

    @property
    def total(self):
        return int(sum(self.map.values()))

    def __str__(self) -> str:
        return "Coordinator map information: {} \nMap mode: {} \nTotal: {}".format(
            self.map, self.mode, self.total)

    def __call__(self, info, *args, **kwds):
        if isinstance(info, int):
            return self.map_id(info)
        if isinstance(info, list):
            return self.map_id_list(info)