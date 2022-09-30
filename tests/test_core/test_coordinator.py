import unittest
import random
from bisect import bisect
import numpy as np

import torch

from fedlab.core.coordinator import Coordinator


# @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CoordinatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
        
    def setUp(self):
        # assume there are 10 processes: 
        # rank=0 is server
        # rank=1-10 represents client (9 serial client trainers)
        world_size = 10  
        # initialize client number for each client process
        self.rank_client_id_map = dict()
        for rank in range(1, world_size):
            self.rank_client_id_map[rank] = random.randint(3, 50)  # each client trainer represents a random number of clients

        self.num_clients = int(sum(self.rank_client_id_map.values()))  # total number of clients
        self.local_coordinator = Coordinator(self.rank_client_id_map, 'LOCAL')
        self.global_coordinator = Coordinator(self.rank_client_id_map, 'GLOBAL')

        num_clients_list = [v for k,v in sorted(self.rank_client_id_map.items())]
        num_clients_cumsum = np.cumsum(num_clients_list).tolist()
        num_clients_cumsum.insert(0, 0)
        self.num_clients_cumsum = num_clients_cumsum

    def _map_id_new(self, cid, mode):
        # different method to map global client id to (rank, local_id)
        # can be faster than original map_id function
        rank = bisect(self.num_clients_cumsum, cid)
        local_id = cid - self.num_clients_cumsum[rank-1]
        global_id = cid
        ret_id = local_id if mode == 'LOCAL' else global_id
        return (rank, ret_id)

    def _map_id_list_new(self, id_list, mode):
        map_dict = {}
        for id in id_list:
            rank, id = self._map_id_new(id, mode)
            if rank in map_dict.keys():
                map_dict[rank].append(id)
            else:
                map_dict[rank] = [id]
        return map_dict

    def test_init_local(self):
        self.assertEqual(self.local_coordinator.mode, 'LOCAL')

    def test_init_global(self):
        self.assertEqual(self.global_coordinator.mode, 'GLOBAL')

    def test_map_id_local(self):
        fail_flag = False
        for cid in range(self.num_clients):
            orig_res = self.local_coordinator.map_id(cid)
            new_res = self._map_id_new(cid, 'LOCAL')
            if orig_res[0] != new_res[0] or orig_res[1] != new_res[1]:
                fail_flag = True
                break
        self.assertFalse(fail_flag)

    def test_map_id_global(self):
        fail_flag = False
        for cid in range(self.num_clients):
            orig_res = self.global_coordinator.map_id(cid)
            new_res = self._map_id_new(cid, 'GLOBAL')
            if orig_res[0] != new_res[0] or orig_res[1] != new_res[1]:
                fail_flag = True
                break
        self.assertFalse(fail_flag)

    def test_map_id_list_local(self):
        random_cids = random.sample(list(range(self.num_clients)), 10)
        new_res = self._map_id_list_new(random_cids, 'LOCAL')
        orig_res = self.local_coordinator.map_id_list(random_cids)
        self.assertDictEqual(new_res, orig_res)

    def test_map_id_list_global(self):
        random_cids = random.sample(list(range(self.num_clients)), 10)
        new_res = self._map_id_list_new(random_cids, 'GLOBAL')
        orig_res = self.global_coordinator.map_id_list(random_cids)
        self.assertDictEqual(new_res, orig_res)

    def test_switch(self):
        self._switch_global_to_local()
        self._switch_local_to_global()
        self._switch_invalid()
    
    def _switch_local_to_global(self):
        coor = Coordinator(self.rank_client_id_map, 'LOCAL')
        self.assertEqual(coor.mode, 'LOCAL')
        coor.switch()
        self.assertEqual(coor.mode, 'GLOBAL')

    def _switch_global_to_local(self):
        coor = Coordinator(self.rank_client_id_map, 'GLOBAL')
        self.assertEqual(coor.mode, 'GLOBAL')
        coor.switch()
        self.assertEqual(coor.mode, 'LOCAL')

    def _switch_invalid(self):
        coor = Coordinator(self.rank_client_id_map, 'OTHER')
        self.assertEqual(coor.mode, 'OTHER')
        with self.assertRaises(ValueError):
            coor.switch()

    def test_total_local(self):
        # check LOCAL mode
        self.assertIsInstance(self.local_coordinator.total, int)
        self.assertEqual(self.local_coordinator.total, self.num_clients)
    
    def test_total_global(self):
        # check GLOBAL mode
        self.assertIsInstance(self.global_coordinator.total, int)
        self.assertEqual(self.global_coordinator.total, self.num_clients)

    def test_call_local(self):
        # single int
        single_cid = 3
        res1 = self.local_coordinator(single_cid)
        res2 = self.local_coordinator.map_id(single_cid)
        self.assertEqual(res1, res2)
        # list of int
        list_cid = [3, 4, 5]
        res3 = self.local_coordinator(list_cid)
        res4 = self.local_coordinator.map_id_list(list_cid)
        self.assertDictEqual(res3, res4)

    def test_call_global(self):
        # single int
        single_cid = 3
        res1 = self.global_coordinator(single_cid)
        res2 = self.global_coordinator.map_id(single_cid)
        self.assertEqual(res1, res2)
        # list of int
        list_cid = [3, 4, 5]
        res3 = self.global_coordinator(list_cid)
        res4 = self.global_coordinator.map_id_list(list_cid)
        self.assertDictEqual(res3, res4)

    def test_str_local(self):
        local_str = str(self.local_coordinator)
        local_str_new = self._make_str(self.local_coordinator)
        self.assertEqual(local_str, local_str_new)

    def test_str_global(self):
        global_str = str(self.global_coordinator)
        global_str_new = self._make_str(self.global_coordinator)
        self.assertEqual(global_str, global_str_new)

    def _make_str(self, coor):
        return "Coordinator map information: {} \nMap mode: {} \nTotal: {}".format(
            coor.map, coor.mode, coor.total)
