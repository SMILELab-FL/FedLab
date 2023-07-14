from abc import abstractmethod
from typing import Any


class FedBoardDelegate:
    def __init__(self):
        pass

    @abstractmethod
    def read_client_label(self, client_id: str, client_rank: str, type: str) -> list[Any]:
        """

        Args:
            client_id: which client
            type: usually 'train', 'test' and 'val'

        Returns:
            list of client labels

        """
        pass

    @abstractmethod
    def sample_client_data(self, client_id: str, client_rank: str, type: str, amount: int) -> list[tuple[Any, Any]]:
        pass
