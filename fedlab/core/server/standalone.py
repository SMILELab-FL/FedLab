from ..model_maintainer import ModelMaintainer


class StandaloneServer(ModelMaintainer):
    def __init__(self, model, cuda) -> None:
        super().__init__(model, cuda)

    def main_loop(self):
        pass