# Customer By Congjian

class ExRecorder:
    def __init__(self, filename):
        self._title = ["cross_entropy", "accuracy", "top5_accuracy"]
        self._filename = filename
        self._records = []

    def add_log(self, record):
        print(record)
        temp = []
        for key in record:
            temp.append(record[key].detach().item())
        self._records.append(temp)

    def add_log_direct(self, record):
        print(record)
        temp = []
        for key in record:
            temp.append(record[key])
        self._records.append(temp)

    def save_to_file(self):
        with open(self._filename, "w") as f:
            for item in self._records:
                f.write(str(item) + "\n")
