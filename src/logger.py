import datetime
import os

log_folder = "./log"


class Log(object):
    folder: str

    def __init__(self):
        super(Log, self).__init__()
        if not os.path.isdir(log_folder):
            os.mkdir(log_folder)
        self.folder = os.path.join(log_folder, f"log_{datetime.datetime.now()}")
        os.mkdir(self.folder)

    def write(self, name: str, data: str):
        filepath = os.path.join(self.folder, f"{name}.log")
        if not os.path.isfile(filepath):
            with open(filepath, "w") as _:
                pass
        with open(filepath, "a") as fp:
            fp.write(f"{data}\n")

    def write_log(self, name: str, data: str):
        self.write(name, f"{datetime.datetime.now()}: {data}")

# default log
log = Log()