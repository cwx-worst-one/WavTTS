import pickle
import numpy as np


class SerializerNode(object):
    def __init__(self, data, header):
        self.data = data
        self.header = header


class Serializer(object):
    def __init__(self):
        self.result = {}
        self.file_path = ""

    def set_file_path(self, path):
        self.file_path = path

    def add_data(self, wav_name, data, **header):
        if len(self.file_path) == 0:
            return
        data = np.array(data, dtype=np.float32)
        node = SerializerNode(data, header)
        if wav_name in self.result:
            self.result[wav_name].append(node)
        else:
            self.result[wav_name] = [node]

    def finish(self):
        if len(self.file_path) == 0:
            return
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.result, f)


serializer = Serializer()
