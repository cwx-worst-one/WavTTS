import bytedtos
import pickle


class GetMsg:
    def __init__(self):
        self.tos = bytedtos.Client("lab-speech-engine", "8ZI2BF8IB0JIFP9CRII3")
        self.tos.stream = True

    def get_msg_from_tos(self):
        processor_output_file = self.tos.get_object("processor_output.pkl").raw
        message_deserialization = pickle.load(processor_output_file)
        return message_deserialization
