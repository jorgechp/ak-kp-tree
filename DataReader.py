import torch
from sklearn.metrics import f1_score

class DataReader():

    def __init__(self):
        self._ak_token_dict = dict()
        self._token_ak_dict = dict()
        self._kp_token_dict = dict()
        self._token_kp_dict = dict()
        self._num_of_ak_words = 0
        self._num_of_kp_words = 0
        self._num_of_lines = 0

    def generate_instance(self, words):
        instance = torch.zeros([self._num_of_ak_words])

        for word in words:
            index = self._ak_token_dict[word]
            instance[index] = 1
        return instance

    def getKp(self, kp_token):
        return self._token_kp_dict[kp_token]


    def generate_data(self,folder, ak_file, kp_file):
        with open(folder+"/"+ak_file, "r") as f:
            input_lines = f.readlines()
            input_lines = [line.strip().split(",") for line in input_lines]

        with open(folder+"/"+kp_file, "r") as f:
            output_lines = f.readlines()
            output_lines = [line.strip().split(",") for line in output_lines]

        self._num_of_lines = len(input_lines)

        ak_set, kp_set = set(), set()
        for line in input_lines:
            ak_set.update(set(line))
        for line in output_lines:
            kp_set.update(set(line))

        self._num_of_ak_words = len(ak_set)
        self._num_of_kp_words = len(kp_set)

        for index, word in enumerate(ak_set):
            self._ak_token_dict[word] = index
            self._token_ak_dict[index] = word

        for index, word in enumerate(kp_set):
            self._kp_token_dict[word] = index
            self._token_kp_dict[index] = word

        train_data = torch.zeros([self._num_of_lines, self._num_of_ak_words]).type(torch.FloatTensor)
        test_data = torch.zeros([self._num_of_lines, self._num_of_kp_words]).type(torch.FloatTensor)

        for num_line, line in enumerate(input_lines):
            for word in line:
                index_of_word = self._ak_token_dict[word]
                train_data[num_line, index_of_word] = 1

        for num_line, line in enumerate(output_lines):
            for word in line:
                index_of_word = self._kp_token_dict[word]
                test_data[num_line, index_of_word] = 1

        return train_data, test_data, len(ak_set), len(kp_set)


    def compute_accuracy(self, y_pred, y_real):
        pred_array = torch.zeros([self._num_of_lines,self._num_of_kp_words])
        for line, index in enumerate(y_pred[1].tolist()):
            pred_array[line, index] = 1

        f1 = f1_score(y_real, pred_array, average='macro')
        return f1






