import pickle

import torch

from AbstractPredictor import AbstractPredictor
from nn.KP_nn import KP_NN


class NNPredictor(AbstractPredictor):


    def _predict_words(self, data_manager, model, words, k=50, threshold=0.9):
        model.eval()
        instance = data_manager.generate_instance(words)
        predicted_kp = set()
        with torch.no_grad():
            prediction = torch.topk(model(instance.view(1, self._num_of_inputs)), k=k)
            word_indices = prediction[1].tolist()[0]
            word_probabilities = prediction[0].tolist()[0]
            for index, probability in zip(word_indices, word_probabilities):
                if probability > threshold:
                    predicted_kw = data_manager.getKp(index)
                    predicted_kp.add(predicted_kw)
        return predicted_kp


    def prepare(self, model_path, data_path):

        with open(data_path, "rb") as input_file:
            self._data_manager = pickle.load(input_file)

        self._num_of_inputs = self._data_manager._num_of_ak_words
        self._num_of_outputs = self._data_manager._num_of_kp_words
        self._model = KP_NN(self._num_of_inputs, self._num_of_outputs )
        self._model.load_state_dict(torch.load(model_path))


    def compute_kp_scores(self, ak_set, energy = 0.7):
        self._model.eval()

        words_instance = list(ak_set)
        return self._predict_words(self._data_manager, self._model, words_instance, threshold = energy)

    def compute_kp_set(self, ak_set, energy = 0.7):
        return self.compute_kp_scores(ak_set)




