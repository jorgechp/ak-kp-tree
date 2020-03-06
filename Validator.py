import statistics
import random


from AbstractPredictor import AbstractPredictor


class Validator(object):

    def _load_from_file(self, file_path):
        f_keywords = open(file_path, "r")
        keywords_lines = [line.rstrip('\r\n').split(',') for line in f_keywords.readlines()]
        f_keywords.close()

        return keywords_lines

    def _generate_from_file(self, ak_file_path, kp_file_path):
        self._ak_lines = self._load_from_file(ak_file_path)
        self._kp_lines = self._load_from_file(kp_file_path)

    def load_validator(self,ak_file_path: str, kp_file_path: str):
        self._generate_from_file(ak_file_path,kp_file_path)

    def split_training_test(self, training_rate = 0.7):
        total_element_number = len(self._ak_lines)
        training_len = int(total_element_number * training_rate)
        index_number_list = list(range(0,total_element_number))
        random.shuffle(index_number_list)

        ak_shuffled_lines = list()
        kp_shuffled_lines = list()

        for index in index_number_list:
            ak_shuffled_lines.append(self._ak_lines[index])
            kp_shuffled_lines.append(self._kp_lines[index])

        self._ak_training_lines = ak_shuffled_lines[:training_len]
        self._kp_training_lines = kp_shuffled_lines[:training_len]
        self._ak_test_lines = ak_shuffled_lines[training_len:]
        self._kp_test_lines = kp_shuffled_lines[training_len:]

    def test_battery(self, predictor: AbstractPredictor, ak_lines, kp_lines, energy=0.5):
        recall_scores = []
        false_negative_rate_scores = []

        for ak_input, kp_output in zip(ak_lines, kp_lines):
            kp_output_to_validate = predictor.compute_kp_set(ak_set = set(ak_input), energy=energy)
            kp_output_set = set(kp_output)
            if len(kp_output_to_validate) == 0:
                print("hola")
            true_positives = len(kp_output_to_validate.intersection(kp_output_set))
            false_negatives = len(kp_output_set.difference(kp_output_to_validate))
            recall = float(true_positives) / (true_positives + false_negatives)
            false_negative_rate = float(false_negatives) / (false_negatives + true_positives)

            recall_scores.append(recall)
            false_negative_rate_scores.append(false_negative_rate)

        return statistics.mean(recall_scores), statistics.mean(false_negative_rate_scores)

    def validate(self, predictor, energy = 0.5):
        predictor_instance = predictor()
        predictor_instance.generate_from_lines(self._ak_training_lines, self._kp_training_lines)

        training_results = self.test_battery(predictor_instance, self._ak_training_lines, self._kp_training_lines, energy = energy)
        test_results = self.test_battery(predictor_instance, self._ak_test_lines, self._kp_test_lines , energy = energy)

        return training_results, test_results
