import statistics
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


    def compute_training(self, predictor: AbstractPredictor, energy = 0.5):
        accuracy_scores = []
        precision_scores = []
        for ak_input, kp_output in zip(self._ak_lines, self._kp_lines):
            kp_output_to_validate = predictor.compute_kp_set(ak_input, energy= energy)

            #TODO generar matriz de confusion
            correct_elements = len(kp_output_to_validate.intersection(kp_output))
            total_elements = len(kp_output_to_validate.union(kp_output))
            total_output_elements = len(kp_output)
            accuracy_scores.append(correct_elements/float(total_elements))
            precision_scores.append(correct_elements/float(total_output_elements))
        return statistics.mean(accuracy_scores),statistics.mean(precision_scores)
