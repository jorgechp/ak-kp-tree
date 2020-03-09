import pandas as pd

from AbstractPredictor import AbstractPredictor


class MeanProbabilityPredictor(AbstractPredictor):


    def _compute_occurrence_matrix(self, ak_lines, ak_set, kp_lines, kp_set):
        self._occurrence_matrix = pd.DataFrame(0, index=kp_set, columns=ak_set, dtype=int)

        self._keyword_ocurrencies = dict()

        for ak_doc in ak_lines:
            for ak_word in ak_doc:
                if ak_word in self._keyword_ocurrencies:
                    self._keyword_ocurrencies[ak_word] += 1
                else:
                    self._keyword_ocurrencies[ak_word] = 1

        for ak_doc, kp_doc in zip(ak_lines, kp_lines):
            for kp_word in kp_doc:
                for ak_word in ak_doc:
                    self._occurrence_matrix[ak_word][kp_word] += 1


    def _compute_frequency_matrix(self):
        self._frequency_matrix = self._occurrence_matrix.copy().astype(float)
        for ak_word in self._frequency_matrix.columns:
            self._frequency_matrix[ak_word] /= self._keyword_ocurrencies[ak_word]


    def compute_kp_scores(self, ak_set):
        if len(ak_set) > 0:
            filtered_frequency_matrix = self._frequency_matrix[ak_set]
            return filtered_frequency_matrix.mean(axis=1)
        else:
            return False




