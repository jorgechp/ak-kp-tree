import pandas as pd

from AbstractPredictor import AbstractPredictor


class ProbabilityPredictor(AbstractPredictor):

    def _compute_occurrence_matrix(self, ak_lines, ak_set, kp_lines, kp_set):
        self._occurrence_matrix = pd.DataFrame(0, index=kp_set, columns=ak_set, dtype=int)

        for ak_doc, kp_doc in zip(ak_lines, kp_lines):
            for kp_word in kp_doc:
                for ak_word in ak_doc:
                    self._occurrence_matrix[ak_word][kp_word] += 1

    def _compute_frequency_matrix(self):
        self._frequency_matrix = self._occurrence_matrix.div(self._occurrence_matrix.sum(axis=1), axis=0)

    def compute_kp_scores(self, ak_set):
        filtered_frequency_matrix = self._frequency_matrix[ak_set]
        return filtered_frequency_matrix.sum(axis=1)



