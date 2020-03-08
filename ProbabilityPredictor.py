import pandas as pd

from AbstractPredictor import AbstractPredictor


class ProbabilityPredictor(AbstractPredictor):

    def _compute_occurrence_matrix(self, ak_lines, ak_set, kp_lines, kp_set):
        self._occurrence_matrix = pd.DataFrame(0, index=kp_set, columns=ak_set, dtype=float)

        for ak_doc, kp_doc in zip(ak_lines, kp_lines):
            for kp_word in kp_doc:
                for ak_word in ak_doc:
                    self._occurrence_matrix[ak_word][kp_word] += 1

    def _compute_frequency_matrix(self):
        sums = self._occurrence_matrix.sum(axis=1)
        for col in self._occurrence_matrix:
            self._occurrence_matrix[col] /= sums
        self._frequency_matrix = self._occurrence_matrix
        # self._frequency_matrix = self._occurrence_matrix.div(self._occurrence_matrix.sum(axis=1), axis=0)

    def compute_kp_scores(self, ak_set):
        existing_keywords = self._frequency_matrix.columns.intersection(ak_set)
        if len(existing_keywords) > 0:
            filtered_frequency_matrix = self._frequency_matrix[existing_keywords]
            return filtered_frequency_matrix.sum(axis=1)
        else:
            return False




