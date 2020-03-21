from abc import abstractmethod

from AbstractPredictor import AbstractPredictor


class AbstractStatisticPredictor(AbstractPredictor):
    def _load_from_file(self, file_path):
        f_keywords = open(file_path, "r")
        keywords_lines = [self._clean(line).rstrip('\r\n').split(',') for line in f_keywords.readlines()]
        f_keywords.close()
        return keywords_lines

    def _generate_set_from_lines(self, lines):
        keyword_set = set()

        for keyword in lines:
            keyword_set.update(keyword)
        return keyword_set

    def compute_existing_keywords(self, ak_set):
        return self._frequency_matrix.columns.intersection(ak_set)

    def generate_from_file(self, ak_file_path, kp_file_path):
        ak_lines = self._load_from_file(ak_file_path)
        kp_lines = self._load_from_file(kp_file_path)

        ak_set = self._generate_set_from_lines(ak_lines)
        kp_set = self._generate_set_from_lines(kp_lines)

        self._compute_occurrence_matrix(ak_lines, ak_set, kp_lines, kp_set)
        self._compute_frequency_matrix()

    def generate_from_lines(self, ak_lines, kp_lines):
        ak_set = self._generate_set_from_lines(ak_lines)
        kp_set = self._generate_set_from_lines(kp_lines)

        self._compute_occurrence_matrix(ak_lines, ak_set, kp_lines, kp_set)
        self._compute_frequency_matrix()

    @abstractmethod
    def _compute_occurrence_matrix(self, ak_lines, ak_set, kp_lines, kp_set):
        pass

    @abstractmethod
    def _compute_frequency_matrix(self):
        pass
