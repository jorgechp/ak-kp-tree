from MeanProbabilityPredictor import MeanProbabilityPredictor
from ProbabilityPredictor import ProbabilityPredictor
from Validator import Validator

AK_FILE = "ak_preprocessed.txt"
KP_FILE = "kp_preprocessed.txt"

# predictor = ProbabilityPredictor()
# predictor.generate_from_file(AK_FILE, KP_FILE)
# print(predictor.compute_kp_scores({'technicalanalysis','sentimentembeddings','markettrendprediction','supervisedlearning','timeseriesanalysis','sentimentanalysis'}))
# # predictor.compute_kp_scores({'petalo'})
# # predictor.compute_kp_scores({'concurrency'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills','petalo'})
# # predictor.compute_kp_scores({'petalo', 'concurrency'})
#
#
# print("Mean predictor")
#
# predictor = MeanProbabilityPredictor()
# predictor.generate_from_file(AK_FILE, KP_FILE)
# # print(predictor.compute_kp_scores({'technicalanalysis','sentimentembeddings','markettrendprediction','supervisedlearning','timeseriesanalysis','sentimentanalysis'}))
# # predictor.compute_kp_scores({'concurrency'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills','petalo'})
# # predictor.compute_kp_scores({'petalo', 'concurrency'})

TRAINING_RATE = 0.9
ENERGY_RATE = 1.0


print("Validation 1")

validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=TRAINING_RATE)
results = validator.validate(ProbabilityPredictor, energy=ENERGY_RATE)
print(results)

print("Validation 2")

validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=TRAINING_RATE)
results = validator.validate(MeanProbabilityPredictor, energy=ENERGY_RATE)
print(results)






