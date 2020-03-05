from MeanProbabilityPredictor import MeanProbabilityPredictor
from ProbabilityPredictor import ProbabilityPredictor
from Validator import Validator

AK_FILE = "ak_preprocessed.txt"
KP_FILE = "kp_preprocessed.txt"

predictor = ProbabilityPredictor()
predictor.generate_from_file(AK_FILE, KP_FILE)
# predictor.compute_kp_scores({'petalo'})
# predictor.compute_kp_scores({'concurrency'})
# predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills'})
# predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills','petalo'})
# predictor.compute_kp_scores({'petalo', 'concurrency'})


print("Mean predictor")

predictor = MeanProbabilityPredictor()
predictor.generate_from_file(AK_FILE, KP_FILE)
# predictor.compute_kp_scores({'petalo'})
# predictor.compute_kp_scores({'concurrency'})
# predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills'})
# predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills','petalo'})
# predictor.compute_kp_scores({'petalo', 'concurrency'})



print("Validation 1")

validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=0.6)
results = validator.validate(ProbabilityPredictor, energy=0.5)
print(results)

print("Validation 2")

validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=0.7)
results = validator.validate(MeanProbabilityPredictor, energy=0.7)
print(results)






