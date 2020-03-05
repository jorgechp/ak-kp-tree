from MeanProbabilityPredictor import MeanProbabilityPredictor
from ProbabilityPredictor import ProbabilityPredictor
from Validator import Validator

AK_FILE = "ak.txt"
KP_FILE = "kp.txt"

predictor = ProbabilityPredictor()
predictor.generate_from_file(AK_FILE, KP_FILE)
predictor.compute_kp_scores({'a'})
predictor.compute_kp_scores({'d'})
predictor.compute_kp_scores({'a', 'c'})
predictor.compute_kp_scores({'a', 'c', 'b'})
predictor.compute_kp_scores({'a', 'b'})


print(predictor.compute_kp({'a'}))
print(predictor.compute_kp({'d'}))
print(predictor.compute_kp({'c'}))
print(predictor.compute_kp({'a','c'}))
print(predictor.compute_kp({'a','c','b'}))
print(predictor.compute_kp({'a','b'}))

print(predictor.compute_kp_set({'a'}))

print("Mean predictor")

predictor2 = MeanProbabilityPredictor()
predictor2.generate_from_file(AK_FILE, KP_FILE)
predictor2.compute_kp_scores({'a'})
predictor2.compute_kp_scores({'d'})
predictor2.compute_kp_scores({'a', 'c'})
predictor2.compute_kp_scores({'a', 'c', 'b'})
predictor2.compute_kp_scores({'a', 'b'})


print(predictor2.compute_kp({'a'}))
print(predictor2.compute_kp({'d'}))
print(predictor2.compute_kp({'c'}))
print(predictor2.compute_kp({'a','c'}))
print(predictor2.compute_kp({'a','c','b'}))
print(predictor2.compute_kp({'a','b'}))

print(predictor2.compute_kp_set({'a'}))


print("Validation 1")

validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=0.6)
results = validator.validate(ProbabilityPredictor, energy=0.5)
print(results)

print("Validation 2")

validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=0.6)
results = validator.validate(MeanProbabilityPredictor, energy=0.5)
print(results)






