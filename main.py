from Predictor import Predictor


AK_FILE = "ak.txt"
KP_FILE = "kp.txt"

predictor = Predictor()
predictor.generate_from_file(AK_FILE, KP_FILE)
predictor.compute_kp_probabilities({'a'})
predictor.compute_kp_probabilities({'d'})
predictor.compute_kp_probabilities({'a','c'})
predictor.compute_kp_probabilities({'a','c','b'})
predictor.compute_kp_probabilities({'a','b'})


print(predictor.compute_kp({'a'}))
print(predictor.compute_kp({'d'}))
print(predictor.compute_kp({'c'}))
print(predictor.compute_kp({'a','c'}))
print(predictor.compute_kp({'a','c','b'}))
print(predictor.compute_kp({'a','b'}))

print(predictor.compute_kp_set({'a'}))









