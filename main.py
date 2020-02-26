import pandas as pd

AK_FILE = "ak.txt"
KP_FILE = "kp.txt"

f_ak = open(AK_FILE,"r")
ak = f_ak.readlines()
f_ak.close()

f_kp = open(KP_FILE,"r")
kp = f_kp.readlines()
f_kp.close()


ak_set = set()
kp_set = set()

for ak_doc, kp_doc in zip(ak,kp):
    ak_set.update(ak_doc.rstrip('\r\n').split(','))
    kp_set.update(kp_doc.rstrip('\r\n').split(','))

occurrence_matrix = pd.DataFrame(0, index=ak_set, columns=kp_set, dtype=int)

for ak_doc, kp_doc in zip(ak,kp):
    ak_word_list = ak_doc.rstrip('\r\n').split(',')
    for kp_word in kp_doc.rstrip('\r\n').split(','):
        for ak_word in ak_word_list:
            occurrence_matrix[kp_word][ak_word] += 1

frequency_matrix = occurrence_matrix / occurrence_matrix.sum()








