event_num_attribs = [
    "P-PDG",
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP",
    "P-JUS-CKGL",
    #'T-JUS-CKGL', won't be included, will be dropped by column transformer
    "QGL",
]
event_time_attrib = ["timestamp"]
event_class_attrib = "class"

num_class_types = 9
