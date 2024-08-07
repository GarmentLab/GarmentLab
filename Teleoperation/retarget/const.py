HAND_KEYPOINT_NAMES = [
    "palm",
    "thbase",
    "thmiddle",
    "thdistal",
    "thtip",
    "ffknuckle",
    "ffmiddle",
    "ffdistal",
    "fftip",
    "mfknuckle",
    "mfmiddle",
    "mfdistal",
    "mftip",
    "rfknuckle",
    "rfmiddle",
    "rfdistal",
    "rftip",
    "lfknuckle",
    "lfmiddle",
    "lfdistal",
    "lftip",
]

PALM_KEYPOINT_NAMES = ["palm", "ffknuckle", "mfknuckle", "rfknuckle"]
PALM_KEYPOINT_INDICES = [
    HAND_KEYPOINT_NAMES.index(name) for name in PALM_KEYPOINT_NAMES
]

HAND_VISULIZATION_LINKS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
