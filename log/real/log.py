import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

def clean(filepath: str) -> StringIO:
    lines = []
    length = None
    with open(filepath) as fp:
        for line in fp:
            l = line.split(",")
            if length is None:
                length = len(l)
            lines.append(",".join(l[:length]))

    string = "\n".join(lines)
    return StringIO(string)

df = pd.read_csv(clean("./hop_1_window_10_scale_1000.csv"))

pass
