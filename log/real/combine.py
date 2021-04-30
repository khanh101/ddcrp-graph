import numpy as np
import pandas as pd
from io import StringIO

lines = []
length = None
with open("/home/khanh/data.csv") as fp:
    for line in fp:
        l = line.split(",")
        if length is None:
            length = len(l)
        lines.append(",".join(l[:length]))

string = "\n".join(lines)

TESTDATA = StringIO(string)

df = pd.read_csv(TESTDATA, sep=",")

print((df["improved performance"] / df["naive performance"]).mean())

pass
