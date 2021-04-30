import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter(df0: pd.DataFrame, attr: str, value: float, range: float = 0.1):
    return filter_range(df0, attr, value * (1 - range), value * (1 + range))


def filter_range(df0: pd.DataFrame, attr: str, lower: float, upper):
    df1 = df0[df0[attr] >= lower]
    df2 = df1[df1[attr] < upper]
    return df2


df = pd.read_csv("./log.csv")


def get_ratio(df: pd.DataFrame):
    # modularity
    ratio = df["improved modularity"].to_numpy() / df["naive modularity"].to_numpy()
    ratio = ratio[~np.isnan(ratio)]
    return ratio


ratio = get_ratio(df)
prob = (ratio < 1).mean()
print(prob)
print(ratio.mean(), ratio.std())
hist, edges = np.histogram(ratio, bins=20, range=[0, 2])
centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(hist))]
plt.title(f"marginal ratio of modularity")
plt.xlabel(f"ratio of improved modularity over naive modularity")
plt.ylabel(f"occurrences")
plt.bar(centers, hist, width=centers[1] - centers[0])
plt.show()

for average_degree in [10, 20, 30, 40, 50]:
    df1 = filter(df, "average degree", average_degree)
    plt.title(f"modularity over predicted number of clusters, average degree {average_degree}")
    plt.xlabel("number of clusters")
    plt.ylabel("modularity")
    plt.xlim([0, 300])
    plt.ylim([0, 0.4])
    plt.scatter(df1["predicted cluster size"], df1["naive modularity"], s=8)
    plt.scatter(df1["predicted cluster size"], df1["improved modularity"], s=8)
    plt.show()

x = []
y = []
step = 10
minx = 0
maxx = 300
for i in range(minx, 1 + maxx, step):
    j = i + step
    df1 = filter_range(df, "predicted cluster size", i, j)
    x.append((i + j) / 2)
    y.append(get_ratio(df1).mean())
plt.title(f"marginal of modularity over predicted number of clusters")
plt.xlabel(f"number of clusters")
plt.ylabel(f"ratio")
plt.xlim([minx, maxx])
plt.ylim([0.8, 1.5])
plt.plot([minx, maxx], [1, 1], 'black')
plt.plot([50, 50], [0.8, 1.5], 'black')
plt.plot(x, y)
plt.show()

for average_degree in [10, 20, 30, 40, 50]:
    df1 = filter(df, "average degree", average_degree)
    x = (np.sqrt(1 / df1["scale"]) ** 50) * df1["graph size"]
    y = df1["predicted cluster size"]

    plt.scatter(average_degree * np.log(x), np.log(y))
plt.show()
