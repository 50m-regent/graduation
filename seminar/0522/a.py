from matplotlib import pyplot
import numpy
import seaborn

pyplot.rcParams["text.usetex"] = True

n = 8192
d = 3072

encoding = numpy.asarray(
    [i / (10000 ** (2 * numpy.arange(0, d) / d)) for i in range(1, n + 1)]
)

fig = pyplot.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
seaborn.heatmap(numpy.cos(encoding), ax=ax1)
seaborn.heatmap(numpy.sin(encoding), ax=ax2)
ax1.set_xlabel(r"$j$")
ax2.set_xlabel(r"$j$")
ax1.set_ylabel(r"$i$")
ax2.set_ylabel(r"$i$")

fig.savefig("absolute.png", dpi=300, bbox_inches="tight")

theta = numpy.asarray(
    [
        (i * (10000 ** (-2 * (numpy.arange(1, d // 2) - 1) / d))) % (2 * numpy.pi)
        for i in range(1, n + 1)
    ]
)
fig = pyplot.figure()
ax = fig.add_subplot(1, 1, 1)
seaborn.heatmap(theta, ax=ax, cmap="hsv")
ax.set_xlabel(r"$i$")
ax.set_ylabel(r"$m$")

fig.savefig("rope.png", dpi=300, bbox_inches="tight")
