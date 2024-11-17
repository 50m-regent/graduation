import torch
from matplotlib import pyplot
from tqdm import tqdm


N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SAMPLE_SIZE = 1000000

pyplot.rcParams["text.usetex"] = True


def search_cos_similarity(n: int) -> None:
    vec = torch.randn(n)
    vecs = torch.randn(SAMPLE_SIZE, n)

    cos_similarity = torch.cosine_similarity(vec, vecs)

    pyplot.hist(cos_similarity, bins=100, range=(-1, 1), density=True, label=f"$n={n}$")

    pyplot.xlim(-1, 1)
    pyplot.legend()

    pyplot.savefig(f"outputs/cos_similarity/{n}.pdf", bbox_inches="tight")
    pyplot.cla()


def main() -> None:
    for n in tqdm(N):
        search_cos_similarity(n)


if __name__ == "__main__":
    main()
