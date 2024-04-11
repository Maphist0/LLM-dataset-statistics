import numpy as np
import matplotlib.pyplot as plt

maxbin = 2000
binsize = 10

def draw_req_res_distribution(req_tokens, res_tokens, fname, xlabel="Request Tokens"):
    bins = np.linspace(0, maxbin, int(maxbin / binsize))
    digitized = np.digitize(req_tokens, bins)
    bin_means = [res_tokens[digitized == i].mean() for i in range(1, len(bins))]
    bin_median = [np.median(res_tokens[digitized == i]) for i in range(1, len(bins))]

    plt.plot(bins[:-1], bin_median, color='C0')
    plt.plot(bins[:-1], bin_means, '--', color='orange')
    plt.ylabel('Response Tokens')
    plt.xlabel(xlabel)
    plt.savefig(fname)
