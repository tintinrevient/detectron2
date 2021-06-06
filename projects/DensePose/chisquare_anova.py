from scipy.stats import chisquare
import scipy.stats as stats
import numpy as np

# nose-to-neck
# neck-to-rsho, rsho-to-relb, relb-to-rwrist
# neck-to-lsho, lsho-to-lelb, lelb-to-lwrist
# neck-to-midhip
# midhip-to-rhip, rhip-to-rknee, rknee-to-rankle
# midhip-to-lhip, lhip-to-lknee, lknee-to-lankle
man = [46.74, 52.72, 78.64, 21.70, 50.77, 85.45, 32.25, 184.34, 34.05, 94.02, 130.26, 32.02, 94.86, 134.97]
woman = [46.88, 49.42, 77.35, 14.83, 47.63, 87.16, 23.64, 179.06, 33.38, 96.31, 127.25, 31.31, 97.21, 127.16]

classical = [46.45, 38.62, 76.07, 16.66, 36.20, 75.24, 21.68, 156.17, 28.06, 76.85, 149.65, 25.21, 83.02, 125.58]
modern = [49.07, 39.05, 78.26, 28.57, 38.58, 78.63, 26.40, 170.54, 28.19, 96.02, 127.18, 25.95, 94.53, 128.36]

full = [50.00, 38.10, 77.59, 27.74, 36.19, 78.49, 29.76, 158.63, 27.88, 99.28, 121.52, 25.54, 97.87, 123.06]
nude = [48.32, 38.20, 73.18, 29.74, 36.35, 74.72, 24.11, 159.03, 26.97, 88.36, 124.57, 25.57, 87.78, 130.61]

# Chi-squared test
# obs = np.array([man, women]).T
# print(obs.shape)
# print(obs.ravel())
#
# result = chisquare(obs, axis=None)
# # result = chisquare(obs)
# print(result)

# ANOVA
result = stats.f_oneway(full, nude)
print(result)