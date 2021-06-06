from scipy.stats import chisquare
import numpy as np

# nose-to-neck
# neck-to-rsho, rsho-to-relb, relb-to-rwrist
# neck-to-lsho, lsho-to-lelb, lelb-to-lwrist
# neck-to-midhip
# midhip-to-rhip, rhip-to-rknee, rknee-to-rankle
# midhip-to-lhip, lhip-to-lknee, lknee-to-lankle
man = np.array([46.74, 52.72, 78.64, 21.70, 50.77, 85.45, 32.25, 184.34, 34.05, 94.02, 130.26, 32.02, 94.86, 134.97])
woman = np.array([46.88, 49.42, 77.35, 14.83, 47.63, 87.16, 23.64, 179.06, 33.38, 96.31, 127.25, 31.31, 97.21, 127.16])

full = np.array([50.00, 38.10, 77.59, 27.74, 36.19, 78.49, 29.76, 158.63, 27.88, 99.28, 121.52, 25.54, 97.87, 123.06])
nude = np.array([48.32, 38.20, 73.18, 29.74, 36.35, 74.72, 24.11, 159.03, 26.97, 88.36, 124.57, 25.57, 87.78, 130.61])

obs = np.array([full, nude]).T
print(obs.shape)

result = chisquare(obs, axis=None)
print(result)