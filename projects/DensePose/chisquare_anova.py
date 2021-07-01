from scipy.stats import chisquare
import scipy.stats as stats
import numpy as np

# keypoints
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

# area from DensePose
# head, torso
# right upper arm, right lower arm, left upper arm, left lower arm
# right thigh, right calf, left thigh, left calf
area_coco_man = [3164.60, 8491.30, 1540.29, 1060.48, 1549.22, 1089.24, 3923.21, 2294.13, 3980.68, 2229.65]
area_coco_woman = [2539.04, 6133.00, 1254.51, 838.16, 1301.31, 854.25, 3303.78, 1871.76, 3361.22, 1884.94]

area_michelangelo = [2951.20, 10734.24, 2416.96, 1632.0, 2416.96, 1632.0, 4300.0, 3302.4, 4300.0, 3302.4]
area_paul_delvaux = [2490.13, 9751.77, 1289.60, 1012.30, 1289.60, 1012.30, 3550.49, 1890.99, 3550.49, 1890.99]

area_coco_surf_woman = [2668, 5684, 1416, 883, 1416, 883, 3560, 1728, 3560, 1728]
area_impressionism_nude_woman = [2552, 7854, 1334, 1134, 1334, 1134, 3280, 1980, 3280, 1980]

# length of limbs
# head, torso
# upper arm, lower arm
# upper leg, lower leg
length_vitruve = [62, 167, 62.5, 75, 125, 108]
length_michelangelo = [62, 128.4, 72.8, 68.0, 86.0, 86.0]
length_coco_man = [62, 113.1, 62.7, 51.0, 89.4, 79.5]

# Chi-squared test
# obs = np.array([man, women]).T
# print(obs.shape)
# print(obs.ravel())
#
# result = chisquare(obs, axis=None)
# # result = chisquare(obs)
# print(result)

# ANOVA
result = stats.f_oneway(area_coco_surf_woman, area_impressionism_nude_woman)
# result = stats.f_oneway(length_vitruve, length_michelangelo)
print(result)