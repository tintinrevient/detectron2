import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import pickle
from pycocotools.coco import COCO
import os
import cv2


# Now we can visualize the SMPL template model.
# Now read the smpl model.

with open('./models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
    X, Y, Z = [Vertices[:,0], Vertices[:,1], Vertices[:,2]]


# Let us define some functions to visualize the SMPL model vertices as point clouds,
# showing the whole body and zooming into the face

def smpl_view_set_axis_full_body(ax, azimuth=0):
    ## Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.55
    ax.set_xlim(- max_range, max_range)
    ax.set_ylim(- max_range, max_range)
    ax.set_zlim(-0.2 - max_range, -0.2 + max_range)
    ax.axis('off')


def smpl_view_set_axis_face(ax, azimuth=0):
    ## Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.1
    ax.set_xlim(- max_range, max_range)
    ax.set_ylim(- max_range, max_range)
    ax.set_zlim(0.45 - max_range, 0.45 + max_range)
    ax.axis('off')


## Now let's rotate around the model and zoom into the face.

fig = plt.figure(figsize=[16, 4])

ax = fig.add_subplot(141, projection='3d')
ax.scatter(Z, X, Y, s=0.02, c='k')
smpl_view_set_axis_full_body(ax)

ax = fig.add_subplot(142, projection='3d')
ax.scatter(Z, X, Y, s=0.02, c='k')
smpl_view_set_axis_full_body(ax, 45)

ax = fig.add_subplot(143, projection='3d')
ax.scatter(Z, X, Y, s=0.02, c='k')
smpl_view_set_axis_full_body(ax, 90)

ax = fig.add_subplot(144, projection='3d')
ax.scatter(Z, X, Y, s=0.2, c='k')
smpl_view_set_axis_face(ax, -40)

plt.show()


import numpy as np
import copy
import cv2
from scipy.io import loadmat
import scipy.spatial.distance
import os


class DensePoseMethods:
    def __init__(self):

        ALP_UV = loadmat(os.path.join(os.path.dirname(__file__), './UV_data/UV_Processed.mat'))
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]

        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
        UV_symmetry_filename = os.path.join(os.path.dirname(__file__), './UV_data/UV_symmetry_transforms.mat')
        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        ### This is a function to get the mirror symmetric UV labels.
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)

        for i in (range(24)):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))

                U_loc = (U[jj] * 255).astype(np.int64)
                V_loc = (V[jj] * 255).astype(np.int64)

                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]

        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)

        for i in (range(14)):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]

        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x

        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0

        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False

        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)

        if (np.dot(uCrossW, uCrossV) < 0):
            return False

        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom

        return ((r <= 1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)

        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)

        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom

        return (1 - (r + t), r, t)

    def IUV2FBC(self, I_point, U_point, V_point):
        P = [U_point, V_point, 0]
        FaceIndicesNow = np.where(self.FaceIndices == I_point)
        FacesNow = self.FacesDensePose[FaceIndicesNow]

        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0],
                         np.zeros(self.U_norm[FacesNow][:, 0].shape))).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1],
                         np.zeros(self.U_norm[FacesNow][:, 1].shape))).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2],
                         np.zeros(self.U_norm[FacesNow][:, 2].shape))).transpose()


        for i, [P0, P1, P2] in enumerate(zip(P_0, P_1, P_2)):
            if (self.barycentric_coordinates_exists(P0, P1, P2, P)):
                [bc1, bc2, bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return (FaceIndicesNow[0][i], bc1, bc2, bc3)

        # If the found UV is not inside any faces, select the vertex that is closest!

        D1 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()

        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()

        if ((minD1 < minD2) & (minD1 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.)
        elif ((minD2 < minD1) & (minD2 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.)
        else:
            return (FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.)

    def FBC2PointOnSurface(self, FaceIndex, bc1, bc2, bc3, Vertices):

        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1

        p = Vertices[Vert_indices[0], :] * bc1 + \
            Vertices[Vert_indices[1], :] * bc2 + \
            Vertices[Vert_indices[2], :] * bc3

        return (p)


DP = DensePoseMethods()

# Demo data
# pkl_file = open('./UV_data/demo_dp_single_ann.pkl', 'rb')
# Demo = pickle.load(pkl_file, encoding='latin1')

coco_folder = os.path.join('datasets', 'coco')
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_minival2014.json'))

selected_im = 466986

# Load the image
im = dp_coco.loadImgs(selected_im)[0]

# Load annotations for the selected image
ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
anns = dp_coco.loadAnns(ann_ids)
ann = anns[0] # the first person

num_of_points = len(ann['dp_x'])

bbr = np.round(ann['bbox'])

point_x = np.array(ann['dp_x']) / 255. * bbr[2]  # Strech the points to current box.
point_y = np.array(ann['dp_y']) / 255. * bbr[3]  # Strech the points to current box.

point_I = np.array(ann['dp_I'])
point_U = np.array(ann['dp_U'])
point_V = np.array(ann['dp_V'])

x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]

point_x = point_x + x1; point_y = point_y + y1

collected_x = np.zeros(point_x.shape)
collected_y = np.zeros(point_x.shape)
collected_z = np.zeros(point_x.shape)

for i, (ii,uu,vv) in enumerate(zip(point_I, point_U, point_V)):

    # Convert IUV to FBC (faceIndex and barycentric coordinates.)
    FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu, vv)

    # Use FBC to get 3D coordinates on the surface.
    p = DP.FBC2PointOnSurface(FaceIndex, bc1, bc2, bc3, Vertices)

    collected_x[i] = p[0]
    collected_y[i] = p[1]
    collected_z[i] = p[2]

fig = plt.figure(figsize=[15,5])

# Visualize the image and collected points.
ax = fig.add_subplot(131)

# Now read the image and show
# ax.imshow(Demo['ICrop'])
im_name = os.path.join(coco_folder, 'val2014', im['file_name'])
print('Image name:', im_name)
input_im = cv2.imread(im_name)
ax.imshow(input_im[:,:,::-1]); plt.axis('off')

ax.scatter(point_x, point_y, 11, np.arange(num_of_points))
plt.title('Points on the image'); ax.axis('off')

## Visualize the full body smpl male template model and collected points
ax = fig.add_subplot(132, projection='3d')
ax.scatter(Z, X, Y, s=0.02, c='k')
ax.scatter(collected_z, collected_x, collected_y, s=25, c= np.arange(num_of_points))
smpl_view_set_axis_full_body(ax)
plt.title('Points on the SMPL model')

## Now zoom into the face.
ax = fig.add_subplot(133, projection='3d')
ax.scatter(Z, X, Y, s=0.2, c='k')
ax.scatter(collected_z, collected_x, collected_y, s=55, c=np.arange(num_of_points))
smpl_view_set_axis_face(ax)
plt.title('Points on the SMPL model')

plt.show()