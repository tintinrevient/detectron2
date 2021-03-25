import torch
import matplotlib.pyplot as plt
import numpy as np

# libraries for reading data from files
from scipy.io import loadmat
from pytorch3d.io.utils import _read_image
import pickle

# data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV
)

# add path for demo utils functions
import sys
import os
# sys.path.append(os.path.abspath(''))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# set paths
uv_data_dir = 'UV_data'
data_filename = os.path.join(uv_data_dir, 'UV_Processed.mat')
tex_filename = os.path.join(uv_data_dir, 'texture_from_SURREAL.png')

# rename your .pkl file or change this string
verts_filename = os.path.join('models', 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

# load SMPL and texture data
with open(verts_filename, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    v_template = torch.Tensor(data['v_template']).to(device) # (6890, 3)

ALP_UV = loadmat(data_filename)
tex = torch.from_numpy(_read_image(file_name=tex_filename, format='RGB') / 255. ).unsqueeze(0).to(device)

verts = torch.from_numpy((ALP_UV["All_vertices"]).astype(int)).squeeze().to(device) # (7829, 1)
U = torch.Tensor(ALP_UV['All_U_norm']).to(device) # (7829, 1)
V = torch.Tensor(ALP_UV['All_V_norm']).to(device) # (7829, 1)
faces = torch.from_numpy((ALP_UV['All_Faces'] - 1).astype(int)).to(device)  # (13774, 3)
face_indices = torch.Tensor(ALP_UV['All_FaceIndices']).squeeze()

# display the texture image
plt.figure(figsize=(10, 10))
plt.imshow(tex.squeeze(0).cpu())
plt.grid("off");
plt.axis("off");

# map each face to a (u, v) offset
offset_per_part = {}
already_offset = set()
cols, rows = 4, 6
for i, u in enumerate(np.linspace(0, 1, cols, endpoint=False)):
    for j, v in enumerate(np.linspace(0, 1, rows, endpoint=False)):
        part = rows * i + j + 1  # parts are 1-indexed in face_indices
        offset_per_part[part] = (u, v)

# iterate over faces and offset the corresponding vertex u and v values
for i in range(len(faces)):
    face_vert_idxs = faces[i]
    part = face_indices[i]
    offset_u, offset_v = offset_per_part[int(part.item())]

    for vert_idx in face_vert_idxs:
        # vertices are reused, but we don't want to offset multiple times
        if vert_idx.item() not in already_offset:
            # offset u value
            U[vert_idx] = U[vert_idx] / cols + offset_u
            # offset v value
            # this also flips each part locally, as each part is upside down
            V[vert_idx] = (1 - V[vert_idx]) / rows + offset_v
            # add vertex to our set tracking offsetted vertices
            already_offset.add(vert_idx.item())

# invert V values
U_norm, V_norm = U, 1 - V

# create our verts_uv values
verts_uv = torch.cat([U_norm[None],V_norm[None]], dim=2) # (1, 7829, 2)

# there are 6890 xyz vertex coordinates but 7829 vertex uv coordinates.
# this is because the same vertex can be shared by multiple faces where each face may correspond to a different body part.
# therefore when initializing the Meshes class,
# we need to map each of the vertices referenced by the DensePose faces (in verts, which is the "All_vertices" field)
# to the correct xyz coordinate in the SMPL template mesh.
v_template_extended = torch.stack(list(map(lambda vert: v_template[vert-1], verts))).unsqueeze(0).to(device) # (1, 7829, 3)

# add a batch dimension to faces
faces = faces.unsqueeze(0)

texture = TexturesUV(maps=tex, faces_uvs=faces, verts_uvs=verts_uv)
mesh = Meshes(v_template_extended, faces, texture)

# initialize a camera.
# world coordinates +Y up, +X left and +Z in.
R, T = look_at_view_transform(2.7, 0, 0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# define the settings for rasterization and shading. here we set the output image to be of size
# 512x512. as we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# place a point light in front of the person.
lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])

# create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
# interpolate the texture uv coordinates for each vertex, sample from a texture image and
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)

images = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.grid("off");
plt.axis("off");

# rotate the person by increasing the elevation and azimuth angles to view the back of the person from above.
R, T = look_at_view_transform(2.7, 10, 180)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# move the light location so the light is shining on the person's back.
lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)

# re-render the mesh, passing in keyword arguments for the modified components.
images = renderer(mesh, lights=lights, cameras=cameras)

plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.grid("off");
plt.axis("off");

plt.show()