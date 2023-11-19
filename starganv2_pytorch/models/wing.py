# Copyright 2023 Lornatang Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import numpy as np
import torch
from skimage.filters import gaussian
from torchvision.utils import save_image

from fan import FAN


def tensor2numpy255(tensor):
    """Converts torch tensor to numpy array."""
    return ((tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype("uint8")


def np2tensor(image):
    """Converts numpy array to torch tensor."""
    return torch.FloatTensor(image).permute(2, 0, 1) / 255 * 2 - 1


class FaceAligner():
    def __init__(self, fname_wing, fname_celeba_mean, output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fan = FAN(fname_pretrained=fname_wing).to(self.device).eval()
        scale = output_size // 256
        self.CELEB_REF = np.float32(np.load(fname_celeba_mean)["mean"]) * scale
        self.xaxis_ref = landmarks2xaxis(self.CELEB_REF)
        self.output_size = output_size

    def align(self, imgs, output_size=256):
        """ imgs = torch.CUDATensor of BCHW """
        imgs = imgs.to(self.device)
        landmarkss = self.fan.get_landmark(imgs).cpu().numpy()
        for i, (img, landmarks) in enumerate(zip(imgs, landmarkss)):
            img_np = tensor2numpy255(img)
            img_np, landmarks = pad_mirror(img_np, landmarks)
            transform = self.landmarks2mat(landmarks)
            rows, cols, _ = img_np.shape
            rows = max(rows, self.output_size)
            cols = max(cols, self.output_size)
            aligned = cv2.warpPerspective(img_np, transform, (cols, rows), flags=cv2.INTER_LANCZOS4)
            imgs[i] = np2tensor(aligned[:self.output_size, :self.output_size, :])
        return imgs

    def landmarks2mat(self, landmarks):
        T_origin = points2T(landmarks, "from")
        xaxis_src = landmarks2xaxis(landmarks)
        R = vecs2R(xaxis_src, self.xaxis_ref)
        S = landmarks2S(landmarks, self.CELEB_REF)
        T_ref = points2T(self.CELEB_REF, "to")
        matrix = np.dot(T_ref, np.dot(S, np.dot(R, T_origin)))
        return matrix


def points2T(point, direction):
    point_mean = point.mean(axis=0)
    T = np.eye(3)
    coef = -1 if direction == "from" else 1
    T[:2, 2] = coef * point_mean
    return T


def landmarks2eyes(landmarks):
    idx_left = np.array(list(range(60, 67 + 1)) + [96])
    idx_right = np.array(list(range(68, 75 + 1)) + [97])
    left = landmarks[idx_left]
    right = landmarks[idx_right]
    return left.mean(axis=0), right.mean(axis=0)


def landmarks2mouthends(landmarks):
    left = landmarks[76]
    right = landmarks[82]
    return left, right


def rotate90(vec):
    x, y = vec
    return np.array([y, -x])


def landmarks2xaxis(landmarks):
    eye_left, eye_right = landmarks2eyes(landmarks)
    mouth_left, mouth_right = landmarks2mouthends(landmarks)
    xp = eye_right - eye_left  # x" in pggan
    eye_center = (eye_left + eye_right) * 0.5
    mouth_center = (mouth_left + mouth_right) * 0.5
    yp = eye_center - mouth_center
    xaxis = xp - rotate90(yp)
    return xaxis / np.linalg.norm(xaxis)


def vecs2R(vec_x, vec_y):
    vec_x = vec_x / np.linalg.norm(vec_x)
    vec_y = vec_y / np.linalg.norm(vec_y)
    c = np.dot(vec_x, vec_y)
    s = np.sqrt(1 - c * c) * np.sign(np.cross(vec_x, vec_y))
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    return R


def landmarks2S(x, y):
    x_mean = x.mean(axis=0).squeeze()
    y_mean = y.mean(axis=0).squeeze()
    # vectors = mean -> each point
    x_vectors = x - x_mean
    y_vectors = y - y_mean

    x_norms = np.linalg.norm(x_vectors, axis=1)
    y_norms = np.linalg.norm(y_vectors, axis=1)

    indices = [96, 97, 76, 82]  # indices for eyes, lips
    scale = (y_norms / x_norms)[indices].mean()

    S = np.eye(3)
    S[0, 0] = S[1, 1] = scale
    return S


def pad_mirror(img, landmarks):
    H, W, _ = img.shape
    img = np.pad(img, ((H // 2, H // 2), (W // 2, W // 2), (0, 0)), "reflect")
    small_blurred = gaussian(cv2.resize(img, (W, H)), H // 100, multichannel=True)
    blurred = cv2.resize(small_blurred, (W * 2, H * 2)) * 255

    H, W, _ = img.shape
    coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    weight_y = np.clip(coords[0] / (H // 4), 0, 1)
    weight_x = np.clip(coords[1] / (H // 4), 0, 1)
    weight_y = np.minimum(weight_y, np.flip(weight_y, axis=0))
    weight_x = np.minimum(weight_x, np.flip(weight_x, axis=1))
    weight = np.expand_dims(np.minimum(weight_y, weight_x), 2) ** 4
    img = img * weight + blurred * (1 - weight)
    landmarks += np.array([W // 4, H // 4])
    return img, landmarks


def align_faces(args, input_dir, output_dir):
    import os
    from torchvision import transforms
    from PIL import Image
    from starganv2_pytorch.utils.common import denorm

    aligner = FaceAligner(args.wing_path, args.lm_path, args.img_size)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    fnames = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    fnames.sort()
    for fname in fnames:
        image = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        x = transform(image).unsqueeze(0)
        x_aligned = aligner.align(x)
        x_aligned = denorm(x_aligned)
        save_image(x_aligned, os.path.join(output_dir, fname), nrow=1, padding=0)
        print("Saved the aligned image to %s..." % fname)
