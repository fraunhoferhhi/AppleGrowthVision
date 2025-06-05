import argparse
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from sympy import Eq, solve, Symbol

class TRSCamera:
    """Intermediate Camera Type to hold converted camera data"""
    def __init__(self, camera_id=0, width=0, height=0):
        self.camera_id_ = camera_id
        self.width_ = width
        self.height_ = height
        self.camera_mat_ = np.eye(3)
        self.distortion_ = np.zeros((1, 5))
        self.extr_ = np.eye(4)
    
    def invert(self):
        if self.extr_.shape == (4, 4):
            rot_inv = np.linalg.inv(self.extr_[:3, :3])
            self.extr_[:3, :3] = rot_inv
            self.extr_[:3, 3] = -rot_inv @ self.extr_[:3, 3]

def rotation_to_quaternion(R):
    """Convert rotation matrix to quaternion"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        j = (i + 1) % 3
        k = (i + 2) % 3
        
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0, 0, 0, 0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
        x, y, z, w = q
    
    # Ensure quaternion is not all zeros
    if not any((w, x, y, z)):
        w = 1.0
    
    return [w, x, y, z]

def convert_xml_to_cameras(xml_path):
    """Extract camera data from VMCalib XML and convert to TRSCamera objects"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    cameras = {}
    
    for idx, cam_elem in enumerate(root.findall('.//camera')):
        # Extract basic camera parameters
        raw_width = int(cam_elem.find('.//width').text) if cam_elem.find('.//width') is not None else 0
        raw_height = int(cam_elem.find('.//height').text) if cam_elem.find('.//height') is not None else 0
        cam_id = int(cam_elem.find('.//ordinal').text) if cam_elem.find('.//ordinal') is not None else idx
        
        # Calculate intrinsic parameters
        aspect = float(cam_elem.find('.//aspect').text) if cam_elem.find('.//aspect') is not None else 1.333
        hangle = float(cam_elem.find('.//hangle').text) if cam_elem.find('.//hangle') is not None else 0.4
        x0 = float(cam_elem.find('.//x0').text) if cam_elem.find('.//x0') is not None else 0
        y0 = float(cam_elem.find('.//y0').text) if cam_elem.find('.//y0') is not None else 0
        
        # Apply the crucial width/height swapping logic from VMCamera.py
        if aspect < 1.0 and raw_width > raw_height:
            width = raw_height
            height = raw_width
        else:
            width = raw_width
            height = raw_height
        
        # Create camera object
        camera = TRSCamera(camera_id=cam_id, width=width, height=height)
        
        # Calculate focal length (using the same logic as in VMCamera computeFxFy)
        tan_half_h = np.tan(hangle * 0.5)
        width_angle = 2.0 * np.arctan(aspect * tan_half_h)
        fx = width * (0.5 / np.tan(width_angle * 0.5))
        fy = height * (0.5 / tan_half_h)
        
        # Calculate principal point
        cx = 0.5 * (x0 + 1.0) * width
        cy = 0.5 * (1.0 - y0) * height
        
        # Set camera matrix
        camera.camera_mat_ = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Extract distortion parameters
        kappa = [0, 0, 0]
        for i, elem in enumerate([e for e in cam_elem.iter() if 'kappa' in e.tag][:3]):
            kappa[i] = float(elem.text)
        
        camera.distortion_ = np.array([[kappa[0], kappa[1], 0, 0, kappa[2]]])
        
        # Extract rotation params
        nx = float(cam_elem.find('.//nx').text) if cam_elem.find('.//nx') is not None else 0
        ny = float(cam_elem.find('.//ny').text) if cam_elem.find('.//ny') is not None else 1
        nz = float(cam_elem.find('.//nz').text) if cam_elem.find('.//nz') is not None else 0
        theta = float(cam_elem.find('.//theta').text) if cam_elem.find('.//theta') is not None else 0
        
        # Extract translation
        tx = float(cam_elem.find('.//tx').text) if cam_elem.find('.//tx') is not None else 0
        ty = float(cam_elem.find('.//ty').text) if cam_elem.find('.//ty') is not None else 0
        tz = float(cam_elem.find('.//tz').text) if cam_elem.find('.//tz') is not None else 0
        
        # Normalize axis
        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
        if norm > 0:
            nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        # Convert to rotation matrix using Rodriguez formula
        if abs(theta) < 1e-8:
            R = np.eye(3)
        else:
            c = np.cos(theta)
            s = np.sin(theta)
            v = 1 - c
            
            R = np.array([
                [nx*nx*v + c,    nx*ny*v - nz*s, nx*nz*v + ny*s],
                [ny*nx*v + nz*s, ny*ny*v + c,    ny*nz*v - nx*s],
                [nz*nx*v - ny*s, nz*ny*v + nx*s, nz*nz*v + c]
            ])
        
        # Create extrinsic matrix and apply VMLib to OpenCV conversion
        R_t = R.T
        t = -R_t @ np.array([tx, ty, tz])
        
        # Apply coordinate system conversion (Y and Z axis inversion)
        flip = np.diag([1.0, -1.0, -1.0])
        R_conv = flip @ R_t @ flip
        t_conv = t * np.array([1.0, -1.0, -1.0])
        
        # Update extrinsic matrix
        camera.extr_ = np.eye(4)
        camera.extr_[:3, :3] = R_conv
        camera.extr_[:3, 3] = t_conv
        
        cameras[idx] = camera
    
    return cameras

def save_cameras_to_json(cameras, output_path):
    """Save camera data to JSON in the required format"""
    data = {
        "ref_camera_id": 1,
        "cameras": []
    }
    
    # Calculate offset for camera IDs
    offset = 0
    first_camera = next(iter(cameras.values())) if cameras else None
    if first_camera and first_camera.camera_id_ != 1:
        y = Symbol('y')
        eq = Eq(y + first_camera.camera_id_, 1.0)
        offset = int(solve(eq)[0])
    
    for i, camera in enumerate(cameras.values()):
        # Invert transformation for non-reference cameras
        camera_copy = TRSCamera(camera.camera_id_, camera.width_, camera.height_)
        camera_copy.camera_mat_ = camera.camera_mat_.copy()
        camera_copy.distortion_ = camera.distortion_.copy()
        camera_copy.extr_ = camera.extr_.copy()
        
        if (camera_copy.camera_id_ + offset) != 1:
            camera_copy.invert()
        
        # Convert rotation matrix to quaternion
        quat = rotation_to_quaternion(camera_copy.extr_[:3, :3])
        
        # Add camera to JSON
        data['cameras'].append({
            "camera_id": (camera_copy.camera_id_ + offset),
            "image_prefix": "L_/L_" if (camera_copy.camera_id_ + offset) == 1 else "R_/R_",
            "cam_from_rig_translation": [
                float(camera_copy.extr_[0, 3]),
                float(camera_copy.extr_[1, 3]), 
                float(camera_copy.extr_[2, 3])
            ],
            "cam_from_rig_rotation": quat
        })
    
    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump([data], f)
    print(f"Written camera data to JSON file: {output_path}")

def save_camera_params_json(cameras, output_path):
    """Save camera parameters to JSON in format with intrinsics and stereo information"""
    
    # Create basic structure
    data = {
        "ref_camera": 0,  # Assuming reference camera is the first one
        "cameras": [],
        "stereo": {}
    }
    
    # Add camera intrinsic parameters
    for i, camera in enumerate(cameras.values()):
        cam_data = {
            "camera_id": camera.camera_id_,
            "fx": float(camera.camera_mat_[0, 0]),
            "fy": float(camera.camera_mat_[1, 1]),
            "cx": float(camera.camera_mat_[0, 2]),
            "cy": float(camera.camera_mat_[1, 2]),
            # Add distortion parameters
            "k1": float(camera.distortion_[0, 0]),
            "k2": float(camera.distortion_[0, 1]),
            "p1": float(camera.distortion_[0, 2]),
            "p2": float(camera.distortion_[0, 3]),
            "k3": float(camera.distortion_[0, 4])
        }
        
        data["cameras"].append(cam_data)
    
    # Add stereo information - if we have at least 2 cameras
    if len(cameras) >= 2:
        cam_keys = list(cameras.keys())
        cam0 = cameras[cam_keys[0]]
        cam1 = cameras[cam_keys[1]]
        
        # Calculate relative transformation from cam0 to cam1
        # First, get the inverse of cam0's transformation matrix
        cam0_inv = np.eye(4)
        cam0_inv[:3, :3] = np.linalg.inv(cam0.extr_[:3, :3])
        cam0_inv[:3, 3] = -cam0_inv[:3, :3] @ cam0.extr_[:3, 3]
        
        # Then compute the relative transformation
        rel_transform = np.matmul(cam1.extr_, cam0_inv)
        
        # Extract translation components
        tx, ty, tz = rel_transform[:3, 3]
        
        # Extract rotation components using Euler angles
        R = rel_transform[:3, :3]
        
        # Convert rotation matrix to Euler angles (rx, ry, rz)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0
        
        # Add to stereo section
        data["stereo"] = {
            "tx": float(tx),
            "ty": float(ty),
            "tz": float(tz),
            "rx": float(rx),
            "ry": float(ry),
            "rz": float(rz)
        }
    
    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Written camera parameters to JSON file: {output_path}")

def process_calibcon(input_path, output_json=None, params_json=None):
    """Main function to process calibration files"""
    input_path = Path(input_path)
    
    # Get XML file path
    if input_path.is_dir():
        xml_files = list(input_path.glob('*.xml'))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {input_path}")
        file_path = xml_files[0]
    else:
        file_path = input_path
    
    # Convert XML to camera objects
    cameras = convert_xml_to_cameras(file_path)
    
    # Save to camera transform JSON if requested
    if output_json:
        save_cameras_to_json(cameras, output_json)
        
    # Save to camera parameters JSON if requested
    if params_json:
        save_camera_params_json(cameras, params_json)
    
    return cameras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VMCalib XML to OpenCV format')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input XML file or directory')
    parser.add_argument('-o', '--output', type=str, help='Output JSON file for camera transforms')
    parser.add_argument('-p', '--params', type=str, help='Output JSON file for camera parameters')
    args = parser.parse_args()
    
    cameras = process_calibcon(args.input, args.output, args.params)
    print(f"Successfully processed {len(cameras)} cameras")

# Example usage:
# As a module
""" from calibcon_script import process_calibcon

cameras = process_calibcon("input.xml", params_json="parameters.json") """

# From command line
""" python calibcon_script.py -i input.xml -o transforms.json -p parameters.json """
