import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..')) 
import pyrealsense2 as rs
from FoundationPose.estimater import *
from FoundationPose.mask_gen import *
import tkinter as tk
from tkinter import filedialog
import glob
import pickle as pkl
from PIL import Image
from pdb import set_trace
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import time

# TODO: modify the path of real_world
path_to_real_world = "/PATH/TO/real_world/"

def main():

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--est_refine_iter', type=int, default=4)
    parser.add_argument('--track_refine_iter', type=int, default=2) # 4
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    root = tk.Tk()
    root.withdraw()

    mesh_path = filedialog.askopenfilename()
    if not mesh_path:
        print("No mesh file selected")
        exit(0)
    mesh = trimesh.load(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx)

    CONFIG_PATH = f"{path_to_real_world}/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    CHECKPOINT_PATH = f"{path_to_real_world}/repos/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    
    gdino_model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    DEVICE = "cuda"  
    BOX_TRESHOLD = 0.2    
    TEXT_TRESHOLD = 0.5  
    # TODO: modify the text prompt
    TEXT_PROMPT = "A purple plate"    

    # TODO: modify the intrinsics of head camera
    cam_K = np.array([[604.69128418 , 0., 317.77008057],
                    [0., 603.96697998, 249.44818115],
                    [0., 0., 1.]])

    time.sleep(3)

    # TODO: modify the task name
    task_name = "handover_and_insert_the_plate" 

    task_folder = f'{path_to_real_world}/real_data/{task_name}'
    obj_6dpose_folders = f'{task_folder}/obj_6dpose'
    
    episodes = sorted(glob.glob(f'{task_folder}/episode*'))
    for episode_id, episode_folder in enumerate(episodes):
        print(f"================ episode {episode_id} ================")

        obj_6dpose_folder = f"{obj_6dpose_folders}/episode{episode_id:04d}"
        os.makedirs(obj_6dpose_folder, exist_ok=True)

        obj_6dpose_file = f'{obj_6dpose_folder}/obj_6dpose.pkl'

        if os.path.exists(obj_6dpose_file):
            print(f'{obj_6dpose_file} exists')
            continue

        cv2.destroyAllWindows()
        i = 0
        steps_folder = f'{episode_folder}/steps'
        steps = sorted(glob.glob(f'{steps_folder}/*'))

        keyframes = []
        for step in steps:
            with open(f'{step}/other_data.pkl', 'rb') as f:
                other_data = pickle.load(f)
            if other_data['is_keyframe']:
                step_id = int(step.split('/')[-1])
                keyframes.append(step_id)
        print(keyframes)

        for camera in ['head']:
            obj_6dpose = []

            if_register = True
            
            for step_id, step in enumerate(steps):
                with open(f'{step}/other_data.pkl', 'rb') as f:
                    data = pkl.load(f)
                    extr = data['extr'][camera]

                color_file = f'{step}/{camera}_rgb.jpg'
                color_image = cv2.imread(color_file)[...,::-1] # BGR to RGB
                depth_file = f'{step}/{camera}_depth_x10000_uint16.png'
                depth_x10000_uint16 = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                depth_array = depth_x10000_uint16 / 10000

                H, W = color_image.shape[:2]
                color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
                depth = cv2.resize(depth_array, (W,H), interpolation=cv2.INTER_NEAREST)
                depth[(depth<0.1) | (depth>=np.inf)] = 0


                # if if_register or (step_id+1) in keyframes:
                if if_register:
                    if_register = False
                # if i==0:
                    print(f"================ registering {step_id} ================")
                    mask_file_path = create_mask(camera, steps_folder, gdino_model, DEVICE, BOX_TRESHOLD, TEXT_TRESHOLD, TEXT_PROMPT, step_id)
                    mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
                    if len(mask.shape)==3:
                        for c in range(3):
                            if mask[...,c].sum()>0:
                                mask = mask[...,c]
                                break
                    mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)

                    # print(mask.shape)
                    # set_trace()
                    torch.cuda.synchronize() 
                    t1 = time.time()
                    pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                    torch.cuda.synchronize() 
                    t2 = time.time()
                    print(f"register time: {t2-t1}")
                
                else:
                    torch.cuda.synchronize() 
                    t1 = time.time()
                    pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
                    torch.cuda.synchronize() 
                    t2 = time.time()
                    print(f"track time: {t2-t1}")
                
                
                center_pose = pose@np.linalg.inv(to_origin)
                center_pose_in_world = extr@center_pose
                obj_6dpose.append(center_pose_in_world)
                vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
                window_name = episode_folder.split('/')[-2] + ' ' + episode_folder.split('/')[-1]
                cv2.namedWindow(window_name)
                cv2.displayOverlay(window_name, f"step: {step_id}")
                cv2.imshow(window_name, vis[...,::-1])
                op = cv2.waitKey(10)
                if op == ord("r"):
                    if_register = True
                i += 1
            
            obj_6dpose = np.array(obj_6dpose)
            print(obj_6dpose.shape)
            with open(obj_6dpose_file, 'wb') as f:
                pkl.dump(obj_6dpose, f)
                print(f'{obj_6dpose_file} saved')


if __name__ == '__main__':
    main()
