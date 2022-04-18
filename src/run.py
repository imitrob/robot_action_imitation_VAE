import copy, math
import numpy as np
import os, pickle
from openpose import util
from openpose.body import Body
from openpose.hand import Hand
from pynput import keyboard
from pepper.robot import Pepper
import glob
import cv2
import time
from vae.vae_learner import IncremVAE
from offline_helpers import train_vae_offline, vae_reconstruct
from vae.infer import load_model
from vae.vae_learner import load_actions_from_path
import random
from vae.utils import check_action_ok
import argparse

armdic = {"wave":[math.radians(0), math.radians(180)], "fly":[math.radians(180), math.radians(180)], "dance":[math.radians(0), math.radians(0)]}

def move_robot(robot, angles, hands, label=None):
    if angles is not None:
        body_angles_in_radians = [math.radians(x) for x in angles[:4]]
        if len(angles) < 6:
            direction = armdic[label]
        else:
            direction = []
            direction.append(math.radians(0)) if int(angles[-2]) > 50 else direction.append(math.radians(180))
            direction.append(math.radians(0)) if int(angles[-1]) > 50 else direction.append(math.radians(180))
        body_angles_in_radians = direction + body_angles_in_radians
        robot.move_joint_by_angle(["LShoulderPitch", "RShoulderPitch", "LShoulderRoll", "LElbowRoll", "RShoulderRoll", "RElbowRoll"], body_angles_in_radians, 0.4)
        time.sleep(.2)
    if hands is not None:
        robot.hand("left", int(hands[0]) > 50)
        robot.hand("right", int(hands[1]) > 50)


def analyse_image(img, pth):
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')
    candidate, subset = body_estimation(img)
    canvas = copy.deepcopy(img)
    canvas, body_angles = util.draw_bodypose(canvas, candidate, subset)
    # Offset joints
    if body_angles:
        body_angles[0] = body_angles[0] - 90
        if body_angles[1] <0:
            body_angles.append(0)
            body_angles[1] = -1*body_angles[1]
        else:
            body_angles.append(100)
        if body_angles[3] <0:
            body_angles.append(0)
            body_angles[3] = -1*body_angles[3]
        else:
            body_angles.append(100)
        body_angles[1] = - (180 - body_angles[1])
        body_angles[2] = - (body_angles[2] - 90)
        body_angles[3] = 180 - body_angles[3]
        hands_list = util.handDetect(candidate, subset, img)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(img[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)

        canvas, is_left_hand_open, is_right_hand_open = util.draw_handpose(canvas, all_hand_peaks, False)
        #cv2.imwrite(pth, canvas)
        print("Saved image {}".format(pth))
        lhand = 100 if is_left_hand_open else 0
        rhand = 100 if is_right_hand_open else 0
        return body_angles, [lhand, rhand]
    return None, None

def on_press(key):
    global record
    if key.char == "r":
        record = True
        time.sleep(2)
        record = False
    if key.char == "s":
        print("stop detected")
        record = False

def on_release(key):
    pass

def mimic_direct(robot):
    while True:
        path = os.path.join("./dataset/img_direct.png")
        oriImg = robot.get_camera_frame(show=False)
        angles, hands = analyse_image(oriImg, path)
        move_robot(robot, angles, hands)

def collect_data_loop(robot, action_cfg):
    robot.subscribe_camera("camera_top", 2, 30)
    actions = list(action_cfg.keys())
    for a in actions:
        video_counter = 0
        for x in range(action_cfg[a]):
          print("PRESS 'R' TO RECORD")
          print("SHOW ACTION: {}".format(a.upper()))
          while True:
            if record:
                 robot.record_video(name="{}_{}".format(a, video_counter), seconds=4)
                 video_counter += 1
                 print("DONE")
                 break

def save_poses_images(path):
    dirs = [x[0] for x in os.walk(path)]
    for dir in dirs[1:]:
        print(dir)
        all_files = glob.glob(os.path.join(dir, "./*.png"))
        pkls = glob.glob(os.path.join(dir, "*.pkl"))
        if len(pkls) > 0:
            print("Found poses in ", dir)
            continue
        indices = [int(os.path.basename(all_files[i]).split("_")[1].split(".")[0]) for i, x in enumerate(all_files)]
        files_sorted = [x[1] for x in sorted(zip(indices, all_files))][:]
        all_poses = [[], []]
        for f in files_sorted:
            im = cv2.imread(f)
            angles, hands = analyse_image(im, f.replace(".png", "_pose.png"))
            all_poses[0].append(angles)
            all_poses[1].append(hands)
        data_p = np.asarray([x + y for x, y in zip(all_poses[0], all_poses[1])])
        with open(os.path.join(dir, "poses.pkl"), "wb") as handle:
            pickle.dump(data_p, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sample_action(vae, label, robot=None, check=False):
    ok = False
    if check:
        while not ok:
            sample = vae.generate_samples(1, int(vae.classes[label]))[0].squeeze()
            ok = check_action_ok(sample.unsqueeze(0), [label])[0] == 1
    else:
        sample = vae.generate_samples(1, int(vae.classes[label]))[0].squeeze()
    if robot:
        for a in sample:
            move_robot(robot, a[:6]*180, None, label=label)


def eval_trained_reconstructions(model, robot=None):
    o, labels, data = vae_reconstruct(model)
    o = np.expand_dims(o, 0)
    if robot:
        for ind, x in enumerate(data):
                 for a in x:
                    move_robot(robot, a[:6], a[-2:]*180)
        for ind, x in enumerate(o.reshape(o.shape[0], o.shape[1], -1)):
            for i,a in enumerate(x):
                if not sum(a) == 0:
                  move_robot(robot, list(a[:6]*180),  list(a[6:]*180))
            time.sleep(1)

def get_robot(ip, port=9559):
    robot = Pepper(ip, port)
    s = robot.autonomous_life_service.getState()
    if s != "disabled":
        print("Disabling the autonomous life")
        robot.autonomous_life_service.setState("disabled")
    robot.stand()
    robot.set_english_language()
    return robot

def collect_data(robot):
    actions = {"dance":20, "wave":20, "fly":20}
    collect_data_loop(robot, actions)

def explore_dataset(path, robot):
    data = load_actions_from_path(path)
    for sample in data[0]:
        for a in sample:
            move_robot(robot, a[:6], a[-2:] * 180)

def extract_imgs_vid(path):
    all_files = glob.glob(os.path.join(path, "./**.avi"))
    for file in all_files:
        vidcap = cv2.VideoCapture(file)
        folder = file.replace(".avi", "")
        os.makedirs(folder)
        success, image = vidcap.read()
        count = 0
        blocked = random.randint(8,13)
        while success:
            if count >= blocked:
                cv2.imwrite(os.path.join(folder, "img_%d.png" % count), image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        print("Saved {}".format(folder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Specify run mode: record/train/eval/train_eval")
    parser.add_argument("-ip", "--robot_ip", help="Robot's IP address")
    parser.add_argument("-cfg", "--config", default="config.yml", help="Path to training config")
    args = parser.parse_args()
    if args.mode != "train":
        ip_address = args.robot_ip
        robot = get_robot(ip=ip_address)
    if args.mode == "record":
        record = False
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        # Show given actions to the robot according to instructions
        collect_data(robot)
        # then you need to transfer recorded data from the robot ("/home/nao/recordings/cameras") to your local dir. Then do:
        extract_imgs_vid("./path_to_local_collected_data_folder")
        save_poses_images("./path_to_local_collected_data_folder")
    else:
        vae = IncremVAE("./config.yml")
        if args.mode in ["train", "train_eval"]:
            train_vae_offline(vae)
        if args.mode in ["eval", "train_eval"]:
            models = [vae.runPath]
            for i, m in enumerate(models):
                  vae = load_model(m)
                  for ind, a in enumerate(["dance", "fly","wave"]):
                       robot.say("{}".format(a))
                       for x in range(1):
                           sample_action(vae, a, robot)

