# coding=utf-8
import os
from pathlib import Path
from queue import Queue
import argparse
from time import monotonic
import datetime
import throttle
import logging

import cv2
import depthai
import numpy as np
from imutils.video import FPS

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--preview", action="store_true",
                    help="preview camera")
parser.add_argument("-b", "--database-dir", type=str, default="databases",
                    help="path to save/load recognition databases (default: %(default)s)")
parser.add_argument("-n", "--no-enroll", action="store_true",
                    help="do not auto-enroll")
parser.add_argument("-t", "--throttle", type=int, default=10,
                    help="seconds to throttle recognition alerts (default: %(default)d)")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("-r", "--recognize-threshold", type=int, default=85,
                    help="confidence percentage to meet or exceed for recognition (default: %(default)d)")
parser.add_argument("-c", "--continue-threshold", type=int, default=75,
                    help="confidence percentage to meet or exceed to continue enroll (re-recognition) (default: %(default)d)")

args = parser.parse_args()

preview        = args.preview
noenroll       = args.no_enroll
throttle_secs  = args.throttle
db_dir         = args.database_dir
recognize_conf = args.recognize_threshold / 100
enroll_conf    = args.continue_threshold / 100

log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s')

@throttle.wrap(throttle_secs, 1)
def face_detected(name):
    logging.info(f"Face detected was called")
    print(f"{name}\a")

def to_planar(arr: np.ndarray, shape: tuple):
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()

def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())

def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    return x_out.tryGet()

def frame_norm(frame, *xy_vals):
    return (
        np.clip(np.array(xy_vals), 0, 1) * np.array(frame * (len(xy_vals) // 2))[::-1]
    ).astype(int)

def correction(frame, angle=None, invert=False):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1)
    affine = cv2.invertAffineTransform(mat).astype("float32")
    corr = cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if invert:
        return corr, affine
    return corr

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    similarity = np.dot(a, b.T) / (a_norm * b_norm)
    return similarity

def enroll(name, face_frame, results, labels, db_dic):

    if name in db_dic:
        db_ = db_dic[name]
        if len(db_) >= 200:
            logging.info(f"Not enrolling {name}, too many records.")
            return
        db_.append(np.array(results))
    else:
        labels.add(name)
        db_ = np.array(results)
        db_dic[name] = db_
        # Save an image if newly enrolling
        logging.info(f"Enrolling new face as {name}")
        cc = face_frame.copy()
        cv2.imwrite(f"{db_dir}/{name}.jpg", cc)

    if not os.path.exists(db_dir):
        os.mkdir(db_dir)
    np.savez_compressed(f"{db_dir}/{name}", *db_)

def read_db(labels):
    for file in os.listdir(db_dir):
        filename = os.path.splitext(file)
        if filename[1] == ".npz":
            label = filename[0]
            labels.add(label)
    db_dic = {}
    for label in list(labels):
        with np.load(f"{db_dir}/{label}.npz") as db:
            db_dic[label] = [db[j] for j in db.files]
    return db_dic


class DepthAI:
    def __init__(
        self
    ):
        logging.debug("Loading pipeline...")
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 1
        self.lineType = 0

    def create_pipeline(self):
        logging.debug("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        # ColorCamera
        logging.debug("Creating Color Camera...")
        self.cam = self.pipeline.createColorCamera()
        self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
        self.cam.setResolution(
            depthai.ColorCameraProperties.SensorResolution.THE_4_K
        )
        self.cam.setInterleaved(False)
        self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

        self.cam_xout = self.pipeline.createXLinkOut()
        self.cam_xout.setStreamName("preview")
        self.cam.preview.link(self.cam_xout.input)

        self.create_nns()

        logging.info("Pipeline created.")

    def create_nns(self):
        pass

    def create_nn(self, model_path: str, model_name: str, first: bool = False):
        """

        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        logging.debug(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first:
            logging.debug("linked cam.preview to model_nn.input")
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def create_mobilenet_nn(
        self,
        model_path: str,
        model_name: str,
        conf: float = 0.5,
        first: bool = False,
    ):
        """

        :param model_path: model name
        :param model_name: model abbreviation
        :param conf: confidence threshold
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        logging.debug(f"Creating {model_path} MobileNet Neural Network...")
        model_nn = self.pipeline.createMobileNetDetectionNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.setConfidenceThreshold(conf)
        model_nn.input.setBlocking(False)

        if first:
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        try:
            logging.info("Starting pipeline...")
            self.device = depthai.Device(self.pipeline)
        except Exception as e:
            logging.critical("Could not create pipeline: %s", str(e))
            exit(1)

        self.start_nns()

        self.preview = self.device.getOutputQueue(
            name="preview", maxSize=4, blocking=False
        )

    def start_nns(self):
        pass

    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None,
        line_type=None):
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )

    def draw_bbox(self, bbox, color):
        cv2.rectangle(
            img=self.debug_frame,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=2,
        )

    def parse(self):
        if preview:
            self.debug_frame = self.frame.copy()

        s = self.parse_fun()
        # if s :
        #     raise StopIteration()
        if preview:
            cv2.imshow(
                "Camera_view",
                self.debug_frame,
            )
            self.fps_cam.update()
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.fps_cam.stop()
                self.fps_nn.stop()
                logging.debug(
                    f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}"
                )
                raise StopIteration()

    def run_camera(self):
        while True:
            in_rgb = self.preview.tryGet()
            if in_rgb is not None:
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                self.frame = (
                    in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                )
                self.frame = np.ascontiguousarray(self.frame)
                try:
                    self.parse()
                except StopIteration:
                    break

    @property
    def cam_size(self):
        return self._cam_size

    @cam_size.setter
    def cam_size(self, v):
        self._cam_size = v

    def run(self):
        self.fps_cam.start()
        self.fps_nn.start()
        self.run_camera()
        del self.device


class Main(DepthAI):
    def __init__(self):
        self.cam_size = (300, 300)
        super(Main, self).__init__()
        self.face_frame_corr = Queue()
        self.face_frame = Queue()
        self.face_coords = Queue()
        self.labels = set()
        self.db_dic = read_db(self.labels)

    def create_nns(self):

        self.create_mobilenet_nn(
            "models/face-detection-retail-0005_openvino_2021.4_4shave.blob",
            "mfd",
            first=True,
            conf=0.9, # Raised to prevent auto-enroll of non-faces
        )

        self.create_nn(
            "models/head-pose-estimation-adas-0001_openvino_2021.4_4shave.blob",
            "head_pose",
        )
        self.create_nn(
            "models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob",
            "arcface",
        )

    def start_nns(self):
        self.mfd_nn = self.device.getOutputQueue("mfd_nn", 4, False)
        self.head_pose_in = self.device.getInputQueue("head_pose_in", 4, False)
        self.head_pose_nn = self.device.getOutputQueue("head_pose_nn", 4, False)
        self.arcface_in = self.device.getInputQueue("arcface_in", 4, False)
        self.arcface_nn = self.device.getOutputQueue("arcface_nn", 4, False)

    def run_face_mn(self):
        nn_data = self.mfd_nn.tryGet()
        if nn_data is None:
            return False

        bboxes = nn_data.detections
        for bbox in bboxes:
            face_coord = frame_norm(
                self.frame.shape[:2], *[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            )
            self.face_frame.put(
                self.frame[face_coord[1] : face_coord[3], face_coord[0] : face_coord[2]]
            )
            self.face_coords.put(face_coord)
            if preview:
                self.draw_bbox(face_coord, (10, 245, 10))

        return True

    def run_head_pose(self):
        while self.face_frame.qsize():
            face_frame = self.face_frame.get()
            nn_data = run_nn(
                self.head_pose_in,
                self.head_pose_nn,
                {"data": to_planar(face_frame, (60, 60))},
            )
            if nn_data is None:
                return False

            out = np.array(nn_data.getLayerFp16("angle_r_fc"))
            self.face_frame_corr.put(correction(face_frame, -out[0]))

        return True

    def run_arcface(self):
        while self.face_frame_corr.qsize():
            face_coords = self.face_coords.get()
            face_frame = self.face_frame_corr.get()

            nn_data = run_nn(
                self.arcface_in,
                self.arcface_nn,
                {"data": to_planar(face_frame, (112, 112))},
            )

            if nn_data is None:
                return False
            self.fps_nn.update()
            results = to_nn_result(nn_data)

            conf = []
            max_ = 0
            label_ = None
            for label in list(self.labels):
                for j in self.db_dic.get(label):
                    conf_ = cosine_distance(j, results)
                    if conf_ > max_:
                        max_ = conf_
                        label_ = label
            conf.append((max_, label_))
            
            name = conf[0]

            if name[0] >= recognize_conf:
                # Use debug log level to minimize screen scroll
                logging.debug(f"Face detected: {name[1]}; confidence: {name[0] * 100:.2f}%\a")
                face_detected(name[1])

            if name[0] >= enroll_conf:
                logging.info(f"Face detected, updating enrolment: {name[1]}; confidence: {name[0] * 100:.2f}%\a")
                enroll(name[1], face_frame, results, self.labels,
                       self.db_dic)

            if name[0] < 0.2:
                logging.info(f"Face detected, enrolling: {name[1]}; confidence: {name[0] * 100:.2f}%\a")
                enroll(datetime.datetime.now().isoformat().replace(':','_'),
                    face_frame, results, self.labels, self.db_dic)

            if preview:
                self.put_text(
                    f"name:{name[1]}",
                    (face_coords[0], face_coords[1] - 35),
                    (244, 0, 255),
                )
                self.put_text(
                    f"conf:{name[0] * 100:.2f}%",
                    (face_coords[0], face_coords[1] - 10),
                    (244, 0, 255),
                )

        return True

    def parse_fun(self):
        if self.run_face_mn():
            if self.run_head_pose():
                if self.run_arcface():
                    return True


if __name__ == "__main__":
    Main().run()
