# coding=utf-8
import os
from pathlib import Path
from queue import Queue
import argparse
from time import monotonic
import throttle

import cv2
import depthai
import numpy as np
from imutils.video import FPS

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--debug", action="store_true", help="Debug output"
)

parser.add_argument(
    "-e",
    "--enroll",
    type=str,
    default="",
    help="Name to enroll",
)

parser.add_argument(
    "-t",
    "--throttle",
    type=int,
    default=10,
    help="Seconds to throttle recognition alerts",
)

args = parser.parse_args()

debug = args.debug

@throttle.wrap(args.throttle, 1)
def face_detected(name):
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


databases = "databases"

if not os.path.exists(databases):
    os.mkdir(databases)


def create_db(face_frame, results):
    name = args.enroll
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_color = (0, 255, 0)
    line_type = 1
    try:
        with np.load(f"{databases}/{name}.npz") as db:
            db_ = [db[j] for j in db.files][:]
    except Exception as e:
        db_ = []
    db_.append(np.array(results))
    np.savez_compressed(f"{databases}/{name}", *db_)


def read_db(labels):
    for file in os.listdir(databases):
        filename = os.path.splitext(file)
        if filename[1] == ".npz":
            label = filename[0]
            labels.add(label)
    db_dic = {}
    for label in list(labels):
        with np.load(f"{databases}/{label}.npz") as db:
            db_dic[label] = [db[j] for j in db.files]
    return db_dic


class DepthAI:
    def __init__(
        self
    ):
        print("Loading pipeline...")
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 1
        self.lineType = 0

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        # ColorCamera
        print("Creating Color Camera...")
        self.cam = self.pipeline.createColorCamera()
        self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
        self.cam.setResolution(
            depthai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        self.cam.setInterleaved(False)
        self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

        self.cam_xout = self.pipeline.createXLinkOut()
        self.cam_xout.setStreamName("preview")
        self.cam.preview.link(self.cam_xout.input)

        self.create_nns()

        print("Pipeline created.")

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
        print(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first:
            print("linked cam.preview to model_nn.input")
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
        print(f"Creating {model_path} Neural Network...")
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
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")

        self.start_nns()

        self.preview = self.device.getOutputQueue(
            name="preview", maxSize=4, blocking=False
        )

    def start_nns(self):
        pass

    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None, line_type=None):
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
        if debug:
            self.debug_frame = self.frame.copy()

        s = self.parse_fun()
        # if s :
        #     raise StopIteration()
        if debug:
            cv2.imshow(
                "Camera_view",
                self.debug_frame,
            )
            self.fps_cam.update()
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.fps_cam.stop()
                self.fps_nn.stop()
                print(
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
        if len(args.enroll) == 0:
            self.db_dic = read_db(self.labels)

    def create_nns(self):

        # Require higher confidence for enroll to prevent capturing low-confidence face matches
        mobilenet_conf = 0.5 if len(args.enroll) == 0 else 1.0

        self.create_mobilenet_nn(
            #"models/face-detection-retail-0004_openvino_2021.2_4shave.blob",
            "models/face-detection-retail-0005_openvino_2021.4_4shave.blob",
            "mfd",
            first=True,
            conf=mobilenet_conf,
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
            if debug:
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

            if len(args.enroll) > 0:
                create_db(face_frame, results)
            else:
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
                # name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
                if conf[0][0] >= 0.6:
                    name = conf[0]
                    face_detected(name[1])
                else:
                    name = (1 - conf[0][0], "UNKNOWN")
                if debug:
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
