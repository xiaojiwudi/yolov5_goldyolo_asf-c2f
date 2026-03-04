import cv2
import torch
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, check_img_size
from utils.torch_utils import select_device

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将检测框从模型输入图像的尺寸映射到原始图像的尺寸。
    Args:
        img1_shape (tuple): 模型输入图像的尺寸 (height, width)
        coords (torch.Tensor): 检测框坐标 [x1, y1, x2, y2]
        img0_shape (tuple): 原始图像的尺寸 (height, width)
        ratio_pad (tuple, optional): 缩放比例和填充 (ratio, pad)
    Returns:
        torch.Tensor: 映射到原始图像的检测框坐标
    """
    if ratio_pad is None:  # 默认计算缩放比例和填充
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 填充
    else:
        gain, pad = ratio_pad

    coords[:, [0, 2]] -= pad[0]  # x 坐标减去填充
    coords[:, [1, 3]] -= pad[1]  # y 坐标减去填充
    coords[:, :4] /= gain  # 缩放到原始尺寸
    coords[:, :4] = coords[:, :4].clamp(min=0, max=max(img0_shape))  # 限制坐标范围
    return coords

class CameraDetect:
    def __init__(self, weights, device="cpu", imgsz=640, conf_thres=0.25, iou_thres=0.45):
        """
        初始化摄像头检测类
        Args:
            weights (str): YOLOv5模型权重文件路径
            device (str): 使用的设备（如 'cpu' 或 'cuda:0'）
            imgsz (int): 输入图像大小
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值
        """
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        self.imgsz = check_img_size(imgsz, s=self.model.stride)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = self.model.names

    def preprocess(self, img):
        """
        图像预处理
        Args:
            img (np.ndarray): 输入图像
        Returns:
            torch.Tensor: 预处理后的图像
        """
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 归一化到 [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, frame):
        """
        检测目标
        Args:
            frame (np.ndarray): 输入图像帧
        Returns:
            np.ndarray: 带检测结果的图像帧
        """
        img = self.preprocess(frame)
        pred = self.model(img, augment=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f"{self.names[int(cls)]} {conf:.2f}"
                    self.plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)
        return frame

    @staticmethod
    def plot_one_box(x, img, color=(255, 0, 0), label=None, line_thickness=3):
        """
        绘制检测框
        Args:
            x (list): 检测框坐标 [x1, y1, x2, y2]
            img (np.ndarray): 图像
            color (tuple): 检测框颜色
            label (str): 标签
            line_thickness (int): 线条粗细
        """
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        if label:
            font_thickness = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, (255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)

def main():
    print("[INFO] 初始化摄像头检测")
    weights_path = "/home/sb/桌面/yolo模型改进/yolo模型/yolov5/runs/train/exp/weights/best.pt"
    detector = CameraDetect(weights=weights_path, device="cuda:0", imgsz=640)

    cap = cv2.VideoCapture(6)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("[INFO] 摄像头已打开，按 'q' 键退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法读取摄像头帧")
            break

        # 检测目标
        frame = detector.detect(frame)

        # 显示结果
        cv2.imshow("YOLOv5 Detection", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
