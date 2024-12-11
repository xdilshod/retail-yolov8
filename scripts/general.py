import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    original_shape = image.shape[:2]
    resized_image = cv2.resize(image, (640, 640))
    input_data = resized_image / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    return input_data, original_shape

def preprocess_video(video_path, input_size=(640, 640)):
    """
    Preprocess video frames for YOLO model inference.
    Arguments:
        video_path: Path to the input video.
        input_size: Tuple (width, height) for resizing frames.
    Returns:
        frame_generator: Generator that yields frames and preprocessed tensors.
        total_frames: Total number of frames in the video.
        original_shape: Shape (height, width) of the original video frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def frame_generator():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, input_size)
            normalized_frame = resized_frame / 255.0
            input_data = np.transpose(normalized_frame, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            yield frame, input_data
        cap.release()

    return frame_generator(), total_frames, original_shape

def iou(box1, box2):
    """Compute IoU (Intersection over Union) between two boxes."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    iou_score = inter_area / (box1_area + box2_area - inter_area)
    return iou_score

def nms(bboxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to eliminate redundant boxes.
    Arguments:
        bboxes: List of bounding boxes with confidence scores.
        iou_threshold: IoU threshold to suppress overlapping boxes.
    Returns:
        List of filtered bounding boxes.
    """
    bboxes = sorted(bboxes, key=lambda x: x["confidence"], reverse=True)

    filtered_boxes = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        filtered_boxes.append(chosen_box)
        bboxes = [
            box for box in bboxes
            if iou(chosen_box, box) < iou_threshold
        ]

    return filtered_boxes

def postprocess_predictions(predictions, original_shape, confidence_threshold=0.3, iou_threshold=0.3):
    """
    Postprocess the YOLOv8 model predictions with NMS.
    Arguments:
        predictions: Raw predictions from the model.
        original_shape: Tuple (height, width) of the original image.
        confidence_threshold: Minimum confidence score to keep a prediction.
        iou_threshold: IoU threshold for Non-Maximum Suppression.
    Returns:
        List of filtered bounding boxes.
    """
    predictions = np.squeeze(predictions)
    bboxes = []
    orig_h, orig_w = original_shape
    input_size = 640

    for i in range(predictions.shape[1]):
        x_center = predictions[0, i]
        y_center = predictions[1, i]
        width = predictions[2, i]
        height = predictions[3, i]
        confidence = predictions[4, i]

        if confidence < confidence_threshold:
            continue

        x_center = x_center * orig_w / input_size
        y_center = y_center * orig_h / input_size
        width = width * orig_w / input_size
        height = height * orig_h / input_size

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        bboxes.append({
            "x1": max(0, x1),
            "y1": max(0, y1),
            "x2": min(orig_w, x2),
            "y2": min(orig_h, y2),
            "confidence": float(confidence),
        })

    bboxes = nms(bboxes, iou_threshold=iou_threshold)
    return bboxes

def draw_predictions(image_path, predictions, output_path="output.jpg"):
    image = cv2.imread(image_path)
    for pred in predictions:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_path, image)

def save_video_with_predictions(input_path, output_path, predictions_per_frame):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame, predictions in predictions_per_frame:
        for bbox in predictions:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
