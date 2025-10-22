import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
from numpy.linalg import norm
import torch

# ----------------------------
# 初始化 InsightFace
# ----------------------------
print("Loading InsightFace...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace loaded.")

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Face Swap using InsightFace + GFPGAN + Reference Face")
    parser.add_argument('--source', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/Ma.bmp"')
    parser.add_argument('--input', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/AI-muscle.mp4')
    parser.add_argument('--output', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/output.mp4')
    parser.add_argument('--reference', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/螢幕擷取畫面 2025-10-22 094048.png')
    parser.add_argument('--gfpgan', action='store_true', help='Use GFPGAN enhancement')
    parser.add_argument('--threshold', type=float, default=0.3, help='Cosine similarity threshold')
    return parser.parse_args()

# ----------------------------
# 載入影像
# ----------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image from {path}")
    return img

# ----------------------------
# 計算 cosine similarity
# ----------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

# ----------------------------
# 初始化 GFPGAN
# ----------------------------
def init_gfpgan():
    use_cuda = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    device = "cuda" if use_cuda else "cpu"
    print(f"Initializing GFPGAN on {device}...")

    try:
        gfpgan_model = GFPGANer(
            model_path='experiments/pretrained_models/GFPGANv1.3.pth',
            upscale=1,
            arch='clean',
            device=device
        )
        print(f"GFPGAN loaded on {device}.")
        return gfpgan_model
    except Exception as e:
        print(f"GFPGAN initialization failed: {e}")
        return None

# ----------------------------
# 換臉函數
# ----------------------------
def swap_faces(source_img, target_img, ref_embedding, gfpgan=None, alpha=0.7, threshold=0.3):

    source_faces = app.get(source_img)
    target_faces = app.get(target_img)
    if len(source_faces) == 0 or len(target_faces) == 0:
        return target_img

    source_face = source_faces[0]
    source_landmarks = source_face.landmark_2d_106

    output_img = target_img.copy()

    for t_face in target_faces:
        target_embedding = t_face.embedding
        sim = cosine_similarity(ref_embedding, target_embedding)
        if sim < threshold:
            continue

        target_landmarks = t_face.landmark_2d_106

        # 🟡 1️⃣ 額頭補點
        extra_points = []
        top_y = np.min(target_landmarks[:, 1])
        top_x = np.mean(target_landmarks[:, 0])
        extra_points.append([top_x, top_y - 40])
        extra_points.append([top_x - 30, top_y - 30])
        extra_points.append([top_x + 30, top_y - 30])

        # 合併原臉與額頭點
        all_landmarks = np.vstack([target_landmarks, np.array(extra_points)])

        # 🟢 2️⃣ 放大整體遮罩範圍
        center = np.mean(all_landmarks, axis=0)
        scale_x = 1.05
        scale_y = 1.15
        expanded_landmarks = (all_landmarks - center) * [scale_x, scale_y] + center

        # 🟣 3️⃣ 建立最終遮罩
        mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask,
                        cv2.convexHull(expanded_landmarks.astype(np.int32)),
                        255)

        # 🟠 4️⃣ 對齊臉部
        hull_index = cv2.convexHull(target_landmarks.astype(np.int32), returnPoints=False)
        M, _ = cv2.estimateAffinePartial2D(
            source_landmarks[hull_index[:, 0]],
            target_landmarks[hull_index[:, 0]]
        )
        warped_source = cv2.warpAffine(source_img, M, (target_img.shape[1], target_img.shape[0]))

        mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(target_landmarks.astype(np.int32)), 255)


        if gfpgan is not None:
            try:
                restored_faces, restored_img, _ = gfpgan.enhance(
                    warped_source,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True  # 將修復的人臉貼回原圖
                )
                if restored_img is not None and isinstance(restored_img, np.ndarray) and restored_img.size > 0:
                    warped_source = restored_img
            except Exception as e:
                print(f"GFPGAN enhance failed, fallback to original warp: {e}")
                # 失敗就使用原本 warp，不返回 None
                pass

        target_region = cv2.bitwise_and(output_img, output_img, mask=cv2.bitwise_not(mask))
        source_region = cv2.bitwise_and(warped_source, warped_source, mask=mask)
        blended = cv2.addWeighted(target_region, 1-alpha, source_region, alpha, 0)
        output_img[np.where(mask==255)] = blended[np.where(mask==255)]

    return output_img


# ----------------------------
# 影片處理
# ----------------------------
def process_video(source_img, input_path, output_path, ref_embedding, gfpgan=None, threshold=0.3):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        swapped = swap_faces(source_img, frame, ref_embedding, gfpgan=gfpgan, threshold=threshold)
        if swapped is None:
            cap.release()
            out.release()
            print("\nGFPGAN failed during video processing. Re-running without GFPGAN...")
            return False

        out.write(swapped)
        frame_idx += 1

        # 顯示百分比進度條
        progress = (frame_idx / total_frames) * 100
        bar_length = 30  # 進度條長度
        filled_length = int(bar_length * frame_idx // total_frames)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\rProgress: |{bar}| {progress:6.2f}% ({frame_idx}/{total_frames})", end='')

    cap.release()
    out.release()
    print("\nVideo processing finished.")
    return True


# ----------------------------
# 主程式
# ----------------------------
def main():
    args = parse_args()
    source_img = load_image(args.source)
    reference_img = load_image(args.reference)

    ref_faces = app.get(reference_img)
    if len(ref_faces) == 0:
        raise ValueError("Reference face not detected!")
    ref_embedding = ref_faces[0].embedding

    gfpgan_model = None
    if args.gfpgan:
        gfpgan_model = init_gfpgan()

    success = process_video(source_img, args.input, args.output, ref_embedding, gfpgan=gfpgan_model, threshold=args.threshold)

    # 如果 GFPGAN 整段影片失敗，重新跑一次 CPU-only
    if not success:
        print("Re-running video without GFPGAN...")
        process_video(source_img, args.input, args.output, ref_embedding, gfpgan=None, threshold=args.threshold)

if __name__ == "__main__":
    main()



    
# 【目標】  
# 【Objectives】
# Develop a Python program that performs face swap technique on a video using the
# InsightFace library. The user should be able to specify the source face image, input video,
# output video, and implement face swap technique with pipeline processes.

# 【Learning Goals】
# ⚫ Use Argparse to create a flexible command-line Python program.
# ⚫ Apply InsightFace for face detection and face swap.
# ⚫ Process video with OpenCV and generate output.
# ⚫ Understand and implement the AI multimedia pipeline using Python programming.

# 【Python 程式設計】
# Design a Python program to perform face swapping pipeline
# (Pipeline 是指從輸入到輸出「一條龍」完成)：
# python face_swap.py --source source.jpg --input input.mp4 --output output.mp4
# 你可以自定義其他參數 (Arguments)，例如：--help、--reference 等。
# 【Requirements】
# 請同學以組為單位，完成下列事項：
# ⚫ 使用 Argparse 處理參數。
# ⚫ 載入臉部偵測與換臉模型 (務必一次性載入)。
# ⚫ 以逐幀 (frame-by-frame) 方式處理數位視訊。
# (原則上，建議先使用單張影像進行程式設計與測試，影像範例如下圖)。
# ⚫ 偵測每一幀中的臉部區域。
# ⚫ 使用來源人臉影像進行換臉 (若臉部區域略超出畫面範圍，仍然須進行換臉)。
# ⚫ 對置換後的臉部區域使用 Real-ESRGAN 或 GFPGAN 技術增強細節，並貼回目標影
# 像。建議在換臉過程，顯示影像結果。
# ⚫ 將換臉後的影片儲存至--output 指定的路徑。
# ⚫ 若有音訊，則使用 FFmpeg 重新合成視訊與音訊 (使用 Subprocess)，並輸出視訊檔
# 案結果。

# 【Improvements】
# 請同學以組為單位，進一步增強下列程式功能：
# ⚫ 首先，於影片中擷取主角的正面臉部影像，作為參考臉部影像 (Reference Face
# Image)。
# ⚫ 使用 InsightFace 的人臉 embedding vector (嵌入特徵向量) 比對來辨識「主角」。
# ⚫ 只有當影片中偵測到的臉與 --reference 主角臉非常相似 (例如：cosine similarity
#  0.3) 時，才進行換臉，以確保不會誤套其他人臉。
# ⚫ 貼回臉部影像區域時，可嘗試採用 Alpha Blending 技術進行拼貼，使得臉部區域邊
# 緣區域較為貼合。
# ⚫ 使用 FFmpeg 或威力導演 (不要使用試用版，會有浮水印)，進行最後的調整


