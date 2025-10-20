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
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # GPU 可改 ctx_id=0
app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace loaded.")

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Face Swap using InsightFace + GFPGAN + Reference Face")
    parser.add_argument('--source', type=str, required=True, help='Face to swap onto the target')
    parser.add_argument('--input', type=str, required=True, help='Input video')
    parser.add_argument('--output', type=str, required=True, help='Output video')
    parser.add_argument('--reference', type=str, required=True, help='Reference face for main character')
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
# 換臉函數（加主角辨識 + GFPGAN 兼容新版）
# ----------------------------
def swap_faces(source_img, target_img, ref_embedding, gfpgan=None, alpha=0.7, threshold=0.3):
    """
    將 source_img 臉換到 target_img 中，並檢查是否符合主角 (ref_embedding)
    支援 GFPGAN enhancement，並自動轉換為 numpy.ndarray
    """
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
            continue  # 不符合主角，跳過

        target_landmarks = t_face.landmark_2d_106
        hull_index = cv2.convexHull(target_landmarks.astype(np.int32), returnPoints=False)
        M, _ = cv2.estimateAffinePartial2D(source_landmarks[hull_index[:,0]], target_landmarks[hull_index[:,0]])
        warped_source = cv2.warpAffine(source_img, M, (target_img.shape[1], target_img.shape[0]))

        mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(target_landmarks.astype(np.int32)), 255)

        # GFPGAN 修復（新版 API + CPU/GPU 自動切換）
        if gfpgan is not None:
            restored_faces, restored_img, _ = gfpgan.enhance(
                warped_source,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            # 確保 restored_img 是 numpy.ndarray
            if not isinstance(restored_img, np.ndarray):
                restored_img = np.array(restored_img)
            warped_source = restored_img

        # alpha 混合
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        swapped = swap_faces(source_img, frame, ref_embedding, gfpgan=gfpgan, threshold=threshold)
        out.write(swapped)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print("Video processing finished.")

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
        print("Loading GFPGAN...")

        # 自動偵測 GPU 是否可用且 Compute Capability >= 7.0
        use_gpu = False
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            major_cc, minor_cc = torch.cuda.get_device_capability(0)
            print(f"Found GPU: {device_name} with Compute Capability {major_cc}.{minor_cc}")
            if major_cc >= 7:
                use_gpu = True
            else:
                print("GPU Compute Capability < 7.0, GFPGAN will use CPU instead.")
        device_str = 'cuda' if use_gpu else 'cpu'

        gfpgan_model = GFPGANer(
            model_path='experiments/pretrained_models/GFPGANv1.3.pth',
            upscale=1,
            arch='clean',
            device=device_str
        )
        print(f"GFPGAN loaded on {device_str.upper()}.")

    process_video(source_img, args.input, args.output, ref_embedding, gfpgan=gfpgan_model, threshold=args.threshold)

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


