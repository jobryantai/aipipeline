import os
import cv2
from src.utils.face_analysis_diy import FaceAnalysisDIY
from stage4_stylized_avatar import run_stylized_avatar_pipeline

def extract_reference_face(video_path, output_path="speaker_refs/speaker_0.jpg"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    fa = FaceAnalysisDIY(name='buffalo_l', root='pretrained_weights/insightface')
    fa.prepare(ctx_id=0, det_size=(512, 512))
    
    frame_idx = 0
    success = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = fa.get(frame)
        if len(faces) > 0:
            cv2.imwrite(output_path, frame)
            print(f"Successfully extracted reference face from frame #{frame_idx} -> {output_path}")
            success = True
            break
        frame_idx += 1
        
    cap.release()
    if not success:
        raise RuntimeError(f"Could not find any face in video: {video_path}")

if __name__ == "__main__":
    video_file = "test_segment.mp4"
    print(f"Scanning {video_file} for a valid face frame...")
    extract_reference_face(video_file)
    
    print("Running Stage 4 LivePortrait Avatar Animation...")
    run_stylized_avatar_pipeline(
        segments_json_path="segments_diarized.json", 
        speaker_refs_dir="speaker_refs/", 
        output_dir="pipeline_output/stage4_final"
    )
