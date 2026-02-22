################################################################################
# VIDEO FER + XAI DEMO – EXECUTION GUIDE (STABILIZED VERSION)
# ------------------------------------------------------------------------------
# This script performs Facial Emotion Recognition (FER) on a video file
# and overlays Grad-CAM / saliency maps onto detected face regions.
#
# This version includes:
# - Temporal prediction smoothing
# - Optional CAM smoothing
# - Label hold stabilization
#
# General structure:
#   python submission/demo/run_video_demo.py \
#       --video_path path/to/video.mp4 \
#       [OPTIONS]
################################################################################


################################################################################
# 1 DEFAULT: Ensemble with Majority Voting (Stabilized)
################################################################################

python submission/demo/run_video_demo.py \
  --video_path path/to/video.mp4 \
  --model ensemble \
  --models emocatnetsv2_nano emocatnetsv3_nano \
  --ensemble majority \
  --predict_every 3 \
  --smooth_alpha 0.7 \
  --smooth_cam_alpha 0 \
  --hold_label_frames 6 \
  --out_dir submission/demo/out


################################################################################
# 2 Ensemble with Soft Voting (Mean, Stabilized)
################################################################################

python submission/demo/run_video_demo.py \
  --video_path path/to/video.mp4 \
  --model ensemble \
  --models emocatnetsv2_nano emocatnetsv3_nano \
  --ensemble mean \
  --predict_every 3 \
  --smooth_alpha 0.7 \
  --smooth_cam_alpha 0 \
  --hold_label_frames 6 \
  --out_dir submission/demo/out


################################################################################
# 3 Ensemble with Explicit Saliency Model
################################################################################

python submission/demo/run_video_demo.py \
  --video_path path/to/video.mp4 \
  --model ensemble \
  --models emocatnetsv2_nano emocatnetsv3_nano \
  --ensemble mean \
  --saliency_model emocatnetsv3_nano \
  --predict_every 3 \
  --smooth_alpha 0.7 \
  --smooth_cam_alpha 0 \
  --hold_label_frames 6 \
  --out_dir submission/demo/out


################################################################################
# 4 Single Model (EmoCatNets v3, Stabilized)
################################################################################

python submission/demo/run_video_demo.py \
  --video_path path/to/video.mp4 \
  --model emocatnetsv3_nano \
  --predict_every 3 \
  --smooth_alpha 0.7 \
  --smooth_cam_alpha 0 \
  --hold_label_frames 6 \
  --out_dir submission/demo/out


################################################################################
# 5 Single Model (EmoCatNets v2, Stabilized)
################################################################################

python submission/demo/run_video_demo.py \
  --video_path path/to/video.mp4 \
  --model emocatnetsv2_nano \
  --predict_every 3 \
  --smooth_alpha 0.7 \
  --smooth_cam_alpha 0 \
  --hold_label_frames 6 \
  --out_dir submission/demo/out


################################################################################
# 6 Explicit Grad-CAM Layer (Recommended for Nano)
################################################################################

python submission/demo/run_video_demo.py \
  --video_path path/to/video.mp4 \
  --model emocatnetsv3_nano \
  --cam_layer stage3.5.depthwise_conv \
  --predict_every 3 \
  --smooth_alpha 0.7 \
  --smooth_cam_alpha 0 \
  --hold_label_frames 6 \
  --out_dir submission/demo/out


