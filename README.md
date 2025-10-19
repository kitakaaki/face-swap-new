# face-wrap
State-of-the-Art Face-Swap Technique in Digital Video

requirements:
wget -O experiments/pretrained_models/GFPGANv1.3.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
pip install -r requirements.txt

arguments:
--source: Face to swap onto the target
--input: Input video
--output: Output video
--reference: Reference face for main character
--gfpgan: Use GFPGAN enhancement
ex: python swap-face.py --source data/source.jpg --reference data/reference.jpg --input data/input.mp4 --output results/output.mp4 --gfpgan