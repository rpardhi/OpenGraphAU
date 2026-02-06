#!/bin/bash
#SBATCH --job-name=megraph_sit_21_hu_video_mix_home
#SBATCH --output=megraph_sit_21_hu_video_mix_home_%j.out
#SBATCH --error=megraph_sit_21_hu_video_mix_home_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# activate your venv
conda activate open_graph_au 

# run one video for this array index
python AU_Extraction.py \
    --arc resnet50 \
    --stage 2 \
    --exp-name demo \
    --resume checkpoints/OpenGprahAU-ResNet50_second_stage.pth \
    --input /data/sit_21_hu_video_mix_home/processed/normalized \
    --outdir /data/sit_21_hu_video_mix_home/processed/features/megraph \
    --frames_divide 1
