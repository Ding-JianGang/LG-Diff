export MODEL_DIR="/instrcut-pix2pix" # path to model
accelerate launch train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --json_path=/diffusion_re/thermal_diffusion/scp_file/coco/annotations/instances_train2017.json \
    --resolution=512 \
    --per_gpu_batch_size=4 --gradient_accumulation_steps=1 \
    --max_train_steps=150000 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=5e-5 --lr_warmup_steps=0 \
    --seed=1231 \
    --val_image_url_or_path=/diffusion_re/thermal_diffusion/output.png \
    --validation_prompt="Thermal Imaging." \
    --mixed_precision="fp16" \
    --validation_epochs=1 \
    --dataset_type="coco" \
    --output_dir=/diffusion_re/thermal_diffusion/coco_ther_res \
    --num_objects=1 \
    --pretrain_unet=/diffusion_re/thermal_diffusion/coco_ther_res/checkpoint-52000