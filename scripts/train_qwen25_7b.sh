export WANDB_PROJECT="your_project_name"
export WANDB_LOG_MODEL="false"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <RUNNAME> <DATAPATH>"
    exit 1
fi

RUNNAME="$1"
MODELPATH="Qwen/Qwen2.5-7B"
DATAPATH="$2"
MODEL_SIZE="7B"
OUTPUTPATH="./alpaca_gpt4_outputs"
DEVICES="0,1,2,3"  # e.g. 0,1,2,3
NUM_GPUS=$(echo $DEVICES | awk -F',' '{print NF}')
TOTALBSZ=128
BSZPERDEV=1
GRADACC=$(($TOTALBSZ/$NUM_GPUS/$BSZPERDEV))
echo "Training model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BSZPERDEV batch size per GPU, $GRADACC gradient accumulation steps"

LOGFILE="train.log"

accelerate launch train/train.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 1500 \
    --gradient_checkpointing True \
    --save_strategy "no" \
    --save_steps 1500 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --evaluation_strategy "no" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --conv_template "vicuna_v1.1" \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    2>&1 | tee "$LOGFILE"

python finalize_tokenizer.py --model_path ${OUTPUTPATH}/${RUNNAME} --model_type qwen