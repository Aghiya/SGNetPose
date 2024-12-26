BATCH_SIZE=85
EPOCHS=30
INPUT_DIM=4
PRED_DIM=4
DEC_STEPS=45
SEED=100
TS=$(date +"%Y%m%d_%H%M%S")
RUN_DESC="angle_b2"
RUN_NUM=3
log_path="./logs/pie/${RUN_DESC}_${RUN_NUM}_${TS}.log"

source ~/miniconda3/bin/activate
conda activate sgnet

export PYTHONPATH=$PYTHONPATH:/home/aghiya/SGNet.pytorch

find . -type f -name "*.py" -exec cat {} + > /dev/null

screen -dmS pie_${RUN_DESC}_${RUN_NUM} -L -Logfile ${log_path} bash -c "python /home/aghiya/SGNet.pytorch/tools/pie/train_cvae.py --batch_size=${BATCH_SIZE} --epochs=${EPOCHS} --input_dim=${INPUT_DIM} --pred_dim=${PRED_DIM} --dec_steps=${DEC_STEPS} --seed=${SEED}"

sleep 5
tail -f ${log_path}

# python /home/aghiya/SGNet.pytorch/tools/pie/train_cvae.py --batch_size=${BATCH_SIZE} --epochs=${EPOCHS} --input_dim=${INPUT_DIM} --pred_dim=${PRED_DIM} --dec_steps=${DEC_STEPS} --seed=${SEED}
