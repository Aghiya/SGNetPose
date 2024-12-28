BATCH_SIZE=85
EPOCHS=30
SEED=0
POSE_DATA="skeleton" # alternate value is "angle"
TS=$(date +"%Y%m%d_%H%M%S")
RUN_DESC="sgnetpose_${POSE_DATA}"
RUN_NUM=1
LOG_DIR="./logs/pie"
CONDA_DIR="/home/aghiya" # conda installation folder
SGNETPOSE_DIR="/home/aghiya" # repo installation folder

LOG_PATH="${LOG_DIR}/${RUN_DESC}_${RUN_NUM}_${TS}.log"

# Create log dir if it doesn't exist
mkdir -p ${LOG_DIR}

source ${CONDA_DIR}/miniconda3/bin/activate
conda activate sgnet

export PYTHONPATH=$PYTHONPATH:/home/aghiya/jaadpie_pose

# clean up any cached files if running across multiple servers
find . -type f -name "*.py" -exec cat {} + > /dev/null

screen -dmS pie_${RUN_DESC}_${RUN_NUM} -L -Logfile ${LOG_PATH} bash -c "python ${SGNETPOSE_DIR}/SGNetPose/tools/pie/train_cvae.py --batch_size=${BATCH_SIZE} --epochs=${EPOCHS} --seed=${SEED}"

sleep 5
tail -f ${LOG_PATH}

# python ${SGNETPOSE_DIR}/SGNetPose/tools/pie/train_cvae.py --batch_size=${BATCH_SIZE} --epochs=${EPOCHS} --seed=${SEED}