# Define an array of prefixes

# dataset="pie"
dataset="jaad"

if [[ "${dataset}" == "pie" ]];
then
    prefixes=("base_noaug" 
              # "angle_noaug_to_enc" 
              # "angle_noaug_to_dec" 
              # "angle_noaug_to_enc_dec" 
              # "pose_noaug_to_dec" 
              # "pose_noaug_to_enc"
              # "pose_noaug_to_enc_dec"
              # "anglepose_noaug_to_enc"
              # "anglepose_noaug_to_dec"
              # "anglepose_noaug_to_enc_dec"
              "pose_noaug_interleave_128embed_seq1_drop01_to_dec"
              "angle_b2"
              # "bbox_b2"
             )
else
    prefixes=("base_aug" 
          # "angle_aug_to_enc" 
          # "angle_aug_to_dec"
          # "angle_aug_to_enc_dec"
          # "pose_aug_to_enc"
          # "pose_aug_to_dec" 
          # "pose_aug_to_enc_dec"
          # "anglepose_aug_to_enc"
          # "anglepose_aug_to_dec"
          # "anglepose_aug_to_enc_dec"
          # "pose_2augX_interleave_128embed_rnn_drop01_to_dec"
          "pose_2augX_interleave_128embed_rnn_seq1_drop01_to_dec"
          # "pose_2augX_interleave_rnn_seq4_drop01_to_dec"
          # "angle_2augX_interleave_128embed_rnn_drop01_to_dec"
          "angle_2augX_interleave_128embed_rnn_seq1_drop01_to_dec"
          # "angle_2augX_interleave_rnn_seq4_drop01_to_dec"
          # "anglepose_2augX_interleave_rnn_seq4_drop01_to_dec"
          # "bbox_b2"
          )
fi

SGNET_DIR='/home/aghiya/SGNet.pytorch'
LOG_DIR="${SGNET_DIR}/logs/${dataset}"
CHART_DIR="${SGNET_DIR}/charts/${dataset}"

source ~/miniconda3/bin/activate
conda activate sgnet

# Convert the array into space-separated string and pass it to the Python script
python ${SGNET_DIR}/compute_loss.py --directory ${LOG_DIR} --prefixes "${prefixes[@]}" --output_dir ${CHART_DIR}
