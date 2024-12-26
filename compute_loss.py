import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

model_map = { "base_noaug": "Baseline",
              "angle_noaug_to_enc": "A1",
              "angle_noaug_to_dec": "A3",
              "angle_noaug_to_enc_dec": "A5",
              "pose_noaug_to_dec": "A1",
              "pose_noaug_to_enc": "A3",
              "pose_noaug_to_enc_dec": "A5",
              "anglepose_noaug_to_enc": "A2",
              "anglepose_noaug_to_dec": "A4",
              "anglepose_noaug_to_enc_dec": "A6",
              "base_aug": "Baseline",
              "angle_aug_to_enc": "A1",
              "angle_aug_to_dec": "A3",
              "angle_aug_to_enc_dec": "A5",
              "pose_aug_to_dec": "A1",
              "pose_aug_to_enc": "A3",
              "pose_aug_to_enc_dec": "A5",
              "anglepose_aug_to_enc": "A2",
              "anglepose_aug_to_dec": "A4",
              "anglepose_aug_to_enc_dec": "A6",
              "pose_2augX_interleave_128embed_rnn_drop01_to_dec": "B1",
              "pose_2augX_interleave_128embed_rnn_seq1_drop01_to_dec": "Pose",
              "pose_2augX_interleave_rnn_seq4_drop01_to_dec": "B3",
              "angle_2augX_interleave_128embed_rnn_drop01_to_dec": "B1",
              "angle_2augX_interleave_128embed_rnn_seq1_drop01_to_dec": "Angle",
              "angle_2augX_interleave_rnn_seq4_drop01_to_dec": "B3",
              "anglepose_2augX_interleave_rnn_seq4_drop01_to_dec": "B4",
              "pose_noaug_interleave_128embed_seq1_drop01_to_dec": "Pose",
              "bbox_b2": "Bbox",
              "angle_b2": "Angle"
            }

def extract_loss_from_file(file_path):
    # Dictionary to store metrics for each prefix (e.g., MSE_05, MSE_10, etc.)
    metrics = {}

    # Read the log file and extract the relevant data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Ensure the line starts with "MSE" before processing
            if line.startswith("Test Loss"):
                # Split each pair by colon to separate metric name and its value
                metric, value = line.split(':')
                metric = metric.strip()  # Remove leading/trailing whitespace
                value = float(value.strip())  # Convert value to float

                # Store the metric value in the dictionary
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(value)  # Append values for the loss

    return metrics

def process_files(directory, prefixes):
    # Dictionary to store metrics per prefix
    all_metrics = {prefix: {} for prefix in prefixes}  # Initialize empty dict for each prefix

    # Iterate through the prefixes
    for prefix in prefixes:
        # Set the regex pattern for each prefix individually
        pattern = r"^" + prefix + r"_\d_.*\.log$"  # Updated pattern as per your request

        # Get the files matching the pattern for this specific prefix
        files = [f for f in os.listdir(directory) if re.match(pattern, f)]

        # List to store the metrics from all files for this prefix
        prefix_metrics = {metric: [] for metric in ['Test Loss']}

        for file in files:

            # Extract metrics from the file
            file_metrics = extract_loss_from_file(os.path.join(directory, file))
            
            # Store values for each metric in the list of corresponding files for this prefix
            for metric_name in file_metrics:
                if metric_name in prefix_metrics:
                    prefix_metrics[metric_name].append(file_metrics[metric_name])
                else:
                    prefix_metrics[metric_name] = [file_metrics[metric_name]]

        # Add the list of metrics for each prefix to the all_metrics dictionary
        all_metrics[prefix] = prefix_metrics

    return all_metrics

def average_metrics(all_metrics):
    averaged_metrics = {}

    # For each prefix, calculate the average over all files for each metric
    for prefix, metrics in all_metrics.items():
        prefix_averages = {}
        
        # For each metric, average the values across all files
        for metric_name, metric_values in metrics.items():
            # Stack the lists (i.e., values for each file) and compute the average across files
            if metric_values:
                # Ensure each metric has the same number of elements across files (e.g., 50 values per file)
                metric_array = np.array(metric_values)
                if len(metric_array.shape) > 1:
                    # Compute the mean across rows (files)
                    averages = np.mean(metric_array, axis=0)
                    prefix_averages[metric_name] = averages
            else:
                prefix_averages[metric_name] = []

        averaged_metrics[prefix] = prefix_averages

    return averaged_metrics

def plot_metrics(averaged_metrics, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pfx_list = list(averaged_metrics.keys())
    
    metrics_list = list(averaged_metrics[pfx_list[0]].keys())
    
    dataset = output_dir.split('/')[-1]
    
    for metric in metrics_list:
        
        plt.figure()
        
        plt.xlabel("Epochs")
        plt.ylabel("Averaged value")
        plt.title(metric)
        
        for pfx in pfx_list:
            legend_label = f"{'' if model_map[pfx] == 'Baseline' else ''} {model_map[pfx]}".strip()
            plt.plot(range(1, len(averaged_metrics[pfx][metric]) + 1), averaged_metrics[pfx][metric], label=legend_label)
            
        plt.legend()
        plt.grid(True)
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, f"{dataset}_{metric.lower().replace(' ', '_')}.png"))
        plt.close()

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process log files and compute metrics.")
    parser.add_argument('--directory', type=str, required=True, help="Directory containing log files.")
    parser.add_argument('--prefixes', type=str, nargs='+', required=True, help="List of prefixes.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save plots.")
    args = parser.parse_args()

    # Process files and get metrics
    all_metrics = process_files(args.directory, args.prefixes)
    
    # Average metrics across all files per prefix
    averaged_metrics = average_metrics(all_metrics)
    
    # import pdb; pdb.set_trace()
    
    # Plot and save metrics as charts
    plot_metrics(averaged_metrics, args.output_dir)

if __name__ == '__main__':
    main()
