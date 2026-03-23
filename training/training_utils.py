from datetime import datetime

def write_info_file(file_path, model_name, args, elapsed_training_time, 
                    elapsed_test_time, elapsed_pred_time, test_mse, full_mse):
    info_content = []
    
    # Header with timestamp
    info_content.append("=" * 80)
    info_content.append("gaussian process kolmogorov arnold network - run information")
    info_content.append("=" * 80)
    info_content.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ==================== MODEL INFORMATION ====================
    info_content.append("MODEL INFORMATION")
    info_content.append("-" * 80)
    info_content.append(f"Model + Function: {model_name}")
    info_content.append(f"Architecture: {args.model_size}")
    info_content.append(f"Inducing Points per Activation: {args.n_inducing}")
    info_content.append(f"Freeze Latent Grid (x_latent): {args.freeze_x_latent}\n")
    
    # ==================== DATA CONFIGURATION ====================
    info_content.append("DATA CONFIGURATION")
    info_content.append("-" * 80)
    info_content.append(f"Function: {args.function}")
    # info_content.append(f"Number of Samples: {args.n_samples}")
    # info_content.append(f"Test Set Size: {args.test_size} ({int(args.n_samples * args.test_size)} samples)")
    # info_content.append(f"Training Set Size: {int(args.n_samples * (1 - args.test_size))} samples\n")
    
    # ==================== TRAINING CONFIGURATION ====================
    info_content.append("TRAINING CONFIGURATION")
    info_content.append("-" * 80)
    info_content.append(f"Epochs: {args.epochs}")
    info_content.append(f"Batch Size: {args.batch_size}")
    info_content.append(f"Learning Rate: {args.learning_rate}")
    # info_content.append(f"Random Seed (Key): {args.key}\n")
    
    # ==================== PERFORMANCE METRICS ====================
    info_content.append("PERFORMANCE METRICS")
    info_content.append("-" * 80)
    info_content.append(f"Training Time: {elapsed_training_time:.6f} seconds")
    info_content.append(f"Test Prediction Time: {elapsed_test_time:.6f} seconds")
    info_content.append(f"Full Prediction Time: {elapsed_pred_time:.6f} seconds")
    info_content.append(f"Total Inference Time: {elapsed_test_time + elapsed_pred_time:.6f} seconds\n")
    
    # ==================== RESULTS ====================
    info_content.append("RESULTS")
    info_content.append("-" * 80)
    info_content.append(f"Test Set MSE: {test_mse:.6f}")
    info_content.append(f"Full Dataset MSE: {full_mse:.6f}")
    
    # Write to file
    with open(file_path, "w") as f:
        f.write("\n".join(info_content))
    
    print(f"Info file written to {file_path}")
