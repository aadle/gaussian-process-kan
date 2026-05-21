import json
from datetime import datetime

def write_info_file(
    file_path,
    model_name,
    args,
    x,
    x_train,
    x_test,
    elapsed_training_time,
    elapsed_test_time,
    elapsed_pred_time,
    test_mse,
    full_mse,
):
    # Convert JAX array shapes to Python ints first
    total_samples = int(x.shape[0])
    train_samples = int(x_train.shape[0])
    test_samples = int(x_test.shape[0])
    train_pct = round(100 * train_samples / total_samples, 1)
    test_pct = round(100 * test_samples / total_samples, 1)
    
    info_dict = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_information": {
            "model_name": model_name,
            "architecture": getattr(args, "model_size", None),
            "inducing_points_per_activation": getattr(args, "n_inducing", None),
            "freeze_latent_grid": getattr(args, "freeze_x_latent", None),
        },
        "data_configuration": {
            "function": args.function,
            "total_samples": total_samples,
            "training_set": {
                "samples": train_samples,
                "percentage": train_pct,
            },
            "test_set": {
                "samples": test_samples,
                "percentage": test_pct,
            },
        },
        "training_configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "performance_metrics": {
            "training_time_seconds": round(elapsed_training_time, 6),
            "test_prediction_time_seconds": round(elapsed_test_time, 6),
            "full_prediction_time_seconds": round(elapsed_pred_time, 6),
            "total_inference_time_seconds": round(elapsed_test_time + elapsed_pred_time, 6),
        },
        "results": {
            "test_set_mse": round(test_mse, 6),
            "full_dataset_mse": round(full_mse, 6),
        },
    }
    
    with open(file_path, "w") as f:
        json.dump(info_dict, f, indent=4)
    
    print(f"Info file written to {file_path}")
