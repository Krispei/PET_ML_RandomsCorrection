import random
import csv
import os
from datetime import datetime

def generate_optimization_params(num_configs=5, csv_filename="step1_optimization_search.csv"):
    """
    Generates random hyperparameter sets specifically for Step 1: Optimization.
    Focuses on Learning Rate, Batch Size, and Weight Decay.
    """
    
    fieldnames = [
        "timestamp", "trial_id", 
        "lr", "weight_decay", "momentum", "patience"
    ]

    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        for _ in range(num_configs):
            lr = round(10**random.uniform(-4, -1.3), 6)
            
            # 10**-6 to 10**-3
            weight_decay = round(10**random.uniform(-6, -3), 7)

            config = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trial_id": f"opt_{random.randint(1000, 9999)}",
                
                # The "Engine" Parameters
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": random.choice([0.9, 0.95, 0.99]), # Specific to SGD
                
                # Training Duration Guidelines
                "patience": 15 
            }
            
            writer.writerow(config)

if __name__ == "__main__":
    # Generate 10 sets of parameters to start your search
    generate_optimization_params(num_configs=10)
    print(f"\nOptimization parameters saved to step1_optimization_search.csv")