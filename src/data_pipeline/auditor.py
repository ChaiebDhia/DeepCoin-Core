import os
import pandas as pd

def audit_data(raw_data_path):
    print(f"--- Auditing DeepCoin Dataset at {raw_data_path} ---")
    
    # We look specifically at the 'dataset_types' folder in CN_v1
    types_path = os.path.join(raw_data_path, 'dataset_types')
    
    if not os.path.exists(types_path):
        print("Error: Could not find 'dataset_types'. Make sure you unzipped the data!")
        return

    stats = []
    for coin_type in os.listdir(types_path):
        folder_path = os.path.join(types_path, coin_type)
        if os.path.isdir(folder_path):
            count = len(os.listdir(folder_path))
            stats.append({'type': coin_type, 'image_count': count})

    df = pd.DataFrame(stats)
    
    # Pro Analytics
    print(f"Total Unique Coin Types: {len(df)}")
    print(f"Total Images: {df['image_count'].sum()}")
    print("\nTop 5 Most Frequent Coins (Possible Bias):")
    print(df.nlargest(5, 'image_count'))
    print("\nBottom 5 Rarest Coins (Hard for AI to learn):")
    print(df.nsmallest(5, 'image_count'))

if __name__ == "__main__":
    # Update this path to where you unzipped the data
    audit_data('data/raw/CN_dataset_v1')