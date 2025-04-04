import pandas as pd

def clean_car_data(input_csv_path, output_csv_path, min_price=200000000.0):
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    initial_count = len(df)
    print(f"Initial number of records: {initial_count}")
    
    # Filter out entries with price below the minimum threshold
    df_price_filtered = df[df['price'] >= min_price]
    price_filtered_count = len(df_price_filtered)
    price_removed_count = initial_count - price_filtered_count
    price_removed_percentage = (price_removed_count / initial_count) * 100
    
    print(f"Entries with price below {min_price/1000000:.1f} million VND removed: {price_removed_count} ({price_removed_percentage:.2f}%)")
    
    # Get columns that define a unique car (all except id, list_id, list_time)
    car_feature_columns = [col for col in df.columns if col not in ['id', 'list_id', 'list_time']]
    
    # Drop duplicates based on car features only
    df_cleaned = df_price_filtered.drop_duplicates(subset=car_feature_columns, keep='first')
    
    duplicate_removed_count = price_filtered_count - len(df_cleaned)
    duplicate_removed_percentage = (duplicate_removed_count / price_filtered_count) * 100
    
    print(f"Duplicate records removed: {duplicate_removed_count} ({duplicate_removed_percentage:.2f}%)")
    
    final_count = len(df_cleaned)
    total_removed_count = initial_count - final_count
    total_removed_percentage = (total_removed_count / initial_count) * 100
    
    print(f"Total records removed: {total_removed_count} ({total_removed_percentage:.2f}%)")
    print(f"Final number of records: {final_count}")

    df_cleaned.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to {output_csv_path}")
    
    return df_cleaned

if __name__ == "__main__":
    input_path = "car_data.csv"  
    output_path = "car.csv"
    
    clean_car_data(input_path, output_path, min_price=200000000.0)
