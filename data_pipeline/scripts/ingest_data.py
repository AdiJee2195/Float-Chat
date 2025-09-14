# Path: data_pipeline/scripts/ingest_data.py
import os
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine, text

# Set the base directory relative to the script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')

def process_argo_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".nc"):
            filepath = os.path.join(directory, filename)
            try:
                with xr.open_dataset(filepath) as ds:
                    df = ds.to_dataframe()
                    df.columns = [col.lower() for col in df.columns]
                    required_cols = ['latitude', 'longitude', 'pres', 'temp', 'psal']
                    if not all(col in df.columns for col in required_cols):
                        print(f"Skipping {filename} due to missing columns.")
                        continue
                    df['platform_number'] = ds.attrs.get('platform_number', 'UNKNOWN')
                    df.rename(columns={'pres': 'pressure', 'temp': 'temperature', 'psal': 'salinity'}, inplace=True)
                    df.reset_index(inplace=True)
                    df.rename(columns={'TIME': 'time'}, inplace=True)
                    df = df[['platform_number', 'time', 'latitude', 'longitude', 'pressure', 'temperature', 'salinity']]
                    all_data.append(df)
            except Exception as e:
                print(f"Could not process file {filename}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def create_db_engine():
    db_user = 'your_user'
    db_password = 'your_password'
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'argo_data'
    return create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

def main():
    print(f"Processing NetCDF files from: {RAW_DATA_DIR}")
    argo_df = process_argo_files(RAW_DATA_DIR)
    if argo_df.empty:
        print("No data was processed. Exiting.")
        return
    engine = create_db_engine()
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS argo_measurements (
                id SERIAL PRIMARY KEY, platform_number VARCHAR(255), time TIMESTAMP,
                latitude FLOAT, longitude FLOAT, pressure FLOAT, temperature FLOAT, salinity FLOAT
            );
        """))
        connection.commit()
        print(f"Inserting {len(argo_df)} rows into the database...")
        argo_df.to_sql('argo_measurements', engine, if_exists='append', index=False, method='multi')
    print("Data successfully ingested into PostgreSQL! 🎉")

if __name__ == '__main__':
    main()