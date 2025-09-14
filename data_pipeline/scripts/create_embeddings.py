# Path: data_pipeline/scripts/create_embeddings.py
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

# Set the base directory relative to the script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

def get_metadata_from_db():
    db_user = 'your_user'
    db_password = 'your_password'
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'argo_data'
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    query = """
    SELECT platform_number, MIN(time) as start_time, MAX(time) as end_time,
           AVG(latitude) as avg_lat, AVG(longitude) as avg_lon
    FROM argo_measurements GROUP BY platform_number;
    """
    return pd.read_sql(query, engine)

def create_embeddings(metadata_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = metadata_df.apply(
        lambda row: f"ARGO float {row['platform_number']} was active from {row['start_time']} to {row['end_time']} around latitude {row['avg_lat']:.2f} and longitude {row['avg_lon']:.2f}.",
        axis=1
    ).tolist()
    return model.encode(descriptions, show_progress_bar=True)

def main():
    metadata = get_metadata_from_db()
    embeddings = create_embeddings(metadata)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))

    # Ensure the output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    index_path = os.path.join(PROCESSED_DATA_DIR, 'argo_metadata.index')
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'argo_metadata.csv')

    faiss.write_index(index, index_path)
    metadata.to_csv(csv_path, index=False)
    print(f"FAISS index and metadata file created successfully in {PROCESSED_DATA_DIR}! ✨")

if __name__ == '__main__':
    main()