import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import time


class CarDataProcessor:
    """Process and prepare car data for the recommendation system"""

    def __init__(self, csv_path: str, cache_dir: str = "../data_cache"):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "car_data_cache.pkl")

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Check for cache first
        if self.load_from_cache():
            print("Loaded processed data from cache")
        else:
            # The whole process is executed here
            print("Processing data from CSV...")
            start_time = time.time()
            self.df = pd.read_csv(csv_path)
            self.prepare_data()
            self.create_embeddings()
            self.save_to_cache()

            elapsed_time = time.time() - start_time
            print(f"Processed {len(self.df)} cars in {elapsed_time:.2f} seconds")

    def load_from_cache(self):
        # Load processed data from cache
        if not os.path.exists(self.cache_file):
            return False

        try:
            # Check if the CSV file is newer than the cache file
            cache_mtime = os.path.getmtime(self.cache_file)
            csv_mtime = os.path.getmtime(self.csv_path)

            if csv_mtime > cache_mtime:
                print("CSV file is newer than cache, reprocessing data...")
                return False

            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)

            # Extract data from cache
            self.df = cached_data["df"]
            self.car_embeddings = cached_data["car_embeddings"]
            self.model = cached_data["model"]
            self.dimension = cached_data["dimension"]

            # Recreate FAISS index (can't be pickled)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(self.car_embeddings).astype("float32"))

            return True

        except Exception as e:
            print(f"Error loading from cache: {e}")
            return False

    def save_to_cache(self):
        """Save processed data to cache"""
        try:
            cache_data = {
                "df": self.df,
                "car_embeddings": self.car_embeddings,
                "model": self.model,
                "dimension": self.dimension,
                # 
            }

            # Save to disk
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            print(f"Saved processed data to cache: {self.cache_file}")

        except Exception as e:
            print(f"Error saving to cache: {e}")

    def prepare_data(self):
        """Clean and prepare the car dataset"""
        # Convert price to millions for easier interpretation
        self.df["price_million"] = self.df["price"] / 1000000

        # Fill missing values
        self.df["seats"] = self.df["seats"].fillna(5.0)  # Most common in Vietnam
        self.df["type"] = self.df["type"].fillna("Unknown")
        self.df["gearbox"] = self.df["gearbox"].fillna("AT")  # Assume automatic
        self.df["fuel"] = self.df["fuel"].fillna("petrol")  # Most common in Vietnam
        self.df["color"] = self.df["color"].fillna("white")  # Most common
        self.df["origin"] = self.df["origin"].fillna("Unknown")

        # Calculate car age
        current_year = 2025  # You can adjust this to be dynamic
        self.df["age"] = current_year - self.df["manufacture_date"]

        # Create text descriptions for each car
        self.df["description"] = self.df.apply(self._create_car_description, axis=1)

    def _create_car_description(self, row):
        """Create a descriptive text for a car based on its attributes"""
        desc = f"{row['manufacture_date']} {row['brand']} {row['model']}, "

        if not pd.isna(row["type"]):
            desc += f"{row['type']}, "

        if not pd.isna(row["seats"]):
            desc += f"{int(row['seats'])} seats, "

        if not pd.isna(row["gearbox"]):
            transmission = "Automatic" if row["gearbox"] == "AT" else "Manual"
            desc += f"{transmission} transmission, "

        if not pd.isna(row["fuel"]):
            desc += f"{row['fuel']}, "

        if not pd.isna(row["color"]):
            desc += f"{row['color']}, "

        if not pd.isna(row["mileage_v2"]):
            desc += f"{int(row['mileage_v2'])} km, "

        desc += f"Price: {row['price_million']:.1f} million VND, "
        desc += f"Condition: {row['condition']}"

        if not pd.isna(row["origin"]):
            origin_map = {
                "Việt Nam": "Made in Vietnam",
                "Nhật Bản": "Japanese import",
                "Hàn Quốc": "Korean import",
                "Mỹ": "American import",
                "Nước khác": "Other import",
            }
            origin_desc = origin_map.get(row["origin"], row["origin"])
            desc += f", {origin_desc}"

        return desc

    def create_embeddings(self):
        """Create vector embeddings for all cars in the dataset"""
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        print(f"Generating embeddings for {len(self.df)} cars...")
        batch_size = 1000
        total_batches = (len(self.df) + batch_size - 1) // batch_size

        all_embeddings = []
        for i in range(0, len(self.df), batch_size):
            batch = self.df["description"].iloc[i : i + batch_size].tolist()
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            print(f"  Processed batch {i // batch_size + 1}/{total_batches}")

        # Stacks all the batch embedding arrays for FAISS index
        self.car_embeddings = np.vstack(all_embeddings) 

        # fast similarity search
        print("Creating FAISS index...")
        self.dimension = self.car_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.car_embeddings).astype("float32"))
        print("FAISS index created")
