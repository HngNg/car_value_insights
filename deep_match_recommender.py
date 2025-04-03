import pandas as pd
import numpy as np
import torch
import faiss
import json
from typing import List, Dict

from processors.car_data_processors import CarDataProcessor
from processors.user_needs_processor import UserNeedsProcessor
from ollama_client import OllamaClient
from matcher import PersonalizedMatcher


class DeepMatchRecommender:
    """LLM-powered car recommendation system"""

    def __init__(
        self,
        car_processor: CarDataProcessor,
        user_processor: UserNeedsProcessor,
        ollama_client: OllamaClient,
    ):
        self.car_processor = car_processor
        self.user_processor = user_processor
        self.ollama_client = ollama_client

        # MPS is for Apple Hardware Accelerators
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.matcher = PersonalizedMatcher(
            embedding_dim=self.car_processor.dimension, hidden_dim=128
        ).to(self.device)

    def _filter_by_constraints(
        self, user_profile: Dict, relax: bool = False
    ) -> pd.DataFrame:
        df = self.car_processor.df
        filters = []

        if "budget_max" in user_profile:
            budget = user_profile["budget_max"]
            max_price = budget * 1.15 if relax else budget
            filters.append(f"price_million <= {max_price}")

            # For luxury preference, set a minimum
            if user_profile.get("luxury_preference", False) and budget > 1000:
                min_price = budget * 0.4  # This is 40% of budget
                filters.append(f"price_million >= {min_price}")

        if "family_size" in user_profile:
            min_seats = user_profile["family_size"]
            if not relax:
                filters.append(f"seats >= {min_seats}")
            else:
                filters.append(f"seats >= {max(min_seats - 1, 2)}")

        if "car_type_preference" in user_profile:
            car_type = user_profile["car_type_preference"]
            if car_type and car_type != "any" and not relax:
                type_map = {
                    "sedan": "Sedan",
                    "suv": "SUV / Cross over",
                    "hatchback": "Hatchback",
                    "mpv": "Minivan (MPV)",
                    "pickup": "Pickup truck",
                }
                db_type = type_map.get(car_type.lower(), car_type)
                filters.append(f"type == '{db_type}'")

        if "transmission" in user_profile:
            transmission = user_profile["transmission"]
            if transmission and transmission != "any" and not relax:
                trans_value = "AT" if transmission.lower() == "automatic" else "MT"
                filters.append(f"gearbox == '{trans_value}'")

        if "max_age" in user_profile:
            max_age = user_profile["max_age"]
            if not relax:
                filters.append(f"age <= {max_age}")
            else:
                filters.append(f"age <= {max_age + 2}")

        if filters:
            query = " and ".join(filters)
            filtered_df = df.query(query)
        else:
            filtered_df = df

        # For luxury, prioritize luxury brands first
        if (
            user_profile.get("luxury_preference", False)
            or user_profile.get("implicit_needs", {}).get("luxury", 0) > 0.7
        ):
            # Luxury brands in Vietnam
            luxury_brands = [
                "Lexus",
                "Mercedes Benz",
                "Mercedes-Benz",
                "BMW",
                "Audi",
                "Porsche",
                "Land Rover",
                "Range Rover",
                "Volvo",
                "Jaguar",
                "Infiniti",
                "Acura",
                "Cadillac",
                "Lincoln",
                "Genesis",
                "Bentley",
                "Rolls Royce",
            ]
            luxury_df = filtered_df[filtered_df["brand"].isin(luxury_brands)]

            # Enough luxury options
            if len(luxury_df) >= 10:
                return luxury_df

            # Premium non-luxury brands with their high-end models
            premium_brands = [
                "Toyota",
                "Honda",
                "Mazda",
                "Nissan",
                "Ford",
                "Volkswagen",
                "Hyundai",
                "Kia",
            ]
            premium_models = [
                "Camry",
                "Accord",
                "CX-9",
                "Palisade",
                "Highlander",
                "Avalon",
                "Santa Fe",
                "Sorento",
            ]

            premium_df = filtered_df[
                (
                    filtered_df["brand"].isin(premium_brands)
                    & filtered_df["model"].isin(premium_models)
                )
                | (filtered_df["brand"].isin(luxury_brands))
            ]

            # Enough premium options
            if len(premium_df) >= 5:
                return premium_df

        # Return the filtered dataframe
        return filtered_df

    def recommend_cars(self, user_profile: Dict, top_k: int = 5) -> List[Dict]:
        """
        Recommend cars based on user profile with diversity enforcement

        Args:
            user_profile: Processed user profile with preferences
            top_k: Number of recommendations to return

        Returns:
            List of recommended cars with explanations
        """
        # Filter cars based on user constraints
        filtered_df = self._filter_by_constraints(user_profile)

        if len(filtered_df) == 0:
            # Set the "relax" parameter if no cars match the constraints
            filtered_df = self._filter_by_constraints(user_profile, relax=True)

        if len(filtered_df) == 0:
            return []

        # Handle luxury cases
        if (
            user_profile.get("luxury_preference", False)
            and "budget_max" in user_profile
        ):
            budget = user_profile["budget_max"]

            if budget > 1000:  
                filtered_df = filtered_df.sort_values("price_million", ascending=False)

                budget_matches = filtered_df[
                    filtered_df["price_million"] <= budget
                ].head(top_k * 3)

                if len(budget_matches) >= top_k:
                    filtered_df = budget_matches

        # Get filtered car indices and embeddings
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = np.array(
            [self.car_processor.car_embeddings[i] for i in filtered_indices]
        )

        # Get embedding-based recommendations using FAISS
        if "query_embedding" in user_profile:
            query_emb = user_profile["query_embedding"].reshape(1, -1).astype("float32")

            temp_index = faiss.IndexFlatL2(self.car_processor.dimension)
            temp_index.add(filtered_embeddings.astype("float32"))

            # Search for similar cars - get to x5 candidates to ensure diversity
            search_k = min(top_k * 5, len(filtered_df))
            distances, indices = temp_index.search(query_emb, search_k)

            # Map back to original indices
            candidate_indices = [filtered_indices[idx] for idx in indices[0]]
            candidate_cars = self.car_processor.df.iloc[candidate_indices].copy()

            # Get car embeddings for candidates
            car_embs = np.array(
                [self.car_processor.car_embeddings[idx] for idx in candidate_indices]
            )

            # Apply personalized matcher for ranking if there are enough candidates
            if len(candidate_cars) > 0:
                user_emb = query_emb
                user_emb_tensor = torch.tensor(user_emb, dtype=torch.float32).to(
                    self.device
                )
                car_embs_tensor = torch.tensor(car_embs, dtype=torch.float32).to(
                    self.device
                )

                with torch.no_grad():
                    scores = (
                        self.matcher(user_emb_tensor, car_embs_tensor)
                        .cpu()
                        .numpy()
                        .flatten()
                    )

                if (
                    user_profile.get("luxury_preference", False)
                    or user_profile.get("implicit_needs", {}).get("luxury", 0) > 0.7
                ):
                    luxury_brands = [
                        "Lexus",
                        "Mercedes Benz",
                        "Mercedes-Benz",
                        "BMW",
                        "Audi",
                        "Porsche",
                        "Land Rover",
                        "Range Rover",
                        "Volvo",
                        "Jaguar",
                        "Infiniti",
                        "Acura",
                        "Cadillac",
                        "Lincoln",
                        "Genesis",
                        "Bentley",
                        "Rolls Royce",
                    ]

                    for i, car in enumerate(candidate_cars.itertuples()):
                        if car.brand in luxury_brands:
                            scores[i] *= 1.5  # 50% boost for luxury brands
                candidate_cars["match_score"] = scores

                # Generate diversified recommendations
                diversified_cars = self._diversify_recommendations(
                    candidate_cars, top_k
                )

                recommendations = self._prepare_recommendations(
                    diversified_cars, user_profile
                )
                return recommendations

            # Fallback: direct filtering without embedding matching
            diversified_cars = self._diversify_recommendations(
                filtered_df.head(top_k * 3), top_k
            )
            recommendations = self._prepare_recommendations(
                diversified_cars, user_profile
            )
        return recommendations

    def _prepare_recommendations(
        self, cars_df: pd.DataFrame, user_profile: Dict
    ) -> List[Dict]:
        # Prepare recommendation results with explanations
        recommendations = []

        for idx, (_, car) in enumerate(cars_df.iterrows()):
            car_details = {
                "brand": car["brand"],
                "model": car["model"],
                "year": int(car["manufacture_date"]),
                "type": car["type"] if not pd.isna(car["type"]) else "Unknown",
                "seats": int(car["seats"]) if not pd.isna(car["seats"]) else 5,
                "transmission": "Automatic" if car["gearbox"] == "AT" else "Manual",
                "fuel": car["fuel"] if not pd.isna(car["fuel"]) else "petrol",
                "color": car["color"] if not pd.isna(car["color"]) else "Unknown",
                "mileage": int(car["mileage_v2"])
                if not pd.isna(car["mileage_v2"])
                else 0,
                "price": f"{car['price_million']:.1f} million VND",
                "condition": car["condition"],
                "origin": car["origin"] if not pd.isna(car["origin"]) else "Unknown",
            }

            # Generate diverse explanation using Qwen 2.5
            explanation = self.generate_diverse_explanations(
                car_details, user_profile, idx
            )

            # Convert any NumPy types to Python native types for JSON serialization
            match_score = car.get("match_score", 0.5)
            if isinstance(match_score, np.floating):
                match_score = float(match_score)

            # Ensure all values are JSON serializable
            key_features = self._extract_key_features(car)

            recommendation = {
                "id": str(car["id"]), 
                "car": f"{int(car['manufacture_date'])} {car['brand']} {car['model']}",
                "price": f"{float(car['price_million']):.1f} million VND",
                "key_features": key_features,
                "explanation": explanation,
                "description": car["description"],
                "match_score": match_score,
            }

            recommendations.append(recommendation)

        return recommendations

    def _extract_key_features(self, car: pd.Series) -> Dict:
        """Extract key features of a car for display"""
        return {
            "type": str(car["type"]) if not pd.isna(car["type"]) else "Not specified",
            "seats": int(car["seats"])
            if not pd.isna(car["seats"])
            else "Not specified",
            "transmission": "Automatic" if car["gearbox"] == "AT" else "Manual",
            "fuel": str(car["fuel"]) if not pd.isna(car["fuel"]) else "Not specified",
            "mileage": f"{int(car['mileage_v2']):,} km"
            if not pd.isna(car["mileage_v2"])
            else "Not specified",
            "color": str(car["color"])
            if not pd.isna(car["color"])
            else "Not specified",
            "age": f"{int(car['age'])} years old",
        }

    def _diversify_recommendations(
        self, candidate_cars: pd.DataFrame, top_k: int
    ) -> pd.DataFrame:
        # top_k is the number of recommendations to return
        # The goal is to diversify recommendations by avoiding duplicate make/models,
        # "make" and "model" are the columns in the candidate_cars DataFrame 
        # that represent the brand and model of a car

        candidate_cars = candidate_cars.sort_values(
            "match_score", ascending=False
        ).reset_index(drop=True)

        selected_indices = []
        selected_make_models = set()

        # First pass: select top car
        if len(candidate_cars) > 0:
            selected_indices.append(0)
            selected_make_models.add(
                (candidate_cars.iloc[0]["brand"], candidate_cars.iloc[0]["model"])
            )

        # Second pass: iterate through remaining cars and select diverse options
        for i in range(1, len(candidate_cars)):
            # Enough recommendations
            if len(selected_indices) >= top_k:
                break

            make = candidate_cars.iloc[i]["brand"]
            model = candidate_cars.iloc[i]["model"]

            # Skip dpulications
            if (make, model) in selected_make_models:
                continue

            too_similar = False

            for idx in selected_indices:
                selected_car = candidate_cars.iloc[idx]

                # If same brand then price difference should be >20%
                if make == selected_car["brand"]:
                    price_diff_pct = (
                        abs(
                            candidate_cars.iloc[i]["price_million"]
                            - selected_car["price_million"]
                        )
                        / selected_car["price_million"]
                    )

                    # If price difference <20%, consider other features
                    if price_diff_pct < 0.2:
                        if candidate_cars.iloc[i]["type"] == selected_car["type"]:
                            if (
                                abs(
                                    candidate_cars.iloc[i]["seats"]
                                    - selected_car["seats"]
                                )
                                <= 1
                            ):
                                too_similar = True
                                break

            if not too_similar:
                selected_indices.append(i)
                selected_make_models.add((make, model))

        # In case there are not enough diverse cars, fill with highest scoring remaining cars
        if len(selected_indices) < top_k:
            remaining_indices = [
                i for i in range(len(candidate_cars)) if i not in selected_indices
            ]

            remaining_sorted = sorted(
                remaining_indices,
                key=lambda i: candidate_cars.iloc[i]["match_score"],
                reverse=True,
            )

            for i in remaining_sorted:
                if len(selected_indices) >= top_k:
                    break

                selected_indices.append(i)

        return candidate_cars.iloc[selected_indices].reset_index(drop=True)

    def generate_diverse_explanations(self, car_details, user_profile, idx):
        # The goal is to generate diverse explanations by adding a focus instruction
        # to highlight different aspects of the car

        focus_aspects = [
            "Focus on how this car's performance and driving experience match the user's needs.",
            "Focus on how this car's space and practicality features match the user's family needs.",
            "Focus on how this car's value and cost-effectiveness align with the user's budget and economy preference.",
            "Focus on how this car's status and appearance will satisfy the user's luxury preference.",
            "Focus on how this car's reliability and maintenance aspects make it a good choice for the user.",
        ]

        # Select focus based on index of recommended cars
        focus = focus_aspects[idx % len(focus_aspects)]

        system_prompt = f"""
        You are an expert car advisor in Vietnam. Generate a personalized explanation 
        for why this specific car is being recommended to this user. {focus}
        
        Important Vietnamese context to consider:
        - Flooding is a concern in Hanoi, HCMC, and coastal regions during rainy season
        - Family usage often includes extended family members
        - Japanese brands (Toyota, Honda) typically have higher resale value
        - Status considerations are important for business users
        - Fuel economy is a major concern due to rising fuel costs
        
        Keep your explanation conversational, culturally relevant, and under 4 sentences.
        """

        safe_car_details = self._serialize_for_json(car_details)
        safe_user_profile = self._serialize_for_json(
            user_profile, skip_keys=["text_embedding", "query_embedding"]
        )


        prompt = f"Car details:\n{json.dumps(safe_car_details, ensure_ascii=False)}\n\n"
        prompt += (
            f"User profile:\n{json.dumps(safe_user_profile, ensure_ascii=False)}\n\n"
        )
        prompt += "Generate a personalized explanation for why this car is recommended for this user."

        response = self.ollama_client.generate(
            prompt, system_prompt, temperature=0.8, max_tokens=200
        )

        clean_response = response.strip(" \"'")

        return clean_response

    def _serialize_for_json(self, data, skip_keys=None):
        """Helper method to convert any data to JSON-serializable format"""
        if skip_keys is None:
            skip_keys = []

        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if k in skip_keys:
                    continue

                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                elif isinstance(v, np.integer):
                    result[k] = int(v)
                elif isinstance(v, np.floating):
                    result[k] = float(v)
                elif isinstance(v, dict):
                    result[k] = self._serialize_for_json(v, skip_keys)
                else:
                    result[k] = v
            return result
        return data
