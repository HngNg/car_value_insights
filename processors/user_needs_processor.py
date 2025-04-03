import numpy as np
from typing import Dict
class UserNeedsProcessor:
    """Process user input to extract needs and preferences using Qwen 2.5"""
    
    def __init__(self, sentence_model, ollama_client):
        self.model = sentence_model # Pre-trained sentence transformer model
        self.ollama_client = ollama_client

    
    def process_user_input(self, structured_input: Dict, free_text: str = None) -> Dict:
        """
        Process structured and unstructured user input to extract needs and preferences
        
        Args:
            structured_input: Dictionary containing structured user input
                (budget, family size, usage frequency, etc.)
            free_text: Free-form text input from the user about their needs
            
        Returns:
            Dictionary containing processed user needs and preferences
        """
        user_profile = structured_input.copy()
        
        # Extract needs from free text 
        if free_text:
            need_scores = self.ollama_client.extract_user_needs(free_text, structured_input)
            user_profile['implicit_needs'] = need_scores
            
            # Create embedding for the free text
            user_profile['text_embedding'] = self.model.encode([free_text])[0]
            
            # Store the original text for later use
            user_profile['free_text'] = free_text
            
            # Check for luxury preference keywords
            luxury_keywords = [
                "sang trọng", "đẳng cấp", "cao cấp", "luxury", "premium", "đẳng cấp",
                "lexus", "mercedes", "bmw", "audi", "porsche", "range rover", "land rover"
            ]
            text_lower = free_text.lower()
            
            luxury_score = user_profile['implicit_needs'].get('luxury', 0)
            # If luxury keywords found, ensure high luxury score
            for keyword in luxury_keywords:
                if keyword in text_lower:
                    luxury_score = max(luxury_score, 0.9)
                    user_profile['luxury_preference'] = True
                    break
                    
            user_profile['implicit_needs']['luxury'] = luxury_score
        
        # Create a car query embedding
        query_parts = []
        
        # Budget constraints - for high budgets, emphasize premium options
        if 'budget_max' in user_profile:
            budget = user_profile['budget_max']
            
            # For budgets over 1 billion VND, emphasize luxury
            if budget > 1000:
                query_parts.append(f"Premium luxury vehicle within {budget} million VND budget")
                user_profile['luxury_preference'] = True
            else:
                query_parts.append(f"Price under {budget} million VND")
        
        # Family size
        if 'family_size' in user_profile:
            if user_profile['family_size'] > 5:
                query_parts.append("Large family car with many seats")
            else:
                query_parts.append(f"Car with at least {user_profile['family_size']} seats")
        
        # Car usage
        if 'primary_usage' in user_profile:
            usage = user_profile['primary_usage']
            if usage == 'city':
                query_parts.append("Good for city driving and traffic")
            elif usage == 'highway':
                query_parts.append("Comfortable for highway driving")
            elif usage == 'mixed':
                query_parts.append("Versatile for both city and highway")
            elif usage == 'offroad':
                query_parts.append("Good for rough terrain and countryside")
        
        # Region-specific needs
        if 'region' in user_profile:
            region = user_profile['region']
            if region in ['Hanoi', 'Ho Chi Minh City', 'TP HCM', 'TPHCM', 'HCM']:
                query_parts.append("Good for navigating busy city streets")
            if region in ['Hanoi', 'Can Tho', 'Hue', 'Ho Chi Minh City', 'TP HCM', 'TPHCM', 'HCM']:
                query_parts.append("Good for handling flooded roads during rainy season")
            if region in ['Da Lat', 'Sa Pa', 'Ha Giang']:
                query_parts.append("Good for mountainous terrain")
        
        # Add luxury preference if detected
        if user_profile.get('luxury_preference') or user_profile.get('implicit_needs', {}).get('luxury', 0) > 0.7:
            query_parts.append("Luxury premium vehicle with high-end features and prestigious brand")
        
        # Add any specific concerns detected by Qwen
        if 'implicit_needs' in user_profile and 'specific_concerns' in user_profile['implicit_needs']:
            specific_concerns = user_profile['implicit_needs'].get('specific_concerns')
            if specific_concerns and not isinstance(specific_concerns, (int, float)):
                query_parts.append(f"Addresses concerns: {specific_concerns}")
        
        # Combine all parts into a single query
        if query_parts:
            query = " ".join(query_parts)
            user_profile['query_embedding'] = self.model.encode([query])[0]
        elif 'text_embedding' in user_profile:
            # If no structured data but we have text embedding, use that
            user_profile['query_embedding'] = user_profile['text_embedding']
        else:
            # Empty query - will need to rely on filtering only
            user_profile['query_embedding'] = np.zeros(self.model.get_sentence_embedding_dimension())
        
        return user_profile