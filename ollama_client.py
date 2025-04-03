import numpy as np
import re
import json
import requests

class OllamaClient:
    def __init__(self, model_name="qwen2.5:7b-instruct-fp16", api_base="http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{api_base}/api/generate"
        
    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=500):
        # Format the prompt into JSON
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            # Make the API call with POST request
            response = requests.post(self.api_url, json=payload)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response content: {response.text}")
                return f"Error calling API. Status code: {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            # Return empty string in case of error
            return f"Error: {str(e)}"
            
    def extract_user_needs(self, user_text, structured_data=None):
        """
        Extract user needs from free text using Qwen 2.5
        
        Args:
            user_text: Free text input from user
            structured_data: Optional structured data about user preferences
            
        Returns:
            Dictionary with extracted needs and confidence scores
        """
        # Create the system prompt
        system_prompt = """
        You are an expert car recommendation system for the Vietnamese market. 
        Analyze the user's text and extract their implicit needs and preferences.
        Focus on these categories:
        - family: Need for family-oriented vehicle
        - luxury: Desire for status or luxury
        - economy: Concern for fuel efficiency and price
        - off_road: Need for off-road capability
        - city: Focus on city driving
        - business: Need for professional appearance
        - performance: Interest in power and handling
        - specific_concerns: Any specific concerns (e.g., flooding)
        - regional_factors: Region-specific considerations
        
        Return your analysis as a JSON object with scores from 0-1 for each category.
        Example format: {"family": 0.8, "luxury": 0.3, ...}
        """
        
        # Create the prompt
        prompt = f"Here is a user's description of their car needs:\n\n{user_text}"
        
        if structured_data:
            # Convert numpy values to native Python types for JSON serialization
            safe_structured_data = {}
            for k, v in structured_data.items():
                if isinstance(v, np.ndarray):
                    safe_structured_data[k] = v.tolist()
                elif isinstance(v, np.integer):
                    safe_structured_data[k] = int(v)
                elif isinstance(v, np.floating):
                    safe_structured_data[k] = float(v)
                else:
                    safe_structured_data[k] = v
            
            prompt += f"\n\nThey also provided these structured preferences: {json.dumps(safe_structured_data, ensure_ascii=False)}"
        
        prompt += "\n\nExtract their implicit needs and provide a JSON with confidence scores (0-1) for each category."
        
        # Get the response
        response = self.generate(prompt, system_prompt)
        print(f"Raw Ollama response for needs extraction: {response[:200]}...")
        
        # Extract JSON from the response
        try:
            # Find JSON in the response (looking for text between {} brackets)
            json_match = re.search(r'(\{.*\})', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                needs = json.loads(json_str)
                return needs
            else:
                # If no JSON found, try to extract key-value pairs
                needs = {}
                for category in ["family", "luxury", "economy", "off_road", "city", "business", "performance", 
                                "specific_concerns", "regional_factors"]:
                    match = re.search(rf'"{category}":\s*(0\.\d+|1\.0|1)', response)
                    if match:
                        needs[category] = float(match.group(1))
                    else:
                        needs[category] = 0.0
                return needs
        except Exception as e:
            print(f"Error extracting JSON from Ollama response: {e}")
            print(f"Raw response: {response}")
            # Return default values
            return {
                "family": 0.0,
                "luxury": 0.0,
                "economy": 0.0,
                "off_road": 0.0,
                "city": 0.0,
                "business": 0.0,
                "performance": 0.0,
                "specific_concerns": "",
                "regional_factors": ""
            }
            
    def generate_explanation(self, car_details, user_profile):
        """
        Generate personalized explanation for car recommendation
        
        Args:
            car_details: Details of the recommended car
            user_profile: User profile with preferences
            
        Returns:
            Personalized explanation string
        """
        # Create the system prompt
        system_prompt = """
        You are an expert car advisor in Vietnam. Generate a personalized explanation 
        for why this specific car is being recommended to this user. Focus on how the 
        car's features match the user's needs, preferences, and regional considerations.
        
        Important Vietnamese context to consider:
        - Flooding is a concern in Hanoi, HCMC, and coastal regions during rainy season
        - Family usage often includes extended family members
        - Japanese brands (Toyota, Honda) typically have higher resale value
        - Status considerations are important for business users
        - Fuel economy is a major concern due to rising fuel costs
        
        Keep your explanation conversational, culturally relevant, and under 4 sentences. Do not say hi.
        """
        
        # Convert numpy values to native Python types for JSON serialization
        safe_car_details = {}
        for k, v in car_details.items():
            if isinstance(v, np.ndarray):
                safe_car_details[k] = v.tolist()
            elif isinstance(v, np.integer):
                safe_car_details[k] = int(v)
            elif isinstance(v, np.floating):
                safe_car_details[k] = float(v)
            else:
                safe_car_details[k] = v
        
        safe_user_profile = {}
        for k, v in user_profile.items():
            # Skip embedding keys to reduce prompt size
            if k in ['text_embedding', 'query_embedding']:
                continue
                
            if isinstance(v, np.ndarray):
                safe_user_profile[k] = v.tolist()
            elif isinstance(v, np.integer):
                safe_user_profile[k] = int(v)
            elif isinstance(v, np.floating):
                safe_user_profile[k] = float(v)
            elif isinstance(v, dict):
                # Handle nested dictionaries
                safe_user_profile[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, np.ndarray):
                        safe_user_profile[k][sub_k] = sub_v.tolist()
                    elif isinstance(sub_v, np.integer):
                        safe_user_profile[k][sub_k] = int(sub_v)
                    elif isinstance(sub_v, np.floating):
                        safe_user_profile[k][sub_k] = float(sub_v)
                    else:
                        safe_user_profile[k][sub_k] = sub_v
            else:
                safe_user_profile[k] = v
        
        # Create the prompt
        prompt = f"Car details:\n{json.dumps(safe_car_details, ensure_ascii=False)}\n\n"
        prompt += f"User profile:\n{json.dumps(safe_user_profile, ensure_ascii=False)}\n\n"
        prompt += "Generate a personalized explanation for why this car is recommended for this user."
        
        # Get the response
        response = self.generate(prompt, system_prompt, temperature=0.7, max_tokens=200)
        
        # Clean up the response (remove any JSON formatting, quotes, etc.)
        clean_response = response.strip(' "\'')
        
        return clean_response