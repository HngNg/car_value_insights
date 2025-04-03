import json

from processors.car_data_processors import CarDataProcessor
from processors.user_needs_processor import UserNeedsProcessor
from ollama_client import OllamaClient
from deep_match_recommender import DeepMatchRecommender

# Main demo function with robust error handling
def demo_recommender():
    """Demo the car recommendation system with Qwen 2.5 integration"""
    print("Starting car recommendation system with Qwen 2.5 integration...\n")
    
    # Initialize Ollama client with correct model name
    print("Initializing Ollama client...")
    try:
        ollama_client = OllamaClient(model_name="qwen2.5:7b-instruct-fp16", api_base="http://localhost:11434")
        
        # Test Ollama connection
        print("Testing Ollama connection...")
        test_response = ollama_client.generate("Hello, are you working?", temperature=0.5, max_tokens=10)
        if test_response and not test_response.startswith("Error"):
            print(f"Ollama connection successful: {test_response[:50]}...")
        else:
            print(f"Ollama test returned: {test_response}")
            print("Continuing with limited functionality...")
    except Exception as e:
        print(f"Error initializing Ollama client: {e}")
        print("Continuing with limited functionality...")
        ollama_client = None
    
    # Load car data from CSV with caching
    print("\nLoading car data...")
    try:
        car_processor = CarDataProcessor('car.csv', cache_dir="./data_cache")
        print(f"Loaded {len(car_processor.df)} cars")
    except Exception as e:
        print(f"Error loading car data: {e}")
        print("Make sure 'car.csv' exists in the current directory")
        return None
    
    # Initialize user needs processor with Ollama integration
    user_processor = UserNeedsProcessor(car_processor.model, ollama_client)
    
    # Initialize recommender
    recommender = DeepMatchRecommender(car_processor, user_processor, ollama_client)
    
    # Example user input - structured
    structured_input = {
        'budget_max': 750,  # million VND
        'family_size': 2,
        'primary_usage': 'city',
        'region': 'TP HCM',
        'max_age': 4,
        'car_type_preference': 'suv'
    }
    
    # Example user input - free text
    free_text = """
    - Gia đình tôi có 2 người nhưng thỉnh thoảng tôi chở cả ba mẹ.
    - Tôi thường lái xe trong phố và về quê vào cuối tuần.
    - Tôi cần xe tiết kiệm xăng do đường tôi đi làm thường kẹt xe.
    - Tôi muốn xe sang trọng và lái hay để đi cà phê cuối tuần.
    """
    
    # Process user input
    print("\nProcessing user input...")
    try:
        user_profile = user_processor.process_user_input(structured_input, free_text)
        print("User input processed successfully")
        
        # Print extracted implicit needs
        if 'implicit_needs' in user_profile:
            print("\nExtracted implicit needs:")
            for need, score in user_profile['implicit_needs'].items():
                if isinstance(score, (int, float)):
                    print(f"  - {need}: {score:.2f}")
                else:
                    print(f"  - {need}: {score}")
                    
        # Print luxury preference
        if user_profile.get('luxury_preference', False):
            print("Luxury preference detected!")
    except Exception as e:
        print(f"Error processing user input: {e}")
        print("Using basic user profile without implicit needs")
        user_profile = structured_input
    
    # Get recommendations
    print("\nGenerating recommendations...")
    try:
        recommendations = recommender.recommend_cars(user_profile, top_k=5)
        
        if not recommendations:
            print("No matching cars found for the given criteria")
            return None
            
        print(f"Generated {len(recommendations)} recommendations")
        
        # Print recommendations
        print("\n=== Top Recommendations ===")
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. {rec['car']} - {rec['price']}")
            print(f"   Match Score: {rec['match_score']:.2f}")
            print(f"   Key Features: {json.dumps(rec['key_features'], ensure_ascii=False)}")
            print(f"   Why This Car: {rec['explanation']}")
        
        return recommendations
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    recommendations = demo_recommender()