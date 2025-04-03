import gradio as gr
import json
import pandas as pd

from processors.car_data_processors import CarDataProcessor
from processors.user_needs_processor import UserNeedsProcessor
from ollama_client import OllamaClient
from deep_match_recommender import DeepMatchRecommender

# Global variables to store initialized components
car_processor = None
user_processor = None
recommender = None
ollama_client = None

def initialize_system():
    """Initialize all system components"""
    global car_processor, user_processor, recommender, ollama_client
    
    status_messages = []
    
    # Initialize Ollama client with correct model name
    status_messages.append("Initializing Ollama client...")
    try:
        ollama_client = OllamaClient(model_name="qwen2.5:7b-instruct-fp16", api_base="http://localhost:11434")
        
        # Test Ollama connection
        test_response = ollama_client.generate("Hello, are you working?", temperature=0.5, max_tokens=10)
        if test_response and not test_response.startswith("Error"):
            status_messages.append(f"✅ Ollama connection successful: {test_response[:50]}...")
        else:
            status_messages.append(f"⚠️ Ollama test returned: {test_response}")
            status_messages.append("Continuing with limited functionality...")
    except Exception as e:
        status_messages.append(f"⚠️ Error initializing Ollama client: {e}")
        status_messages.append("Continuing with limited functionality...")
    
    # Load car data from CSV with caching
    status_messages.append("\nLoading car data...")
    try:
        car_processor = CarDataProcessor('car.csv', cache_dir="./data_cache")
        status_messages.append(f"✅ Loaded {len(car_processor.df)} cars")
    except Exception as e:
        status_messages.append(f"❌ Error loading car data: {e}")
        status_messages.append("Make sure 'car.csv' exists in the current directory")
        return "\n".join(status_messages)
    
    # Initialize user needs processor with Ollama integration
    user_processor = UserNeedsProcessor(car_processor.model, ollama_client)
    
    # Initialize recommender
    recommender = DeepMatchRecommender(car_processor, user_processor, ollama_client)
    status_messages.append("✅ System initialized successfully!")
    
    return "\n".join(status_messages)

def get_recommendations(
    budget_max, 
    family_size, 
    primary_usage, 
    region, 
    max_age, 
    car_type_preference, 
    transmission_preference, 
    free_text
):
    """Generate car recommendations based on user inputs"""
    global car_processor, user_processor, recommender
    
    # Check if system is initialized
    if car_processor is None or user_processor is None or recommender is None:
        return "System not initialized. Please initialize first."
    
    # Create structured input
    structured_input = {
        'budget_max': float(budget_max),  # million VND
        'family_size': int(family_size),
        'primary_usage': primary_usage,
        'region': region,
        'max_age': int(max_age)
    }
    
    # Add optional preferences if selected
    if car_type_preference != "Any":
        structured_input['car_type_preference'] = car_type_preference.lower()
    
    if transmission_preference != "Any":
        structured_input['transmission'] = transmission_preference.lower()
    
    # Process user input and get recommendations
    try:
        user_profile = user_processor.process_user_input(structured_input, free_text)
        recommendations = recommender.recommend_cars(user_profile, top_k=5)
        
        if not recommendations:
            return "No matching cars found for the given criteria. Try relaxing some constraints."
        
        # Format recommendations as HTML
        result = "<h3>Top Recommendations</h3>"
        for i, rec in enumerate(recommendations):
            result += f"<div style='margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px;'>"
            result += f"<h4>{i+1}. {rec['car']} - {rec['price']}</h4>"
            result += f"<p><b>Match Score:</b> {rec['match_score']:.2f}</p>"
            
            # Key features as a table
            result += "<p><b>Key Features:</b></p>"
            result += "<table style='width: 100%; border-collapse: collapse;'>"
            for feature, value in rec['key_features'].items():
                result += f"<tr><td style='padding: 4px; border-bottom: 1px solid #eee;'><b>{feature.capitalize()}</b></td>"
                result += f"<td style='padding: 4px; border-bottom: 1px solid #eee;'>{value}</td></tr>"
            result += "</table>"
            
            # Explanation with highlighted box
            result += f"<p><b>Why This Car:</b> <span style='background-color: #f0f7ff; padding: 5px; display: block;'>{rec['explanation']}</span></p>"
            result += "</div>"
        
        # Add extracted needs if available
        if 'implicit_needs' in user_profile:
            result += "<h3>Analysis of Your Needs</h3>"
            result += "<table style='width: 100%; border-collapse: collapse;'>"
            for need, score in user_profile['implicit_needs'].items():
                if isinstance(score, (int, float)):
                    # Create a visual gauge for the score
                    gauge = "▓" * int(score * 10) + "░" * (10 - int(score * 10))
                    result += f"<tr><td style='padding: 4px; border-bottom: 1px solid #eee;'><b>{need.capitalize()}</b></td>"
                    result += f"<td style='padding: 4px; border-bottom: 1px solid #eee;'>{gauge} {score:.2f}</td></tr>"
                else:
                    result += f"<tr><td style='padding: 4px; border-bottom: 1px solid #eee;'><b>{need.capitalize()}</b></td>"
                    result += f"<td style='padding: 4px; border-bottom: 1px solid #eee;'>{score}</td></tr>"
            result += "</table>"
            
        return result
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error generating recommendations: {str(e)}\n\n{error_trace}"

# Define the Gradio interface
def create_demo():
    """Create the Gradio demo interface"""
    init_status = initialize_system()
    
    with gr.Blocks(title="Car Recommendation System") as demo:
        gr.Markdown("# 🚗 Smart Car Recommendation System")
        gr.Markdown("This demo uses Deep Learning and NLP to recommend cars based on your needs")
        gr.Textbox(label="System Status", max_lines=2, interactive=False, value=init_status)
               
        gr.Markdown("## Enter Your Preferences")
        
        with gr.Row():
            with gr.Column():
                budget_max = gr.Slider(minimum=100, maximum=5000, value=750, step=50, 
                                        label="Maximum Budget (million VND)")
                family_size = gr.Slider(minimum=1, maximum=8, value=2, step=1, 
                                        label="Family Size (number of people)")
                max_age = gr.Slider(minimum=1, maximum=15, value=4, step=1, 
                                    label="Maximum Car Age (years)")
            
            with gr.Column():
                primary_usage = gr.Radio(
                    ["city", "highway", "mixed", "offroad"], 
                    label="Primary Usage", 
                    value="city"
                )
                region = gr.Dropdown(
                    ["TP HCM", "Ha Noi", "Da Nang", "Can Tho", "Nha Trang", "Da Lat", "Other"], 
                    label="Region", 
                    value="TP HCM"
                )
                car_type = gr.Dropdown(
                    ["Any", "Sedan", "SUV", "Hatchback", "MPV", "Pickup"],
                    label="Car Type Preference",
                    value="Any"
                )
                transmission = gr.Dropdown(
                    ["Any", "Automatic", "Manual"],
                    label="Transmission Type",
                    value="Any"
                )
        
        gr.Markdown("## Additional Needs (Optional)")
        gr.Markdown("Describe your needs, preferences, or concerns in your own words:")
        free_text = gr.Textbox(
            label="Free Text Input", 
            lines=5, 
            placeholder="E.g., I need a fuel-efficient car for city driving. I sometimes travel to countryside on weekends."
        )
        
        submit_button = gr.Button("Get Recommendations", variant="primary")
        
        gr.Markdown("## Recommendations")
        result_html = gr.HTML(label="Recommendations")
        
        # Connect the button to the recommendation function
        submit_button.click(
            get_recommendations, 
            inputs=[
                budget_max, family_size, primary_usage, region, max_age, 
                car_type, transmission, free_text
            ],
            outputs=[result_html]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)