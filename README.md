# Car Value Insights: AI-Powered Car Recommendation System

A culturally-aware, AI-driven car recommendation system designed specifically for the Vietnamese market, helping buyers navigate the complex landscape of car purchasing with personalized, contextually relevant recommendations.

## 🌟 Overview

Vietnam represents a unique automotive market with specific challenges:
- Many buyers are purchasing their first car without family experience or established car-buying traditions
- Vehicles typically cost 2-3x more than in developed markets due to import taxes, making poor choices financially devastating
- Geographical diversity (flood-prone deltas to mountainous terrain) creates highly localized vehicle requirements
- Complex interplay between practical needs, status considerations, and regional factors

This system addresses these challenges by combining contextual filtering, LLM-powered need extraction, and personalized recommendations through a user-friendly web interface.

## ✨ Key Features

- **Contextual Recommendations**: Takes into account Vietnam-specific factors like flooding concerns, regional conditions, and cultural status considerations
- **Free-Text Analysis**: Uses Qwen 2.5 (via Ollama) to extract implicit needs from natural language descriptions
- **Multi-Modal Filtering**: Combines structured inputs (budget, family size) with unstructured text analysis
- **Personalized Matching**: Uses deep learning to match user profiles with suitable vehicles
- **Diversified Results**: Ensures a variety of recommendations rather than similar options
- **Culturally-Aware Explanations**: Generates personalized explanations highlighting why each car matches the user's needs
- **Interactive Interface**: User-friendly Gradio web app for easy interaction

## 🏗️ System Architecture

The system consists of five main components:

1. **OllamaClient**: Interfaces with locally running Ollama API (Qwen 2.5 7B model)
   - Extracts user needs from free text
   - Generates personalized explanations for recommendations

2. **CarDataProcessor**: Prepares car data for recommendation
   - Loads and cleans data from CSV
   - Creates vector embeddings using SentenceTransformer
   - Builds FAISS index for efficient similarity search
   - Implements caching for better performance

3. **UserNeedsProcessor**: Analyzes user requirements
   - Processes structured inputs and free-text descriptions
   - Detects luxury preferences and specific needs
   - Creates query embeddings for matching

4. **PersonalizedMatcher**: Deep learning model for matching users with cars
   - Neural network architecture (PyTorch) for embedding user and car features
   - Creates a shared latent space for comparison
   - Predicts match scores between user profiles and vehicles

5. **DeepMatchRecommender**: Main recommendation engine
   - Filters cars based on user constraints
   - Applies vector similarity search
   - Ensures recommendation diversity
   - Generates personalized explanations with regional context

## 🚀 Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://github.com/ollama/ollama) installed and running locally
- Qwen 2.5 7B Instruct model pulled in Ollama

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/car-value-insights.git
   cd car-value-insights
   ```

2. Set up your environment (choose one option):

   #### Option A: Conda Environment
   ```bash
   # Method 1: Using environment.yml (recommended)
   conda env create -f environment.yml
   conda activate car-insights
   
   # Method 2: Manual creation
   conda create -n car-insights python=3.8
   conda activate car-insights

   # Install PyTorch with conda
   conda install pytorch torchvision -c pytorch

   # Install other dependencies
   pip install -r requirements.txt
   ```

   #### Option B: Pyenv with virtualenv
   ```bash
   # Install Python 3.8 if not already installed
   pyenv install 3.8.12
   
   # Create a virtual environment
   pyenv virtualenv 3.8.12 car-insights
   
   # Activate the environment
   pyenv local car-insights
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. Install Ollama and download the Qwen 2.5 model:
   ```bash
   # Pull the model (only needed once)
   ollama pull qwen2.5:7b-instruct-fp16
   ```

4. Ensure you have the car data CSV file in the root directory:
   ```
   car.csv
   ```

5. Start Ollama with the Qwen 2.5 model:
   ```bash
   ollama run qwen2.5:7b-instruct-fp16
   ```

## 📊 Usage

### Web Interface

1. Start the Gradio web app:
   ```bash
   python app.py
   ```

2. Open your browser at the URL provided (typically http://127.0.0.1:7860)

3. Fill in your preferences:
   - Budget (in million VND)
   - Family size
   - Primary usage (city, highway, mixed, offroad)
   - Region in Vietnam
   - Additional preferences (car type, transmission)
   - Free-text description of your needs (can be in Vietnamese or English)

4. Click "Get Recommendations" to see personalized car suggestions

### Simple Testing

You can also run the test for the recommender by using:

```bash
python test_recommender.py
```

This will run a test with predefined user inputs and display the recommendations in the command line.

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch for personalized matching model
- **Language Models**: Qwen 2.5 7B (via Ollama) for natural language understanding and generation
- **Vector Search**: FAISS for efficient similarity search
- **Embeddings**: SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) for multilingual support
- **Web Interface**: Gradio for interactive user experience
- **Data Processing**: Pandas for data manipulation

## 📂 Project Structure
The project structure is expected to be:

```
car-value-insights/
├── app.py
├── car.csv
├── code_details.txt
├── data_cache
│   └── car_data_cache.pkl
├── deep_match_recommender.py
├── matcher.py
├── ollama_client.py
├── processors
│   ├── car_data_processors.py
│   └── user_needs_processor.py
├── requirements.txt
└── test_recommender.py
```

## 🧩 How to Extend

### Adding New Features

1. **Expanded Regional Context**: Add more region-specific considerations beyond the current set

2. **Integration with Price Prediction**: Combine with ML-based price prediction for price sensitivity analysis

### Customizing for Different Markets

While designed for Vietnam, you can adapt this system for other emerging markets by:

1. Updating regional contexts in the UserNeedsProcessor
2. Modifying luxury brand definitions in DeepMatchRecommender
3. Adjusting system prompts in OllamaClient to reflect local cultural considerations

### Experimenting different choice of open LLM models and LLM server options
Different models may require prompt adjustments in the OllamaClient class
Larger models (e.g., 70B parameters) will require more RAM but may provide better explanations
Some models excel at Vietnamese language tasks (like Bloom-based multilingual models)
For production use, benchmark different models on a test set of user queries


## 🔑 Self-Hosted LLM Server

This system is designed to run entirely locally using Ollama. This approach can be extended to utilized Self-Hosted LLM Servers, using open models such as Phi 4 or Gemma 3. This allowing:
- Direct control over data access and storage, ensuring privacy of user data.
- Full control over the model’s architecture, updates, and configurations, even finetuning
- Reduce expenses associated with cloud-based API calls, especially when processing large volumes of data.



## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
