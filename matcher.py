import torch
import torch.nn as nn

class PersonalizedMatcher(nn.Module):
    """PyTorch model for personalized car matching"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Initialize the matcher model
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden layer dimension
        """
        super(PersonalizedMatcher, self).__init__()
        
        # User preference encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Car feature encoder
        self.car_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Matching network
        self.matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_emb, car_embs):
        """
        Forward pass
        
        Args:
            user_emb: User embedding (1, embedding_dim)
            car_embs: Car embeddings (batch_size, embedding_dim)
            
        Returns:
            Match scores (batch_size, 1)
        """
        # Encode user preferences
        user_features = self.user_encoder(user_emb)  # (1, hidden_dim)
        
        # Encode car features
        car_features = self.car_encoder(car_embs)  # (batch_size, hidden_dim)
        
        # Expand user features to match car features
        user_features_expanded = user_features.expand(car_features.shape[0], -1)
        
        # Concatenate user and car features
        combined = torch.cat([user_features_expanded, car_features], dim=1)
        
        # Compute match scores
        scores = self.matcher(combined)
        
        return scores