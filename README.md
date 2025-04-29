# Movie Recommendation with Neural Collaborative Filtering

## Objective
To build a recommendation system that predicts movie ratings and recommends top movies to users using neural collaborative filtering models.

## Implementation

### 1. Data Preparation
- Utilized the MovieLens dataset (`ratings.csv` and `movies.csv`) to collect user-movie interaction data and metadata.
- Encoded user and movie IDs numerically for use in embedding layers.
- Split the dataset into training (80%) and testing (20%) sets.
- Applied normalization where necessary using MinMaxScaler for consistency.

### 2. Model Training and Evaluation

#### Dot Product Model
- **Architecture**: Embedding layers for users and movies followed by a dot product and sigmoid activation to predict normalized ratings.
- **Training**: Trained for 10 epochs using Adam optimizer.
- **Performance**: Achieved RMSE of approximately **0.86** on the test set.
- **Observations**: Lightweight and interpretable, but limited in capturing complex user-item interactions.

#### Multi-Layer Perceptron (MLP) Model
- **Architecture**: Concatenated user and movie embeddings passed through two hidden dense layers, ending with a sigmoid output layer.
- **Training**: Trained for 20 epochs with a batch size of 256.
- **Performance**: Achieved RMSE of approximately **0.81** on the test set.
- **Observations**: Captures nonlinear patterns better than dot product model but requires more training time.

### 3. Model Evaluation
- Evaluated models using **Root Mean Squared Error (RMSE)**.
- Generated **Top-N recommendations** for users by ranking unrated movies.
- Visualized training loss and predicted vs. actual ratings using Matplotlib.

## Skills Demonstrated
- Python programming for machine learning workflows.
- Neural collaborative filtering using TensorFlow/Keras.
- Data preprocessing, including ID encoding and normalization.
- Evaluation using RMSE and ranking-based recommendation logic.
- Data visualization for model insight and interpretation.

## Tools and Libraries Used
- Python  
- TensorFlow / Keras  
- pandas  
- NumPy  
- scikit-learn  
- Matplotlib
