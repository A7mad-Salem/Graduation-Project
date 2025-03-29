# Graduation-Project
DSC-BiLSTM: Depthwise Seprable Convolution with BiLSTM
📌 Overview
DSC-BiLSTM is a deep learning model designed for sequence classification tasks. It leverages Bidirectional LSTM (BiLSTM) and 1D Separable Convolutional Layers to effectively capture both temporal dependencies and spatial patterns in sequential data. This architecture ensures robust feature extraction and improves classification performance.

🚀 Model Architecture
The DSC-BiLSTM model consists of the following components:

1D Separable Convolutional Layers (Feature Extraction):

Captures spatial patterns in raw sequential data.

Reduces computational complexity while retaining essential features.

Bidirectional LSTM (BiLSTM) Layers (Temporal Dependency Modeling):

Extracts long-range dependencies in sequential data.

Uses a 2-layer BiLSTM with a hidden size of 64.

Dropout Regularization:

Applied in LSTM and fully connected layers to prevent overfitting.

Fully Connected Layers (Classification):

Uses two fully connected layers to map extracted features to the final class predictions.

📊 Training Strategy
Loss Function: Categorical Cross-Entropy (for multi-class classification).

Optimizer: Adam Optimizer with learning rate scheduling (CyclicLR).

Gradient Clipping: Prevents exploding gradients in LSTM layers.

Batch Size: Adjusted dynamically for optimal training efficiency.

Early Stopping: Stops training if no improvement in validation accuracy.

📂 Dataset Handling
The dataset contains 982 sequences of varying lengths.

Features include:

SpeedOverGround, accX, accY, accZ, filteredPitch, filteredRoll, filteredYaw

Sequences are preprocessed and padded to maintain a fixed length for training.

🛠️ How to Train the Model
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/A7mad-Salem/DSC-BiLSTM.git
cd DSC-BiLSTM
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Train the Model
bash
Copy
Edit
python train.py
🔍 Model Evaluation
After training, the model is evaluated using a confusion matrix, accuracy, and loss curves.

🖼️ Example Confusion Matrix
(Insert confusion matrix image here)

Precision & Recall metrics are analyzed to measure classification performance.

Validation accuracy is monitored to ensure model generalization.

📢 Future Improvements
Hyperparameter tuning (batch size, dropout, learning rate).

Data augmentation for improving model robustness.

Deploying the model as an API for real-world applications.

📜 License
This project is licensed under the MIT License.

📬 Contact
For questions or contributions, reach out via LinkedIn or email at xx.ahmadsalem.xx@gmail.com.
