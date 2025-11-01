🧠 Brainwave Emotion Detection App

The Brainwave Emotion Detection App is an intelligent web application that classifies human emotions — Happy, Calm, Sad, and Depressed — from EEG (Electroencephalogram) brainwave data.

It leverages deep learning for emotion classification and features a modern Streamlit interface with intuitive visuals, emotion insights, and probability confidence charts.


| Emotion          | Description                                    |
| ---------------- | ---------------------------------------------- |
| 😄 **Happy**     | Positive engagement, strong beta wave activity |
| 😌 **Calm**      | Relaxed, balanced alpha–theta rhythms          |
| 😢 **Sad**       | Low arousal, moderate alpha wave dominance     |
| 😔 **Depressed** | Reduced energy, emotional imbalance            |


🧬 Model Architecture

Dataset: DEAP EEG dataset

Input shape: (1, 63, 8064)

Classes: Depressed, Sad, Calm, Happy

Frameworks: TensorFlow / Keras

Accuracy: 85.12%

Loss function: Categorical Crossentropy

Optimizer: Adam

🛠️ Tech Stack
| Category          | Tools                              |
| ----------------- | ---------------------------------- |
| **Language**      | Python 3.10+                       |
| **Framework**     | Streamlit                          |
| **AI/ML**         | TensorFlow, Keras                  |
| **Data Handling** | NumPy                              |
| **Visualization** | Matplotlib                         |
| **Styling**       | HTML, CSS (via Streamlit Markdown) |


📚 Research Insight

This project demonstrates the potential of EEG-based emotional recognition in healthcare and AI-driven psychology.
It can be extended to applications such as:

Stress monitoring

Mental health assessment

Neuro-feedback training

Adaptive learning environments

Final model accuracy: 85.12%
