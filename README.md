# Deep Learning Learning Journey

This repository is a personal learning project where I explore and implement various aspects of **Deep Learning** — starting from basic machine learning models to advanced neural network architectures.  
The aim is to build a solid foundation, experiment with different techniques, and understand their practical applications.

---

## 📂 Project Structure

### 1. **Basic Machine Learning Models**
Located in the `OtherMLModels/` folder.

Currently implemented:
#### 📄 Support Vector Machines
Path: `OtherMLModels/SVMs/`

- **`SVMs.pdf`** — A detailed write-up explaining the theory behind Support Vector Machines, including:
  - Mathematical formulation
  - Kernel functions
  - Margin maximization
  - Soft vs. hard margin SVMs
  - Real-world applications
  
#### 📄 Naive Bayes Classifier
Path: `OtherMLModels/Naive-bayes/`

- **`NaiveBayes.pdf`** — Detailed explanation of the Naive Bayes algorithm, including:
  - Probabilistic model fundamentals
  - Conditional independence assumption
  - Formula derivation
  - Pros and cons
  - Real-world use cases

- **`NaiveBayes.ipynb`** — Jupyter Notebook containing code to implement and analyze Naive Bayes on the **Breast Cancer Wisconsin dataset**, including:
  - Data preprocessing
  - Model training
  - Evaluation metrics
  - Performance analysis and visualization

#### 📄 Multi-Layer Perceptron (MLP)
Path: `OtherMLModels/mlp/`

- **`MLP.pdf`** — In-depth explanation of Multi-Layer Perceptrons, including:
  - Network architecture and activation functions
  - Backpropagation and optimization
  - Overfitting and regularization techniques
  - Experimental results on **CIFAR-10** and **CIFAR-100** datasets
  - Insights from accuracy/loss curves
### 2. **Convolutional Neural Networks (CNNs)**
Located in the `CNNs/` folder.

#### 📄 Image Captioning with CNN + RNN
Path: `CNNs/image_captioning_with_CNN/`

- **`Image_captioning_with_RNNs.pdf`** — Detailed study of an image captioning system that integrates:
  - **CNN (InceptionV3)** for high-level image feature extraction.
  - **RNNs (LSTM and GRU)** for sequential caption generation.
  - Use of **GloVe word embeddings** for semantic-rich input representations.
  - Preprocessing steps for image and text data (tokenization, vocabulary building, embedding).
  - Comparison of model architectures:
    - Baseline LSTM
    - Baseline GRU
    - Stacked GRU (3 layers + dropout)
    - Stacked LSTM (3 layers + dropout)
  - Experimental evaluation using **BLEU scores** and qualitative analysis of generated captions.
  - Discussion on optimizer choices (Adam, SGD, RMSprop) and overfitting mitigation techniques.

- **`image_caption.ipynb`** — Jupyter Notebook implementation for:
  - Data preprocessing (Flickr8k dataset)
  - Feature extraction with InceptionV3
  - Caption generation with different RNN architectures
  - Model training and evaluation with BLEU scores

#### 📄 Comprehensive Comparison of CNN Architectures
Path: `CNNs/Comprehensive_comparision_CNNs/`

- **`Writing/`** — Contains the detailed research study (`Comprehensive_comparison_CNNs.pdf`) covering:
  - Comparative analysis of **five ResNet-18-based models**:
    - **Base CNN** (vanilla ResNet-18)
    - **Local Soft Attention CNN**
    - **Global Soft Attention CNN**
    - **Hard Attention CNN** (MetaDOCK-based kernel selection)
    - **Omni-Directional CNN (ODConv)**
  - Task coverage:
    - **Image Classification** on Tiny ImageNet
    - **Image Segmentation** on Pascal VOC 2012
    - **Time Series Analysis** on UCR Adiac dataset
  - Findings:
    - Attention mechanisms and dynamic convolutions consistently outperform baseline CNNs in accuracy and adaptability.
    - ODConv achieved the **highest classification accuracy (73.4%)** and **mIoU (73.09%)** in segmentation.
    - Dynamic CNNs improved time-series classification (mean accuracy 0.653) over base CNN (mean 0.571).
  - Discussion of:
    - Efficiency vs. computational cost trade-offs (FLOPs analysis)
    - Task-specific advantages of different attention mechanisms
    - Future directions in dynamic CNN optimization.

- **`code/`** — Implementation of all CNN variants:
  - Model definitions for Base CNN, Local & Global Soft Attention, Hard Attention, and ODConv.
  - Training pipelines for classification, segmentation, and time series tasks.
  - Evaluation scripts for mIoU, accuracy, and FLOPs computation.
  
---

## 🎯 Goals of the Project

This project is designed as a **comprehensive learning journey** through the different stages of machine learning and deep learning, with the following objectives:

1. **Build Strong Foundations**  
   - Start with classic machine learning algorithms (e.g., SVM, Naive Bayes) to understand the fundamentals of supervised learning, probabilistic reasoning, and evaluation metrics.

2. **Transition to Neural Networks**  
   - Implement and study basic deep learning models like Multi-Layer Perceptrons (MLPs) to bridge the gap between traditional ML and advanced neural architectures.

3. **Explore Convolutional Neural Networks (CNNs)**  
   - Study CNN theory, implement standard architectures, and experiment with their applications in image classification, segmentation, and other domains.

4. **Investigate Advanced CNN Variants**  
   - Conduct a detailed comparative study of dynamic CNNs, including attention-based models (local, global, hard) and Omni-Directional CNNs, to evaluate their performance across multiple tasks.

5. **Integrate Multi-Modal Deep Learning**  
   - Develop an image captioning system combining CNNs for feature extraction and RNNs (LSTM, GRU) for natural language generation, learning how to merge vision and language models.

6. **Hands-on Experimentation & Analysis**  
   - Implement models from scratch, train on real-world datasets, evaluate with relevant metrics (accuracy, mIoU, BLEU), and document findings for each experiment.

7. **Understand Trade-offs in Model Design**  
   - Analyze the balance between performance, computational cost (FLOPs), and model complexity when choosing architectures for different tasks.

8. **Encourage Reproducibility & Knowledge Sharing**  
   - Maintain well-documented code, structured project folders, and detailed theory write-ups to help others replicate experiments and learn from them.


---

## 🛠 Technologies Used

This project leverages a combination of **programming languages**, **frameworks**, and **tools** for implementing, training, and evaluating models across different machine learning and deep learning tasks.

### **Programming Languages**
- **Python 3.x** — Primary language for all implementations and experiments.
- **Jupyter Notebook** — Interactive environment for code, visualizations, and documentation.

### **Core Libraries & Frameworks**
- **NumPy** — Numerical computations and array operations.
- **Pandas** — Data loading, cleaning, and preprocessing.
- **Matplotlib / Seaborn** — Data visualization.
- **scikit-learn** — Classic ML algorithms (SVM, Naive Bayes, etc.) and preprocessing utilities.
- **TensorFlow / Keras** — Deep learning model building, training, and evaluation.
- **PyTorch** *(optional/future)* — Alternative deep learning framework for experimentation.

### **Model-Specific Tools & Techniques**
- **GloVe Embeddings** — Pre-trained word embeddings for semantic-rich text representation in NLP tasks.
- **InceptionV3** — Pre-trained CNN for image feature extraction in image captioning.
- **ResNet-18** — Backbone architecture for CNN and dynamic CNN experiments.
- **Attention Mechanisms** — Local Soft Attention, Global Soft Attention, Hard Attention.
- **Omni-Directional Convolution (ODConv)** — Rotation-invariant CNN variant.
- **Dynamic Convolutions** — Adaptive kernel modulation for efficiency and performance.

### **Datasets Used**
- **Flickr8k** — Image captioning dataset with 8k images and human-annotated captions.
- **Breast Cancer Wisconsin Dataset** — For Naive Bayes classification analysis.
- **CIFAR-10 / CIFAR-100** — For MLP image classification experiments.
- **Tiny ImageNet** — Image classification benchmark.
- **Pascal VOC 2012** — Semantic segmentation benchmark.
- **UCR Adiac** — Time series classification dataset.

### **Development & Workflow Tools**
- **Git** — Version control.
- **GitHub** — Repository hosting and project collaboration.
- **VS Code** — Main code editor.
- **Google Colab** — Cloud-based GPU acceleration for model training.

<a href="https://vikrant-bhati.github.io/Deep-learning/" target="_blank">
  <img src="https://img.shields.io/badge/Visit%20Website-0b5fff?style=for-the-badge&logo=githubpages&logoColor=white" alt="Visit Website">
</a>


