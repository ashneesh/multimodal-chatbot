# Multimodal Retail Chatbot

A comprehensive multimodal AI chatbot for retail product discovery and recommendation that processes images, voice, and text inputs simultaneously. This project demonstrates end-to-end development of a production-ready multimodal AI system with Docker containerization.

## 🎓 Course Information

**Degree:** MSc in Artificial Intelligence  
**Module:** Multi-Modal Chatbots  
**Assignment:** Developing a Retail Product Discovery & Recommendation Multi-Modal Chatbot  
**Institution:** Berlin School of Business & Innovation (BSBI) / University for the Creative Arts

## 📋 Assignment Overview

This assignment required developing a proof-of-concept multimodal chatbot capable of:
- Processing uploaded product images (e.g., clothing, accessories, gadgets)
- Handling voice-based customer queries and preferences
- Accepting text-based questions or refinement requests
- Recommending similar or alternative products using AI-driven analysis

The implementation demonstrates the complete AI pipeline: data acquisition, preprocessing, model design, multimodal fusion, training, evaluation, deployment, UI design, and ethical considerations.

## 🚀 Live Demo


**🔗 [Try the Chatbot on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)**

*Note: Replace with your actual HF Spaces URL after deployment*

## 📦 Model & Resources

- **🤗 Hugging Face Model:** [ashneeshkaur/multimodal-chatbot-repo](https://huggingface.co/ashneeshkaur/multimodal-chatbot-repo)
- **📓 Jupyter Notebook:** [multimodal_chatbot_complete.ipynb](./multimodal_chatbot_complete.ipynb)

## ✨ Features

### Multimodal Input Processing
- **Image Analysis**: ResNet50-based visual feature extraction for product classification
- **Voice Transcription**: OpenAI Whisper for converting speech to text
- **Text Understanding**: BERT-based natural language processing for query intent

### Product Recommendations
- Multimodal fusion architecture combining visual and textual features
- Top-3 product category recommendations with confidence scores
- Sample product images for each recommended category

### User Interface
- Clean, intuitive Streamlit web interface
- Support for image upload (JPG, PNG, JPEG)
- Audio file upload and transcription (WAV, MP3)
- Real-time text input for queries
- Visual product recommendations with thumbnails

## 🏗️ Architecture

### Models Used
- **Vision Encoder**: ResNet50 (ImageNet pretrained, frozen backbone)
- **Text Encoder**: BERT-base-uncased (frozen)
- **Audio Processing**: Whisper (base model)
- **Fusion**: Concatenation-based multimodal fusion with MLP classifier

### Dataset
- **Training Data**: Fashion-MNIST (60,000 training samples, 10 classes)
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- CUDA-capable GPU (optional, for faster inference)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/multimodal-chatbot.git
   cd multimodal-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app_complete.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker compose build
   docker compose up
   ```

2. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`


## 📁 Project Structure

```
.
├── app_complete.py              # Main Streamlit application
├── multimodal_chatbot_complete.ipynb  # Complete implementation notebook
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container definition
├── docker-compose.yml          # Docker Compose configuration
├── .dockerignore               # Docker build exclusions
├── README.md                   # This file
```

## 🔬 Technical Implementation

### Preprocessing
- **Images**: Resize to 224x224, normalization, data augmentation (horizontal flip, rotation)
- **Text**: BERT tokenization with max length 64, padding/truncation
- **Audio**: MFCC feature extraction with librosa (for advanced use cases)

### Training
- Fine-tuned ResNet50 and BERT encoders (backbones frozen)
- Multimodal fusion layer with dropout (0.3)
- Cross-entropy loss with Adam optimizer
- Train/validation split: 80/20

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Per-class performance analysis
- Top-3 recommendation accuracy

## 🐳 Docker Configuration

The project includes complete Docker containerization:
- **Dockerfile**: Defines the runtime environment with all dependencies
- **docker-compose.yml**: Orchestrates the containerized service
- Supports both local development and cloud deployment

## 📚 Key Files

- **[multimodal_chatbot_complete.ipynb](./multimodal_chatbot_complete.ipynb)**: Complete implementation including:
  - Data acquisition and exploration
  - Preprocessing pipelines
  - Model design and training
  - Evaluation metrics
  - Deployment code generation

## 🤝 Usage Example

1. **Upload an image** of a product (e.g., a pair of trousers)
2. **Provide a query** via:
   - Text input: "Show me similar items"
   - Audio upload: Record or upload a voice query
3. **Click "Analyze & Recommend"** to get:
   - Predicted product category
   - Confidence score
   - Top-3 recommendations with sample images

## 🔒 Ethical Considerations

- **Data Privacy**: Temporary audio file processing, no permanent storage
- **Bias Mitigation**: Acknowledgment of dataset limitations (Fashion-MNIST is grayscale, limited diversity)
- **Transparency**: Confidence scores provided for all predictions
- **GDPR Compliance**: Transient data processing pipeline

## 📊 Results

The model achieves competitive performance on the Fashion-MNIST dataset with multimodal fusion, demonstrating the effectiveness of combining visual and textual features for product recommendation.

## 📝 License

This project is part of an academic assignment. Please refer to the assignment guidelines for usage and citation requirements.

## 👤 Author

**Ashneesh Kaur**  
MSc in Artificial Intelligence, BSBI

## 🙏 Acknowledgments

- Fashion-MNIST dataset creators
- Hugging Face for transformer models and infrastructure
- OpenAI for Whisper model
- PyTorch and Streamlit communities

---

**Note**: This is an academic project submitted as part of the Multi-Modal Chatbots module. For questions or collaboration, please refer to the assignment guidelines.
