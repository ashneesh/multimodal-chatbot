
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from PIL import Image
import whisper
import numpy as np
import io
import os

# --- Model Definition (Must match training) ---
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MultimodalClassifier, self).__init__()
        self.visual_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.visual_base.fc = nn.Identity()
        self.text_base = BertModel.from_pretrained('bert-base-uncased')
        self.fc_visual = nn.Linear(2048, 512)
        self.fc_text = nn.Linear(768, 512)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
        
    def forward(self, images, input_ids, attention_mask):
        v_feat = self.fc_visual(self.visual_base(images))
        t_out = self.text_base(input_ids=input_ids, attention_mask=attention_mask)
        t_feat = self.fc_text(t_out.last_hidden_state[:, 0, :])
        return self.classifier(torch.cat((v_feat, t_feat), dim=1))

# --- Utils ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@st.cache_resource
def load_resources():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultimodalClassifier(num_classes=10)
    
    # Load Weights if available
    if os.path.exists('multimodal_model_improved.pth'):
        model.load_state_dict(torch.load('multimodal_model_improved.pth', map_location=device))
    else:
        st.warning("Model weights not found! Using random weights.")
        
    model.to(device)
    model.eval()
    
    whisper_model = whisper.load_model("base")
    return tokenizer, model, whisper_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- UI ---
st.title("Multimodal AI Fashion Assistant")
st.write("Upload an image and describe what you are looking for (voice or text).")

tokenizer, model, whisper_model = load_resources()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, width=300)

with col2:
    st.subheader("Input Query")
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    text_query = st.text_input("Or type here...")

    query_text = ""
    if audio_file:
        st.audio(audio_file)
        if st.button("Transcribe Audio"):
            # Save temp
            with open("temp_audio.tmp", "wb") as f:
                f.write(audio_file.getbuffer())
            result = whisper_model.transcribe("temp_audio.tmp")
            query_text = result["text"]
            st.success(f"Transcribed: {query_text}")
    elif text_query:
        query_text = text_query

if st.button("Analyze & Recommend") and uploaded_file and query_text:
    # Prepare Inputs
    img_tensor = transform(image).unsqueeze(0).to(device)
    encoded = tokenizer([query_text], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attn_mask = encoded['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor, input_ids, attn_mask)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
    pred_class = class_names[pred.item()]
    
    st.success(f"Prediction: **{pred_class}**")
    st.info(f"Confidence: {conf.item():.2%}")
    
    st.write(f"Based on your image and request '{query_text}', we recommend checking our **{pred_class}** section.")
