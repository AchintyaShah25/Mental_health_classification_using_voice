"""
Depression Detection Web Interface
Gradio app for real-time audio-based depression screening
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
class Config:
    MODEL_PATH = "./output/best_model.pt"  # Update to your model path
    SAMPLE_RATE = 16000
    DURATION = 10
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============== MODEL DEFINITION ==============
class CNNModel(nn.Module):
    """Same model architecture as training"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.features(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations

# ============== LOAD MODEL ==============
def load_model():
    """Load trained model"""
    model = CNNModel(num_classes=2)
    
    try:
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {Config.MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}")
        print("Using randomly initialized model for demo")
    
    model = model.to(Config.DEVICE)
    model.eval()
    return model

# Initialize model globally
MODEL = load_model()

# Audio transforms
MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=Config.SAMPLE_RATE,
    n_fft=Config.N_FFT,
    hop_length=Config.HOP_LENGTH,
    n_mels=Config.N_MELS
)
DB_TRANSFORM = T.AmplitudeToDB()

# ============== PROCESSING FUNCTIONS ==============
def process_audio(audio_path):
    """Load and process audio file"""
    if audio_path is None:
        return None, None
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != Config.SAMPLE_RATE:
        resampler = T.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Extract chunk
    chunk_samples = Config.SAMPLE_RATE * Config.DURATION
    if waveform.shape[1] > chunk_samples:
        # Use middle chunk
        start = (waveform.shape[1] - chunk_samples) // 2
        waveform = waveform[:, start:start + chunk_samples]
    else:
        # Pad if too short
        padding = chunk_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))
    
    # Convert to mel spectrogram
    mel_spec = MEL_TRANSFORM(waveform)
    mel_spec = DB_TRANSFORM(mel_spec)
    
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    return waveform, mel_spec

def compute_gradcam(model, input_tensor, target_class):
    """Compute Grad-CAM heatmap"""
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    output = model(input_tensor)
    model.zero_grad()
    target = output[0, target_class]
    target.backward()
    
    gradients = model.get_activations_gradient()
    activations = model.get_activations()
    
    if gradients is None or activations is None:
        return np.zeros((Config.N_MELS, Config.DURATION * Config.SAMPLE_RATE // Config.HOP_LENGTH))
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    
    return heatmap.detach().cpu().numpy()

def create_visualization(waveform, mel_spec, heatmap, prediction, confidence):
    """Create visualization figure"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Waveform
    wave_np = waveform.squeeze().numpy()
    time = np.linspace(0, len(wave_np) / Config.SAMPLE_RATE, len(wave_np))
    axes[0, 0].plot(time, wave_np, color='steelblue', linewidth=0.5)
    axes[0, 0].set_title('Audio Waveform', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_xlim(0, Config.DURATION)
    
    # Mel Spectrogram
    spec_np = mel_spec.squeeze().numpy()
    im1 = axes[0, 1].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im1, ax=axes[0, 1], label='dB')
    
    # Grad-CAM Heatmap
    from scipy.ndimage import zoom
    zoom_factors = (spec_np.shape[0] / heatmap.shape[0], 
                   spec_np.shape[1] / heatmap.shape[1])
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    
    im2 = axes[1, 0].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet')
    axes[1, 0].set_title('Model Attention (Grad-CAM)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im2, ax=axes[1, 0], label='Attention')
    
    # Overlay
    axes[1, 1].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis', alpha=0.7)
    axes[1, 1].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.4)
    axes[1, 1].set_title('Spectrogram + Attention Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time Frame')
    axes[1, 1].set_ylabel('Mel Frequency Bin')
    
    # Add prediction text
    pred_text = f"Prediction: {prediction}\nConfidence: {confidence:.1%}"
    color = 'red' if prediction == 'Signs of Depression Detected' else 'green'
    fig.text(0.5, 0.02, pred_text, ha='center', fontsize=14, fontweight='bold', 
             color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# ============== MAIN PREDICTION FUNCTION ==============
def predict_depression(audio):
    """Main prediction function for Gradio"""
    if audio is None:
        return "Please upload or record an audio file.", None, None
    
    try:
        # Process audio
        waveform, mel_spec = process_audio(audio)
        
        if mel_spec is None:
            return "Error processing audio file.", None, None
        
        # Add batch dimension
        mel_spec_batch = mel_spec.unsqueeze(0).to(Config.DEVICE)
        
        # Get prediction
        MODEL.eval()
        with torch.no_grad():
            output = MODEL(mel_spec_batch)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Compute Grad-CAM
        mel_spec_grad = mel_spec.unsqueeze(0).to(Config.DEVICE)
        heatmap = compute_gradcam(MODEL, mel_spec_grad, pred_class)
        
        # Interpret prediction
        if pred_class == 1:
            prediction = "Signs of Depression Detected"
            risk_level = "HIGH" if confidence > 0.7 else "MODERATE"
            recommendation = """
**Important Notice:**
This screening tool detected potential signs of depression in your voice patterns.

**Recommendations:**
• This is NOT a diagnosis - please consult a mental health professional
• Consider reaching out to a counselor or therapist
• Talk to someone you trust about how you're feeling

**Resources:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• SAMHSA Helpline: 1-800-662-4357
            """
        else:
            prediction = "No Signs of Depression Detected"
            risk_level = "LOW"
            recommendation = """
**Result:**
Based on voice analysis, no significant signs of depression were detected.

**Note:**
• This is a screening tool, not a diagnostic test
• If you're experiencing emotional difficulties, please seek professional help
• Regular mental health check-ins are important for everyone

**Wellness Tips:**
• Maintain regular sleep patterns
• Stay physically active
• Connect with friends and family
• Practice stress management techniques
            """
        
        # Create visualization
        viz_image = create_visualization(waveform, mel_spec, heatmap, prediction, confidence)
        
        # Format result
        result_text = f"""
## Analysis Result

**Prediction:** {prediction}
**Confidence:** {confidence:.1%}
**Risk Level:** {risk_level}

---

{recommendation}
        """
        
        return result_text, viz_image, confidence
        
    except Exception as e:
        return f"Error during analysis: {str(e)}", None, None

# ============== GRADIO INTERFACE ==============
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="Depression Screening Tool",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .gradio-container {max-width: 1200px !important}
        .result-box {padding: 20px; border-radius: 10px;}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🧠 AI-Powered Depression Screening Tool
        
        This tool analyzes voice patterns to screen for potential signs of depression.
        Upload an audio recording or record directly using your microphone.
        
        **⚠️ Disclaimer:** This is a screening tool for educational purposes only. 
        It is NOT a substitute for professional medical diagnosis. If you're experiencing 
        mental health concerns, please consult a qualified healthcare provider.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Input Audio")
                
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or Upload Audio (10+ seconds recommended)"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Voice Patterns",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                **Tips for best results:**
                - Record in a quiet environment
                - Speak naturally for at least 10 seconds
                - Describe your day or read a passage aloud
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Analysis Results")
                
                result_text = gr.Markdown(
                    value="Upload or record audio to see results...",
                    elem_classes=["result-box"]
                )
                
                confidence_bar = gr.Number(
                    label="Confidence Score",
                    visible=False
                )
        
        gr.Markdown("### 🔬 Voice Pattern Visualization")
        visualization = gr.Image(
            label="Spectrogram Analysis",
            type="pil"
        )
        
        # Event handler
        analyze_btn.click(
            fn=predict_depression,
            inputs=[audio_input],
            outputs=[result_text, visualization, confidence_bar]
        )
        
        gr.Markdown("""
        ---
        ### About This Tool
        
        This tool uses a Convolutional Neural Network (CNN) trained on the E-DAIC dataset 
        to analyze acoustic features in speech that may be associated with depression.
        
        **How it works:**
        1. Audio is converted to a mel-spectrogram
        2. The CNN analyzes frequency patterns over time
        3. Grad-CAM visualization shows which regions influenced the prediction
        
        **Limitations:**
        - Trained on limited data (~200 samples)
        - May not generalize to all populations
        - Should not replace professional evaluation
        
        ---
        *Built for educational purposes as part of an AI Foundations course project.*
        """)
    
    return demo

# ============== MAIN ==============
if __name__ == "__main__":
    print("Starting Depression Screening Web App...")
    print(f"Using device: {Config.DEVICE}")
    
    demo = create_interface()
    demo.launch(
        share=True,  # Set to True to create public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
