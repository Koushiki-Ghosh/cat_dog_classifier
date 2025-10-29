import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="🐾 Pet Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        min-height: 200px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }

    .cat-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }

    .dog-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .warning-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #8B4513;
    }

    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }

    .stat-item {
        text-align: center;
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        min-width: 120px;
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .fun-fact {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'cats_detected' not in st.session_state:
    st.session_state.cats_detected = 0
if 'dogs_detected' not in st.session_state:
    st.session_state.dogs_detected = 0
if 'other_objects' not in st.session_state:
    st.session_state.other_objects = 0

model = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem; color: white;">
        🐱 Pet Classifier 🐶
    </h1>
    <p style="font-size: 1.2rem; color: rgba(255, 255, 255, 0.8); margin: 0;">
        AI-Powered Cat vs Dog Detection
    </p>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Your Pet Photo ( Only Dog or Cat )")
    st.markdown("*Supported formats: JPG, PNG, JPEG*")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear photo of a cat or dog for best results"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_resized = image.resize((256, 256))

        # Display image with nice styling
        st.image(
            image,
            caption="📷 Your uploaded image",
            use_container_width=True
        )

        # Predict button with better styling
        if st.button("🔍 Analyze Pet", use_container_width=True, type="primary"):
            if model is not None:
                with st.spinner("🧠 AI is thinking..."):
                    # Add a small delay for better UX
                    time.sleep(1)

                    # Prepare image for prediction
                    img_array = np.array(image_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    prediction = model.predict(img_array, verbose=0)[0][0]

                    # Update session state
                    st.session_state.predictions_made += 1

                    # Store prediction in session state for display in col2
                    st.session_state.current_prediction = prediction
                    st.session_state.show_result = True
            else:
                st.error("❌ Model not loaded. Please check if model.h5 exists.")

    # Fun facts section
    st.markdown("""
    <div class="fun-fact">
        <h4>🧠 Did You Know?</h4>
        <p>This AI model was trained on thousands of cat and dog images to learn the subtle differences between our furry friends!</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Results section
    if hasattr(st.session_state, 'show_result') and st.session_state.show_result:
        prediction = st.session_state.current_prediction

        # Only increment counters for new predictions
        if hasattr(st.session_state, 'just_predicted') and st.session_state.just_predicted:
            st.session_state.predictions_made += 1
            st.session_state.just_predicted = False  # Reset flag

         #   if prediction > 0.9:
          #      st.session_state.dogs_detected += 1
           # elif prediction < 0.1:
            #    st.session_state.cats_detected += 1
            #elif 0.15 <= prediction <= 0.85:
             #   st.session_state.other_objects += 1

        # Display results (always show, but only count once)
        if prediction > 0.96:
            st.markdown(f"""
            <div class="prediction-card dog-card pulse">
                <h2>🐶 It's a Dog!</h2>
                <h3>{prediction*100:.1f}% Confident</h3>
                <p>Woof! This adorable pup has been detected with high confidence.</p>
            </div>
            """, unsafe_allow_html=True)

        elif prediction < 0.01:
            confidence = 1 - prediction
            st.markdown(f"""
            <div class="prediction-card cat-card pulse">
                <h2>🐱 It's a Cat!</h2>
                <h3>{confidence*100:.1f}% Confident</h3>
                <p>Meow! This cute kitty has been identified with high confidence.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Check if it's likely neither cat nor dog (broader range for better detection)
            if 0.1 <= prediction <= 0.9:
                st.markdown(f"""
                <div class="prediction-card warning-card">
                    <h2>🚫 Not a Cat or Dog!</h2>
                    <h3>Confidence: {max(prediction, 1-prediction)*100:.1f}%</h3>
                    <p>This doesn't appear to be a cat or dog. Please upload a clear photo of a cat or dog for accurate classification!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card warning-card">
                    <h2>🤔 Uncertain Result</h2>
                    <h3>Confidence: {max(prediction, 1-prediction)*100:.1f}%</h3>
                    <p>The image might be unclear . Try uploading a clearer photo !</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        # Default state - show upload instructions
        st.markdown("""
        <div class="upload-section" style="text-align: center; padding: 3rem 2rem;">
            <h3>🎯 Ready to Analyze!</h3>
            <p>Upload a photo of your pet on the left and click "Analyze Pet" to see the magic happen!</p>
            <div style="font-size: 4rem; margin: 2rem 0;">🐾</div>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.7);">
    <p>Made with ❤️ using Streamlit and TensorFlow</p>
    <p style="font-size: 0.8rem;">Tip: For best results, use high-quality photos with good lighting!</p>
</div>
""", unsafe_allow_html=True)

# Add some JavaScript for additional interactivity
st.markdown("""
<script>
// Add smooth scrolling and other enhancements
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to buttons
    const buttons = document.querySelectorAll('.stButton > button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.transition = 'all 0.3s ease';
        });
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});
</script>
""", unsafe_allow_html=True)
