import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from io import BytesIO
import tempfile
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from skimage.transform import resize


def resize_image(image: np.ndarray, target_size: tuple = (128, 128)) -> np.ndarray:
    # Get current image dimensions
    current_shape = image.shape

    # Check if resize is needed
    if current_shape[0] == target_size[0] and current_shape[1] == target_size[1]:
        return image

    # Determine number of channels
    if len(current_shape) == 3:
        # Color image with channels
        channels = current_shape[2]
        target_shape = (target_size[0], target_size[1], channels)
    else:
        # Grayscale image without channels
        target_shape = target_size

    resized = resize(image, target_shape, anti_aliasing=True, preserve_range=True)

    # Convert back to original dtype (usually uint8)
    resized = resized.astype(image.dtype)

    return resized


# Skull stripping function
def skull_strip(image):
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Otsu's thresholding to separate brain from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the binary mask
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find the largest connected component (should be the brain)
    labels = measure.label(opening)
    regions = measure.regionprops(labels)

    # If no regions were found, return the original image
    if not regions:
        return image

    # Sort regions by area and select the largest one (brain)
    regions.sort(key=lambda x: x.area, reverse=True)
    brain_label = regions[0].label
    brain_mask = (labels == brain_label).astype(np.uint8) * 255

    # Apply the mask to the original image
    if len(image.shape) > 2:
        result = cv2.bitwise_and(image, image, mask=brain_mask)
    else:
        result = cv2.bitwise_and(gray, gray, mask=brain_mask)

    return result


# Intensity normalization function
def intensity_normalize(image, percentile_low=1, percentile_high=99):
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Find non-zero pixels (brain region)
    non_zero_mask = gray > 0
    if not np.any(non_zero_mask):
        return image  # Return original if no brain regions

    # Calculate percentiles from non-zero pixels
    p_low = np.percentile(gray[non_zero_mask], percentile_low)
    p_high = np.percentile(gray[non_zero_mask], percentile_high)

    # Clip intensities to the percentile range
    clipped = np.clip(gray, p_low, p_high)

    # Normalize to [0, 255] range
    normalized = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    # If original was color, convert back to color
    if len(image.shape) > 2:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    return normalized


# The correct feature extraction function used during training
def extract_features(image):
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create a feature dictionary
    features = {}

    # Basic statistical features
    features["mean"] = np.mean(gray)
    features["std"] = np.std(gray)
    features["median"] = np.median(gray)
    features["min"] = np.min(gray)
    features["max"] = np.max(gray)

    # Histogram-based features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    # Add small epsilon to avoid log(0)
    features["entropy"] = -np.sum(hist * np.log2(hist + 1e-7))

    # Calculate skewness
    m3 = np.mean((gray - features["mean"]) ** 3)
    features["skewness"] = m3 / (features["std"] ** 3 + 1e-9)

    # Calculate kurtosis
    m4 = np.mean((gray - features["mean"]) ** 4)
    features["kurtosis"] = m4 / (features["std"] ** 4 + 1e-9) - 3  # Excess kurtosis

    # Texture features using GLCM (Gray Level Co-occurrence Matrix)
    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    # More robust region segmentation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Label connected components
    labels = measure.label(binary)
    regions = measure.regionprops(labels)

    if regions and len(regions) >= 3:
        # Sort regions by area and select the largest few
        regions.sort(key=lambda x: x.area, reverse=True)

        # Filter regions by shape and size criteria
        potential_hippocampus_regions = []
        for region in regions[1:10]:  # Skip the largest region (usually background)
            # Hippocampus typically has an elongated shape
            if region.area > 100 and region.eccentricity > 0.5:
                # Check relative position (hippocampus is roughly mid-brain structure)
                y, x = region.centroid
                if (0.3 < y / gray.shape[0] < 0.7) and (0.3 < x / gray.shape[1] < 0.7):
                    potential_hippocampus_regions.append(region)

        if potential_hippocampus_regions:
            # Use the largest of the potential hippocampus regions
            hippocampus = max(potential_hippocampus_regions, key=lambda r: r.area)
            features["approx_hippocampal_volume"] = hippocampus.area
            features["hippocampus_eccentricity"] = hippocampus.eccentricity
            features["hippocampus_orientation"] = hippocampus.orientation
        else:
            # Fallback method - use third largest region
            if len(regions) >= 3:
                features["approx_hippocampal_volume"] = regions[2].area
            else:
                features["approx_hippocampal_volume"] = 0
            features["hippocampus_eccentricity"] = 0
            features["hippocampus_orientation"] = 0

        # White matter approximation (largest region)
        features["approx_white_matter"] = regions[0].area

        # Gray matter approximation (second largest region)
        features["approx_gray_matter"] = regions[1].area

        # Approximation of cortical thickness
        y, x = regions[0].centroid
        coords = regions[0].coords
        distances = np.sqrt((coords[:, 0] - y) ** 2 + (coords[:, 1] - x) ** 2)
        features["approx_cortical_thickness"] = np.mean(distances)
    else:
        # Default values if regions can't be found
        features["approx_hippocampal_volume"] = 0
        features["hippocampus_eccentricity"] = 0
        features["hippocampus_orientation"] = 0
        features["approx_white_matter"] = 0
        features["approx_gray_matter"] = 0
        features["approx_cortical_thickness"] = 0

    # Add GLCM texture features for more detailed texture analysis
    if gray.size > 0:
        # Resize image if it's too large to compute GLCM efficiently
        max_size = 256
        if gray.shape[0] > max_size or gray.shape[1] > max_size:
            scale = max_size / max(gray.shape[0], gray.shape[1])
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray_resized = cv2.resize(gray, new_size)
        else:
            gray_resized = gray

        # Normalize to fewer gray levels for GLCM computation
        gray_normalized = (gray_resized / 16).astype(np.uint8)

        # Calculate GLCM
        try:
            glcm = graycomatrix(
                gray_normalized,
                [1],
                [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                16,
                symmetric=True,
                normed=True,
            )

            # Calculate GLCM properties
            features["contrast"] = graycoprops(glcm, "contrast").mean()
            features["dissimilarity"] = graycoprops(glcm, "dissimilarity").mean()
            features["homogeneity"] = graycoprops(glcm, "homogeneity").mean()
            features["energy"] = graycoprops(glcm, "energy").mean()
            features["correlation"] = graycoprops(glcm, "correlation").mean()
            features["ASM"] = graycoprops(glcm, "ASM").mean()
        except Exception as e:
            # If GLCM computation fails, use default values
            st.warning(f"GLCM computation failed: {str(e)}")
            features["contrast"] = 0
            features["dissimilarity"] = 0
            features["homogeneity"] = 0
            features["energy"] = 0
            features["correlation"] = 0
            features["ASM"] = 0

    return features


# Convert feature dictionary to numpy array for model input
def features_dict_to_array(features_dict, feature_keys):
    # Extract features in the correct order
    feature_array = np.array([features_dict.get(key, 0) for key in feature_keys])
    return feature_array


# New function to create ensemble voting classifier from individual models
def create_ensemble_classifier(models_dict, voting_type="hard"):
    estimators = [
        (name.replace(" ", "_").lower(), model) for name, model in models_dict.items()
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
    return voting_clf


def load_models(models_dir):
    models = {}
    individual_models = {}

    try:
        # Model name mapping
        model_name_mapping = {
            "xgboost": "XGBoost",
            "logistic_regression": "Logistic_Regression",
            "random_forest": "Random_Forest",
            "mlp": "MLP",
            "svm_rbf": "svm_rbf",
        }

        for file in os.listdir(models_dir):
            if file.endswith(".pkl"):
                model_type = None
                for key in model_name_mapping.keys():
                    if key in file.lower():
                        model_type = model_name_mapping[key]
                        break

                if model_type:
                    model_path = os.path.join(models_dir, file)
                    try:
                        with open(model_path, "rb") as f:
                            model = joblib.load(f)
                        individual_models[model_type] = model

                    except Exception as e:
                        st.error(f" Failed to load {file}: {e}")

        # Attach working models
        models["individual_models"] = individual_models

        if len(individual_models) >= 2:
            models["ensemble"] = create_ensemble_classifier(
                individual_models, voting_type="hard"
            )
        else:
            st.warning(" Not enough models for ensemble voting.")
            models["ensemble"] = None

        # Always define these — feature_keys are required for feature extraction
        models["feature_keys"] = [
            "mean",
            "std",
            "median",
            "min",
            "max",
            "entropy",
            "skewness",
            "kurtosis",
            "edge_density",
            "approx_hippocampal_volume",
            "hippocampus_eccentricity",
            "hippocampus_orientation",
            "approx_white_matter",
            "approx_gray_matter",
            "approx_cortical_thickness",
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ]

        return models

    except Exception as e:
        st.error(f" General error loading models: {e}")
        return {"individual_models": {}, "ensemble": None, "feature_keys": []}


# Modified apply_rfe_then_pca_for_testing function
def apply_rfe_then_pca_for_testing(feature_df):
    # Load the list of important features
    try:
        important_features = joblib.load("models/rfe_selected_features.pkl")
        print(f"Loaded features: {important_features}")
    except FileNotFoundError:
        print("Warning: Feature list file not found. Using hardcoded list.")
        important_features = [
            "edge_density",
            "mean",
            "median",
            "approx_white_matter",
            "max",
            "approx_hippocampal_volume",
            "hippocampus_orientation",
        ]

    # Check if all features exist in the input data
    available_features = [f for f in important_features if f in feature_df.columns]

    if not available_features:
        return feature_df.values, feature_df.columns.tolist(), None

    # Extract only the important features - ensure it's 2D
    X_selected = feature_df[available_features].values
    if len(X_selected.shape) > 2:
        X_selected = X_selected.reshape(1, -1)

    # Load the saved scaler and apply it (don't fit again)
    try:
        scaler = joblib.load("models/scaler_model.pkl")
        X_scaled = scaler.transform(X_selected)  # Use transform, not fit_transform

        # Load and apply PCA transformation
        pca_model = joblib.load("models/pca_model.pkl")

        X_pca = pca_model.transform(X_scaled)  # Transform with the loaded model

        return X_pca, available_features, X_selected
    except Exception as e:
        import traceback

        traceback.print_exc()
        # Return original features if anything fails
        return X_selected, available_features, X_selected


# Ensemble prediction function
def ensemble_predict(individual_models, features, voting_type="hard"):
    if not individual_models:
        return None, None

    all_predictions = []
    all_probabilities = []

    for name, model in individual_models.items():
        try:
            pred = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            all_predictions.append(pred)
            all_probabilities.append(probs)
        except Exception as e:
            st.warning(f"Error getting predictions from {name}: {str(e)}")

    if not all_predictions:
        return None, None

    if voting_type == "hard":
        unique_classes, counts = np.unique(all_predictions, return_counts=True)
        predicted_class = unique_classes[np.argmax(counts)]
        avg_probs = np.mean(all_probabilities, axis=0)
        return predicted_class, avg_probs
    else:
        avg_probs = np.mean(all_probabilities, axis=0)
        predicted_class = np.argmax(avg_probs)
        return predicted_class, avg_probs

# Class mapping
class_mapping = {
    0: "Non Demented",
    1: "Very mild Dementia",
    2: "Mild Dementia",
    3: "Moderate Dementia",
}

def classify_mri_image(image: np.ndarray, models_dir: str = "models/"):
    # === 1. Resize the image to correct dimensions ===
    image = resize_image(image, target_size=(224, 224))
    
    # === 2. Load models and constants ===
    models_data = load_models(models_dir)
    individual_models = models_data["individual_models"]
    ensemble_model = models_data["ensemble"]
    feature_keys = models_data["feature_keys"]

    if not individual_models or not feature_keys:
        raise ValueError("Models or feature keys not loaded properly.")

    # === 3. Preprocessing image ===
    stripped = skull_strip(image)
    normalized = intensity_normalize(stripped)

    # Display preprocessing results in Streamlit
    st.subheader("Image Preprocessing")

    # Optionally display the images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(normalized, caption="Normalized Image", use_column_width=True)

    # === 4. Extract features ===
    features_dict = extract_features(normalized)
    feature_array = features_dict_to_array(features_dict, feature_keys)

    # Display extracted features
    st.subheader("Feature Extraction")

    # === 5. Convert to DataFrame for feature naming ===
    feature_df = pd.DataFrame([feature_array], columns=feature_keys)
    st.write(f"Feature DataFrame shape: {feature_df.shape}")

    # Display the actual feature values
    st.write("Feature values:")
    st.dataframe(feature_df)

    # === 6. Apply pre-determined feature selection ===
    features_rfe, selected_names, original_features = apply_rfe_then_pca_for_testing(
        feature_df
    )

    st.subheader("Feature Selection Results")
    st.write(f"Selected {len(selected_names)} features:")
    st.json(selected_names)

    # Debug: Print selected features values - ensure it's a properly shaped array
    st.write("Selected feature values:")
    try:
        # Ensure original_features is properly shaped for DataFrame
        if original_features is not None:
            if len(original_features.shape) > 2:
                original_features_2d = original_features.reshape(1, -1)
            else:
                original_features_2d = original_features

            st.write(pd.DataFrame(original_features_2d, columns=selected_names))
        else:
            st.write("No features selected")
    except Exception as e:
        st.write(f"Error displaying features: {str(e)}")
        st.write(
            f"Feature shape: {original_features.shape if original_features is not None else None}"
        )

    # Debug: Print PCA-transformed values
    st.write("PCA-transformed values:")
    try:
        st.write(pd.DataFrame(features_rfe))
    except Exception as e:
        st.write(f"Error displaying PCA features: {str(e)}")
        st.write(
            f"PCA feature shape: {features_rfe.shape if features_rfe is not None else None}"
        )

    # === 7. No additional standardization needed, as it's done in the PCA function ===
    features_final = features_rfe

    # === 8. Make predictions with debugging info ===
    individual_predictions = {}
    individual_probabilities = {}

    for name, model in individual_models.items():
        try:
            print(f"Applying model: {name}")
            print(f"Model type: {type(model)}")
            pred = model.predict(features_final)[0]
            probs = model.predict_proba(features_final)[0]
            label = class_mapping.get(pred, str(pred))
            individual_predictions[name] = label
            individual_probabilities[name] = probs
            print(f"Model {name} predicted: {label}, probabilities: {probs}")
        except Exception as e:
            print(f"Error with model {name}: {str(e)}")
            individual_predictions[name] = f"Error: {str(e)}"

   # === 9. Ensemble Voting ===
    ensemble_class_idx, ensemble_probs = ensemble_predict(individual_models, features_final, voting_type='hard')

    if ensemble_class_idx is not None:
        predicted_label = class_mapping.get(ensemble_class_idx, f"Unknown Class {ensemble_class_idx}")
    else:
        predicted_label = None

    result = {
        "predicted_label": predicted_label,
        "ensemble_probabilities": ensemble_probs.tolist() if ensemble_probs is not None else [],
        "raw_class_index": int(ensemble_class_idx) if ensemble_class_idx is not None else None,
        "individual_predictions": individual_predictions,
        "individual_probabilities": individual_probabilities,
    }

    return result

def main():
    # Set page configuration
    st.set_page_config(page_title="Detection of Alzheimer's Disease Using MRI", layout="wide", page_icon="🧠")

    # Sidebar layout
    st.sidebar.title("🧬 Detection of Alzheimer's Disease Using MRI")
    st.sidebar.markdown("""
    Upload a brain MRI scan below to begin:
    """)
    uploaded_file = st.sidebar.file_uploader("📤 Upload MRI Image", type=["png", "jpg", "jpeg"])

    st.title("🧠 Detection of Alzheimer's Disease Using MRI")
    st.markdown("""
    This AI-powered diagnostic assistant performs the following:
    - 🧹 Preprocesses brain MRI images
    - 📈 Extracts radiomic and geometric features
    - 🧠 Predicts cognitive status using multiple AI models
    - 🗳️ Aggregates decisions using ensemble voting
    """)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📷 Uploaded MRI Image", use_column_width=True)

        with st.spinner("🧠 Classifying MRI image..."):
            result = classify_mri_image(image_np, models_dir="models/")

        st.markdown("---")
        st.subheader("🧪 Diagnosis Result")
        st.success(f"**Predicted Class:** {result['predicted_label']}")

        st.markdown("---")
        st.subheader("📊 Ensemble Class Probabilities")
        prob_chart = pd.DataFrame({
            "Class": list(class_mapping.values()),
            "Probability": result["ensemble_probabilities"]
        })
        st.bar_chart(prob_chart.set_index("Class"))

        st.subheader("🤖 Individual Model Predictions")
        pred_df = pd.DataFrame.from_dict(result["individual_predictions"], orient="index", columns=["Prediction"])
        st.dataframe(pred_df.style.set_properties(**{'text-align': 'center'}))

        st.markdown("---")
        with st.expander("🔍 Show Raw Prediction Probabilities"):
            prob_df = pd.DataFrame(result["individual_probabilities"])
            st.dataframe(prob_df)

    else:
        st.info("Please upload a brain MRI image to begin analysis.")

# Only run when called directly
if __name__ == "__main__":
    main()
