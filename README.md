# Reverse-Image-Search-Using-DL
The **Reverse Image Search** system allows users to upload an image and find visually similar images from a pre-define dataset using deep learning. This technique relies on **feature extraction** using a pre-trained **ResNet50** model (without the top classification layers), converting images into high-dimensional **feature embeddings**.

The project follows a three-stage pipeline:
1) **Feature Extraction & Storage** (`app.py`): Extracts features from dataset images and stores them.
2) **Web Interface for Image Search** (`main.py`): Streamlit-based app for users to upload images and retrieve similar images.
3) **Testing and Debugging Tool** (`test.py`): CLI-based script to test reverse image search with sample images using OpenCV.

# Workflow & Architecture
**1) Feature Extraction (`app.py`)**
- Uses `ResNet50` (pre-trained on ImageNet) for feature extraction.
- Applies **GlobalMaxPooling2D** to reduce feature maps to 1D vectors.
- Extracted features are L2-normalized.
- All image embeddings are stored using `pickle` in `embeddings.pkl`, and corresponding filenames in `filenames.pkl`.

**2) Feature Extraction (`main.py`)**
- Allows users to upload an image through a web interface.
- Extracts features from the uploaded images using the same model.
- Finds top 10 similar images using **NearestNeighbors** (with Euclidean distance).
- Displays results in a reponsive layout using Streamlit columns.

**3) Feature Extraction (`test.py`)**
- Manually loads a test image and find 5 similar images from the dataset.
- Uses OpenCV (`cv2`) to display similar images one by one.

# Key Features
  - **Deep Learning-Based Search**: Utilizes pre-trained ResNet50 to extract meaningful, high-level image features.
  - **Efficient Similarity Matching**: Uses `NearestNeighbors` from Scikit-learn for fast, brute-force similarity search with Euclidean metric.
  - **Image Upload & Live Display**: Streamlit interface allows users to upload images and view search results instantly.
  - **Scalable Design**: Feature vectors are stored separately, allowing future extension to larger datasets.
  - **Modular Code Structure**: Clear separation of logic into preprocessing, model, web app, and testing scripts.
  - **L2 Normalization**: Ensures consistency in feature vector magnitudes for accurate similarity matching.
  - **Batch Feature Extraction**: Uses `tqdm` progress bar for extracting and storing embeddings in bulk.
  - **Pickled Embeddings**: Stores image features and filenames efficiently for fast retrieval during search.
  - **Testing Utility**: Includes `test.py` script for manual feature testing and result verification.
  - **Image Visualization with OpenCV**: Provides GUI-less testing via CLI using OpenCV's display functions.
 
# File Descriptions
`app.py`: Extracts and saves features from all dataset images. Run once for dataset preparation.

`main.py`: Streamlit web app for reverse image search. Main user interface.

`test.py`: Standalone test script to verify image search functionality from command line.

# Dependencies
- tensorflow
- numpy
- scikit-learn
- streamlit
- opencv-python
- Pillow
- pickle
- tqdm
