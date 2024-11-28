# Video Tracking Tool

An interactive tool for testing various object tracking methods on synthetic videos. This app demonstrates:
1. **Optical Flow**: Tracks motion of multiple objects.
2. **Object Tracking**: Tracks a specific object using bounding boxes.
3. **Background Subtraction**: Detects motion by identifying objects against a static background.

## Features
- Choose from predefined synthetic videos (moving circles or rectangles).
- Test multiple tracking methods:
  - Optical Flow
  - Object Tracking
  - Background Subtraction
- Visualize results and download processed videos.


## How to Use
1. Clone the repository:
git clone https://github.com/Essilham/Video-Tracking-Tool.git
2. Install dependencies:
pip install -r requirements.txt
3. run the app:
streamlit run app.py
