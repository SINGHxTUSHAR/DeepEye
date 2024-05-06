[![GitHub license](https://img.shields.io/github/license/SINGHxTUSHAR/DeepEye.svg)](https://github.com/SINGHxTUSHAR/DeepEye/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/SINGHxTUSHAR/DeepEye.svg)](https://GitHub.com/SINGHxTUSHAR/DeepEye/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/SINGHxTUSHAR/DeepEye.svg)](https://GitHub.com/SINGHxTUSHAR/DeepEye/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/SINGHxTUSHAR/DeepEye.svg)](https://GitHub.com/SINGHxTUSHAR/DeepEye/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


[![GitHub watchers](https://img.shields.io/github/watchers/SINGHxTUSHAR/DeepEye.svg?style=social&label=Watch&maxAge=2592000)](https://GitHub.com/SINGHxTUSHAR/DeepEye/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/SINGHxTUSHAR/DeepEye.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/SINGHxTUSHAR/DeepEye/network/)
[![GitHub stars](https://img.shields.io/github/stars/SINGHxTUSHAR/DeepEye.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/SINGHxTUSHAR/DeepEye/stargazers/)

[![Open in Visual Studio Code](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20Visual%20Studio%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://open.vscode.dev/SINGHxTUSHAR/DeepEye)

# DeepEye 👀:
![post-img](https://github.com/SINGHxTUSHAR/DeepEye/assets/113624520/fe278778-776b-4aa6-a3fa-59c687310c98)

This project tackles pedestrian detection on streets using a powerful combination of deep learning models.
YOLOv5, a state-of-the-art object detection model, efficiently identifies people in each video frame. DeepSORT then takes over,
associating these detections with unique tracks. This allows us to not just see people, but track their movement across frames, even in crowded or occluded scenes.
This robust system provides valuable insights into pedestrian activity on streets, making it useful for applications like traffic analysis, crowd monitoring, and even safety measures.

DeepSORT and YOLOv5: A Powerful Duo for Pedestrian Detection
Your project utilizes two key technologies for pedestrian detection on streets: DeepSORT and YOLOv5. Here's a breakdown of their individual roles:

### `YOLOv5 (You Only Look Once):`
* `Function:` YOLOv5 is a powerful object detection model. It excels at identifying objects in real-time within an image or video frame.
* `How it Works:` YOLOv5 divides the input image or frame into a grid. It predicts bounding boxes and confidence scores for each grid cell, indicating the presence and type of object (in this case, pedestrians) within that area.
* `Benefits:` YOLOv5 is known for its speed and accuracy, making it ideal for real-time applications like pedestrian detection on streets.

### `DeepSORT (Simple Online and Realtime Tracking):`
* `Function:` DeepSORT takes over after YOLOv5 identifies potential pedestrians. It's a multi-object tracking algorithm that keeps track of individual pedestrians across frames.
* `How it Works:` DeepSORT uses a combination of techniques for tracking:
   * Kalman Filter: This predicts the future location of previously detected pedestrians based on their movement in past frames.
   * Bounding Box Overlap: DeepSORT compares the predicted locations with new detections from YOLOv5 to confirm or adjust tracks.
   * Re-identification Features: DeepSORT extracts appearance features (like clothing color) to help associate detections with existing tracks, especially in cases of occlusion or crowded scenes.
* `Benefits:` DeepSORT allows you to not just detect pedestrians in individual frames, but also track their movement and path over time. This is crucial for applications like traffic analysis or crowd monitoring.

## Work-Flow ⚔️:
* `Data Preprocessing:`
   * If using videos, break them down into individual frames.
   * Annotate a subset of these frames with bounding boxes around pedestrians. This annotation data helps train YOLOv5 for accurate person detection.

* `YOLOv5 Training:`
Train the YOLOv5 model on the annotated data. This training process allows YOLOv5 to learn the visual characteristics of pedestrians.

* `Real-time Processing:`
   * In real-time, each frame (from video or image) gets fed into the trained YOLOv5 model.
   * YOLOv5 predicts bounding boxes and confidence scores for pedestrians identified in the frame.

* `DeepSORT Tracking:`
  * DeepSORT takes over from YOLOv5. It uses a Kalman Filter to predict the position of previously detected pedestrians in the current frame.
  * DeepSORT then compares these predicted positions with the new detections from YOLOv5.
  * Using a combination of bounding box overlap and DeepSORT's internal feature extractor for re-identification, it associates detections with existing tracks or creates new ones for unseen pedestrians.

* `Output:`
The final output displays the video or image with bounding boxes around detected pedestrians. Each box is assigned a unique ID that persists across frames, allowing you to visualize pedestrian movement.


## Reference 🧧:
* IEEE Xplore <a href="https://ieeexplore.ieee.org/abstract/document/9692002"> Documentation </a>
* ResearchGate <a href="https://www.researchgate.net/publication/363131723_Pedestrian_Target_Tracking_Based_On_DeepSORT_With_YOLOv5"> Documentation </a>


## Requirements💻 :

Ensure you have the following dependencies installed:

- Python (version 3.9.x || 3.7.x)
- IDE: VS-CODE or collab
- Virtual-environment(venv)
- Other dependencies (refer to the requirement.txt)

You can install the required Python packages using:

```bash
pip install -r requirement.txt
```

## Setup 💿:

- Clone the repository:
```bash
git clone https://github.com/SINGHxTUSHAR/DeepEye.git
cd DeepEye
```
- Create a virtual environment (optional but recommended):
```bash
python -m venv venv
```
- Activate the virtual environment:
  - On Windows:
   ```bash
   venv\Scripts\activate
   ```
  - On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

  ## Contributing 📌:
If you'd like to contribute to this project, please follow the standard GitHub fork and pull request process. Contributions, issues, and feature requests are welcome!

## Suggestion 🚀: 
If you have any suggestions for me related to this project, feel free to contact me at tusharsinghrawat.delhi@gmail.com or <a href="https://www.linkedin.com/in/singhxtushar/">LinkedIn</a>.

## License 📝:
This project is licensed under the <a href="https://github.com/SINGHxTUSHAR/DeepEye/blob/main/LICENSE">MIT License</a> - see the LICENSE file for details.
