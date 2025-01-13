# Image Processing and Face Detection Project

## Overview

This project involves multiple image processing techniques using Python libraries, namely Pillow (PIL) and OpenCV. The functionalities include image manipulation, filtering, merging images, drawing shapes and text, as well as performing face detection. The project demonstrates practical applications of image processing in various contexts, such as resizing, rotating, enhancing images, and detecting human faces.

## Requirements

- Pillow
- OpenCV
- Python 3.x

## Features

### 1. **Loading and Displaying Images**
- The project loads two images (`1.jpg` and `2.jpeg`) and displays them using PIL.

### 2. **Saving Images with Different Extensions**
- One of the images is saved with a different file extension (e.g., `image1.png`).

### 3. **Rotating an Image**
- An image is rotated by an angle (in this case, 13 degrees) to simulate a custom rotation based on a birthdate.

### 4. **Applying Filters to Images**
- Multiple filters (e.g., `BLUR`, `SMOOTH_MORE`, `EMBOSS`, `SHARPEN`) are applied to both images and the results are displayed.

### 5. **Merging Two Images**
- The two images are resized and then merged side by side to create a new image.

### 6. **Cropping an Image**
- A section of one image is cropped and displayed.

### 7. **Drawing a Rectangle on an Image**
- A rectangle is drawn on one image using PIL's `ImageDraw` module.

### 8. **Adding Text to Images**
- A text watermark is added to both images and saved as new images.

### 9. **Creating a Contact Sheet**
- A contact sheet with 8 variations of the first image is created by adjusting the contrast and arranging them into a grid.

### 10. **Face Detection**
- Using OpenCV's Haar Cascade Classifiers, faces are detected in an image (`poza.jpg`), and rectangles are drawn around the detected faces. The execution time of the face detection function is also measured.

## File Structure

- `1.jpg`, `2.jpeg`, `poza.jpg`: Input images used in the project.
- `haarcascade_frontalface_default.xml`: Haar Cascade classifier for face detection.
- `haarcascade_eye.xml`: Haar Cascade classifier for eye detection.
