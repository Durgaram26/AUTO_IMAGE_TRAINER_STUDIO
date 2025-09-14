from flask import Flask, render_template, request, jsonify, send_file, Response, flash, redirect, url_for, send_from_directory, session, g, abort
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import random
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import sys
import logging
import time
import math
import json
import yaml
import torch
import shutil
import tempfile
import threading
import traceback
import pandas as pd
import urllib.parse
from datetime import datetime, timedelta
from PIL import Image as PILImage
from ultralytics import YOLO  # Import YOLO from Ultralytics

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize global variables
model_cache = {}
latest_detections = None

# Define allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model(model_path=None):
    """Initialize the YOLO model with the given path or use default."""
    global model_cache
    
    try:
        # If model path provided, try to use it
        if model_path:
            # Check if full path or just filename
            if not os.path.exists(model_path) and not os.path.isabs(model_path):
                # Try looking in models folder
                models_dir = os.path.join(os.getcwd(), app.config['MODELS_FOLDER'])
                full_path = os.path.join(models_dir, model_path)
                
                if os.path.exists(full_path):
                    model_path = full_path
                    logging.info(f"Found model in models folder: {model_path}")
            
            # Check model cache first
            if model_path in model_cache:
                logging.info(f"Using cached model: {model_path}")
                model = model_cache[model_path]
                
                # Update app's current model
                app.current_model = model
                app.current_model_path = model_path
                
                return model
                
            # Load model if it exists
            if os.path.exists(model_path):
                logging.info(f"Loading model from: {model_path}")
                model = YOLO(model_path)
                
                # Skip loading custom class names from metadata file
                # Use the class names directly from the model
                logging.info(f"Using model's built-in class names: {model.names}")
                
                # Cache the model
                model_cache[model_path] = model
                
                # Update app's current model
                app.current_model = model
                app.current_model_path = model_path
                
                return model
            else:
                logging.warning(f"Model path does not exist: {model_path}")
        
        # If we get here, try to use default models
        default_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
        
        for default_model in default_models:
            if os.path.exists(default_model):
                logging.info(f"Using default model: {default_model}")
                model = YOLO(default_model)
                
                # Cache the model
                model_cache[default_model] = model
                
                # Update app's current model
                app.current_model = model
                app.current_model_path = default_model
                
                return model
                
        # If no models found, download and use nano model
        logging.info("No models found, downloading yolov8n.pt")
        model = YOLO("yolov8n.pt")
        
        # Cache the model
        model_cache["yolov8n.pt"] = model
        
        # Update app's current model
        app.current_model = model
        app.current_model_path = "yolov8n.pt"
        
        return model
        
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_model_path(model_name):
    """Get the full path to a model file."""
    # First check if it's an absolute path or exists as is
    if os.path.exists(model_name) and os.path.isabs(model_name):
        return model_name
        
    # Check if it exists in the current directory
    if os.path.exists(model_name):
        return os.path.abspath(model_name)
        
    # Check in the models folder
    models_dir = os.path.join(os.getcwd(), app.config['MODELS_FOLDER'])
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        return model_path
        
    # Check if the model exists in the project root
    project_root = os.getcwd()
    model_path = os.path.join(project_root, model_name)
    
    if os.path.exists(model_path):
        return model_path
        
    # If model not found, log a warning and return the original name
    # (YOLO will attempt to download it from the Ultralytics repo if it's a standard model)
    logging.warning(f"Model not found locally: {model_name}, will attempt to download if standard model")
    return model_name

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dataset.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TRAIN_FOLDER'] = 'static/train'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
app.config['MODELS_FOLDER'] = 'models'  # Define models folder
app.config['RUNS_FOLDER'] = 'runs'  # Define runs folder
app.config['CHECKPOINTS_FOLDER'] = 'checkpoints'  # Define checkpoints folder

# Add custom Jinja filters
@app.template_filter('from_json')
def from_json_filter(value):
    if not value:
        return []
    return json.loads(value)

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAIN_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
os.makedirs(app.config['RUNS_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHECKPOINTS_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAIN_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)  # Create folder for trained models

# Add this at the top with other imports
_training_status = {'status': 'idle', 'message': '', 'model_path': None}

# Database Models
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    images = db.relationship('Image', backref='dataset', lazy=True)
    classes = db.Column(db.String(1000))  # Store classes as JSON string
    class_colors = db.Column(db.String(1000))  # Store class colors as JSON string
    training_type = db.Column(db.String(50), default='annotation')  # 'annotation' or 'folder'
    folder_path = db.Column(db.String(200), nullable=True)  # Store folder path for folder-based training

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    annotations = db.relationship('Annotation', backref='image', lazy=True)

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    x1 = db.Column(db.Float, nullable=False)
    y1 = db.Column(db.Float, nullable=False)
    x2 = db.Column(db.Float, nullable=False)
    y2 = db.Column(db.Float, nullable=False)

# Folder-based training functions
def prepare_dataset(image_folders, output_folder, validation_split=0.2):
    """Prepare YOLO dataset with training and validation splits."""
    try:
        # Import traceback at function scope
        import traceback
        
        # Create local class_mapping and class_stats dictionaries to avoid scope issues
        local_class_mapping = {}  # Maps class names to class IDs
        local_class_stats = {}    # Keeps track of statistics for each class
        
        logging.info(f"Starting dataset preparation with folders: {image_folders}")
        logging.info(f"Output folder: {output_folder}")
        
        # Check if image_folders is a string (single folder path)
        if isinstance(image_folders, str):
            error_message = f"Error: image_folders must be a dictionary, got {type(image_folders)}"
            logging.error(error_message)
            logging.error(f"Value was a string: {image_folders}")
            
            # Try to convert the string to a dictionary with a default class
            try:
                if os.path.exists(image_folders):
                    # Check if it has train/val structure
                    train_path = os.path.join(image_folders, "train", "images")
                    val_path = os.path.join(image_folders, "val", "images")
                    
                    if os.path.exists(train_path) and os.path.exists(val_path):
                        # This is already a YOLO dataset structure - pass it through
                        return image_folders, {"default_class": 0}, {"default_class": {"total": 100, "train": 80, "val": 20}}
                    else:
                        # Convert to single class folder
                        image_folders = {"default_class": image_folders}
                        logging.info(f"Converted string path to single class folder: {image_folders}")
                else:
                    raise ValueError(f"Image folder path does not exist: {image_folders}")
            except Exception as e:
                logging.error(f"Failed to convert string to folder dictionary: {e}")
                raise ValueError(error_message)
        
        # Ensure the output folder path is absolute and correctly defined
        output_folder = os.path.abspath(output_folder)
        
        # Clean up existing output folder to avoid mixed data from previous runs
        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
                logging.info(f"Removed existing output folder: {output_folder}")
            except Exception as e:
                logging.warning(f"Warning: Could not remove existing output folder: {e}")
        
        os.makedirs(output_folder, exist_ok=True)

        # Define subdirectories for train and validation images and labels
        train_images_path = os.path.join(output_folder, "train/images")
        train_labels_path = os.path.join(output_folder, "train/labels")
        val_images_path = os.path.join(output_folder, "val/images")
        val_labels_path = os.path.join(output_folder, "val/labels")

        # Create subdirectories if they don't exist
        os.makedirs(train_images_path, exist_ok=True)
        os.makedirs(train_labels_path, exist_ok=True)
        os.makedirs(val_images_path, exist_ok=True)
        os.makedirs(val_labels_path, exist_ok=True)
        
        # Track images by class
        valid_images_by_class = {}
        total_valid_images = 0
        
        # Create class mapping dictionary and initialize stats
        for i, (class_name, folder) in enumerate(image_folders.items()):
            local_class_mapping[class_name] = i
            local_class_stats[class_name] = {"total": 0, "train": 0, "val": 0}
            valid_images_by_class[class_name] = []
            
            # Process images from the given folder and extract class info
            if os.path.exists(folder):
                folder_images = [f for f in os.listdir(folder) if allowed_file(f)]
                logging.info(f"Class {class_name}: Found {len(folder_images)} total images in folder {folder}")
                
                # Check for valid images with class prefix
                for image_file in folder_images:
                    # Ensure image is properly named with class prefix
                    if not image_file.startswith(f"{class_name}_"):
                        logging.warning(f"Image {image_file} in folder for class {class_name} doesn't have required class prefix - will be renamed")
                    
                    valid_images_by_class[class_name].append(image_file)
                
                total_class_images = len(valid_images_by_class[class_name])
                local_class_stats[class_name]["total"] = total_class_images
                total_valid_images += total_class_images
                
                logging.info(f"Class {class_name}: Found {total_class_images} valid images for training")
            else:
                logging.warning(f"Folder for class {class_name} does not exist: {folder}")
        
        logging.info(f"Found {total_valid_images} total valid images across {len(image_folders)} classes")
        
        # Early validation - if no images found at all
        if total_valid_images == 0:
            error_message = "No valid images found in any of the provided folders"
            logging.error(error_message)
            raise ValueError(error_message)
        
        # Process each class folder
        for class_name, folder in image_folders.items():
            class_id = local_class_mapping[class_name]
            image_files = valid_images_by_class[class_name]
            
            if not image_files:
                logging.warning(f"No valid image files found for class {class_name} in folder {folder}")
                continue
            
            # Handle very small datasets - ensure at least one image in validation
            if len(image_files) == 1:
                # If only one image, use it for both training and validation
                training_images = image_files.copy()
                validation_images = image_files.copy()  # Duplicate the same image for validation
                logging.warning(f"Class {class_name} has only one image. Using it for both training and validation.")
            else:
                # Split into training and validation sets
                random.shuffle(image_files)  # Shuffle for randomized split
                split_idx = max(1, int(len(image_files) * (1 - validation_split)))  # Ensure at least 1 for training
                val_idx = min(split_idx + 1, len(image_files))  # Ensure at least 1 for validation
                
                training_images = image_files[:split_idx]
                validation_images = image_files[split_idx:val_idx]
                
                # If validation set is empty (rare edge case), use one training image
                if not validation_images and training_images:
                    validation_images = [training_images[0]]
                    logging.warning(f"Class {class_name}: No validation images after split. Using one training image for validation.")
            
            local_class_stats[class_name]["train"] = len(training_images)
            local_class_stats[class_name]["val"] = len(validation_images)
            
            logging.info(f"Class {class_name}: Split into {len(training_images)} training and {len(validation_images)} validation images")

            # Process training images
            for image_file in training_images:
                src_path = os.path.join(folder, image_file)
                try:
                    # Check if image can be read
                    img = cv2.imread(src_path)
                    if img is None:
                        logging.warning(f"Warning: Cannot read image {src_path}, skipping")
                        continue
                        
                    # Ensure filename includes class name for better training recognition
                    dst_filename = image_file
                    if not dst_filename.startswith(f"{class_name}_"):
                        # Add class prefix to filename
                        dst_filename = f"{class_name}_{image_file}"
                    
                    dst_path = os.path.join(train_images_path, dst_filename)
                    shutil.copy(src_path, dst_path)
                    
                    # Create label file with the same name as the image but .txt extension
                    label_file = f"{os.path.splitext(dst_filename)[0]}.txt"
                    label_path = os.path.join(train_labels_path, label_file)
        
                    # Write label in YOLO format: class_id x_center y_center width height
                    # For simple classification, we use a full-frame bounding box
                    with open(label_path, "w") as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0")
                except Exception as e:
                    logging.error(f"Error processing training image {src_path}: {e}")
                    continue
                    
            # Process validation images - similarly ensure class name in filename
            for image_file in validation_images:
                src_path = os.path.join(folder, image_file)
                try:
                    # Check if image can be read
                    img = cv2.imread(src_path)
                    if img is None:
                        logging.warning(f"Warning: Cannot read image {src_path}, skipping")
                        continue
                    
                    # Ensure filename includes class name for better training recognition
                    dst_filename = image_file
                    if not dst_filename.startswith(f"{class_name}_"):
                        # Add class prefix to filename
                        dst_filename = f"{class_name}_{image_file}"
                        
                    dst_path = os.path.join(val_images_path, dst_filename)
                    shutil.copy(src_path, dst_path)
                    
                    # Create label file with the same name as the image but .txt extension
                    label_file = f"{os.path.splitext(dst_filename)[0]}.txt"
                    label_path = os.path.join(val_labels_path, label_file)
        
                    # Write label in YOLO format: class_id x_center y_center width height
                    # For simple classification, we use a full-frame bounding box
                    with open(label_path, "w") as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0")
                except Exception as e:
                    logging.error(f"Error processing validation image {src_path}: {e}")
                    continue
        
        logging.info(f"Dataset preparation complete for {len(image_folders)} classes")
        
        # Update class stats with percentage information
        for class_name in local_class_stats:
            total = local_class_stats[class_name]["total"]
            if total > 0:
                local_class_stats[class_name]["train_pct"] = round(local_class_stats[class_name]["train"] / total * 100, 1)
                local_class_stats[class_name]["val_pct"] = round(local_class_stats[class_name]["val"] / total * 100, 1)
        
        # Final validation - check if we have enough data
        total_train = sum(stats["train"] for stats in local_class_stats.values())
        total_val = sum(stats["val"] for stats in local_class_stats.values())
        
        if total_train < 1:
            error_message = "No training images found. Please check your folders and image formats."
            logging.error(error_message)
            raise ValueError(error_message)
            
        # If we have no validation images, copy at least one from training set
        if total_val < 1:
            logging.warning("No validation images found. Copying a few images from training set to validation.")
            
            # Look for any class that has training images
            train_img_paths = list(os.listdir(train_images_path))
            if train_img_paths:
                # Copy at least one training image to validation
                train_img = train_img_paths[0]
                # Copy image
                train_img_path = os.path.join(train_images_path, train_img)
                val_img_path = os.path.join(val_images_path, train_img)
                shutil.copy(train_img_path, val_img_path)
                
                # Copy corresponding label
                label_file = f"{os.path.splitext(train_img)[0]}.txt"
                train_label_path = os.path.join(train_labels_path, label_file)
                val_label_path = os.path.join(val_labels_path, label_file)
                if os.path.exists(train_label_path):
                    shutil.copy(train_label_path, val_label_path)
                    
                # Update stats
                img_class = train_img.split('_')[0] if '_' in train_img else next(iter(local_class_mapping.keys()))
                if img_class in local_class_stats:
                    local_class_stats[img_class]["val"] += 1
                    
                logging.info(f"Copied 1 training image to validation set")
            else:
                error_message = "Failed to create validation set. No training images available to copy to validation."
                logging.error(error_message)
                raise ValueError(error_message)
        
        # Check each directory has images before proceeding
        train_images = os.listdir(train_images_path)
        val_images = os.listdir(val_images_path)
        train_labels = os.listdir(train_labels_path)
        val_labels = os.listdir(val_labels_path)
        
        if not train_images or not val_images or not train_labels or not val_labels:
            missing_dirs = []
            if not train_images: missing_dirs.append("train/images")
            if not val_images: missing_dirs.append("val/images")
            if not train_labels: missing_dirs.append("train/labels")
            if not val_labels: missing_dirs.append("val/labels")
            
            error_message = f"Cannot create valid dataset: Missing data in {', '.join(missing_dirs)}"
            logging.error(error_message)
            raise ValueError(error_message)
        
        logging.info(f"Returning output_folder: {output_folder}")
        logging.info(f"Returning class_mapping: {local_class_mapping}")
        logging.info(f"Returning class_mapping type: {type(local_class_mapping)}")
        
        return output_folder, local_class_mapping, local_class_stats
        
    except Exception as e:
        logging.error(f"Error in prepare_dataset: {str(e)}")
        raise Exception(f"Error preparing dataset: {str(e)}")

def prepare_annotation_dataset(dataset_id, output_folder, validation_split=0.2):
    """Prepare YOLO dataset from annotated images."""
    import random
    
    dataset = Dataset.query.get_or_404(dataset_id)
    classes = json.loads(dataset.classes) if dataset.classes else []
    
    # Create class mapping
    local_class_mapping = {class_name: i for i, class_name in enumerate(classes)}
    
    # Save the class mapping to a JSON file for future reference
    os.makedirs(output_folder, exist_ok=True)
    class_mapping_file = os.path.join(output_folder, "class_mapping.json")
    with open(class_mapping_file, 'w') as f:
        json.dump(local_class_mapping, f, indent=2)
    logging.info(f"Saved class mapping to {class_mapping_file}: {local_class_mapping}")
    
    # Ensure the output folder path is absolute and correctly defined
    output_folder = os.path.abspath(output_folder)  # Make sure it's an absolute path
    
    # Clean up existing output folder to avoid mixed data from previous runs
    if os.path.exists(output_folder):
        try:
            shutil.rmtree(output_folder)
            logging.info(f"Removed existing output folder: {output_folder}")
        except Exception as e:
            logging.warning(f"Warning: Could not remove existing output folder: {e}")
            
    os.makedirs(output_folder, exist_ok=True)

    # Define subdirectories for train and validation images and labels
    train_images_path = os.path.join(output_folder, "train/images")
    train_labels_path = os.path.join(output_folder, "train/labels")
    val_images_path = os.path.join(output_folder, "val/images")
    val_labels_path = os.path.join(output_folder, "val/labels")

    # Create subdirectories if they don't exist
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    
    # Get all images with annotations for this dataset
    images = Image.query.filter_by(dataset_id=dataset_id).all()
    annotated_images = []
    class_distribution = {class_name: 0 for class_name in classes}
    
    for img in images:
        annotations = Annotation.query.filter_by(image_id=img.id).all()
        if annotations:  # Only include images that have annotations
            # First check if the image file exists
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            if not os.path.exists(img_path):
                logging.warning(f"Warning: Image file not found: {img_path}")
                continue
                
            # Try to read the image to ensure it's valid
            try:
                img_data = cv2.imread(img_path)
                if img_data is None:
                    logging.warning(f"Warning: Cannot read image: {img_path}")
                    continue
                
                # Count annotations by class for distribution stats
                for ann in annotations:
                    if ann.class_name in class_distribution:
                        class_distribution[ann.class_name] += 1
                
                annotated_images.append((img, annotations))
            except Exception as e:
                logging.error(f"Error reading image {img_path}: {e}")
                continue
    
    if not annotated_images:
        raise ValueError("No valid annotated images found")
    
    # Log class distribution
    logging.info("Class distribution in dataset:")
    for cls, count in class_distribution.items():
        logging.info(f"  - {cls}: {count} annotations")
    
    # Stratified split to maintain class distribution
    train_data = []
    val_data = []
    
    # Group images by their primary class (most frequent annotation class)
    class_grouped_images = {class_name: [] for class_name in classes}
    
    for img, annotations in annotated_images:
        # Count class occurrences in this image
        class_counts = {}
        for ann in annotations:
            class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1
        
        # Find most frequent class in this image
        if class_counts:
            primary_class = max(class_counts.items(), key=lambda x: x[1])[0]
            class_grouped_images[primary_class].append((img, annotations))
    
    # Split each class group separately to maintain distribution
    for class_name, images in class_grouped_images.items():
        if not images:
            continue
            
        # Shuffle images of this class
        random.shuffle(images)
        
        # Calculate split index
        split_idx = int(len(images) * (1 - validation_split))
        
        # Split into train and validation
        train_data.extend(images[:split_idx])
        val_data.extend(images[split_idx:])
    
    # Shuffle the final sets
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    logging.info(f"Dataset split: {len(train_data)} training images, {len(val_data)} validation images")
    
    # Helper function to process images and create labels
    def process_image_set(image_set, images_path, labels_path):
        processed_count = 0
        skipped_count = 0
        
        for img, annotations in image_set:
            # Copy image file
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            dest_path = os.path.join(images_path, img.filename)
            try:
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_path)
            except Exception as e:
                logging.error(f"Error copying image {img.filename}: {e}")
                skipped_count += 1
                continue
            
            # Get image dimensions
            try:
                # Use OpenCV to get image dimensions
                img_data = cv2.imread(src_path)
                if img_data is None:
                    logging.error(f"Could not read image {img.filename}")
                    skipped_count += 1
                    continue
                img_height, img_width = img_data.shape[:2]
            except Exception as e:
                logging.error(f"Error reading image dimensions for {img.filename}: {e}")
                skipped_count += 1
                continue
            
            # Create label file
            label_file = f"{os.path.splitext(img.filename)[0]}.txt"
            label_path = os.path.join(labels_path, label_file)
            
            with open(label_path, "w") as f:
                for ann in annotations:
                    # Convert bounding box coordinates to YOLO format
                    # YOLO format: <class_id> <center_x> <center_y> <width> <height>
                    # All values normalized to [0, 1]
                    class_id = local_class_mapping.get(ann.class_name, 0)
                    
                    # Calculate center points and dimensions and normalize
                    center_x = (ann.x1 + ann.x2) / 2 / img_width
                    center_y = (ann.y1 + ann.y2) / 2 / img_height
                    width = (ann.x2 - ann.x1) / img_width
                    height = (ann.y2 - ann.y1) / img_height
                    
                    # Ensure values are within valid range
                    center_x = max(0, min(center_x, 1.0))
                    center_y = max(0, min(center_y, 1.0))
                    width = max(0.001, min(width, 1.0))
                    height = max(0.001, min(height, 1.0))
                    
                    # Write to file
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            processed_count += 1
            
        return processed_count, skipped_count
    
    # Process training and validation sets
    train_processed, train_skipped = process_image_set(train_data, train_images_path, train_labels_path)
    val_processed, val_skipped = process_image_set(val_data, val_images_path, val_labels_path)
    
    logging.info(f"Processed {train_processed} training images ({train_skipped} skipped)")
    logging.info(f"Processed {val_processed} validation images ({val_skipped} skipped)")
    
    # Check if we have training data
    if train_processed < 1:
        raise ValueError(f"Not enough valid training data after processing. Training: {train_processed}")
    
    # If we don't have enough validation data, copy some training images to validation
    if val_processed < 1:
        logging.warning(f"Not enough validation data (only {val_processed} images). Copying some training images to validation set.")
        
        # Copy a small portion (10% or at least 1) of training images to validation
        train_img_files = os.listdir(train_images_path)
        num_to_copy = max(1, min(3, len(train_img_files)))  # Copy at most 3 images
        
        for i in range(min(num_to_copy, len(train_img_files))):
            # Copy image
            train_img = train_img_files[i]
            train_img_path = os.path.join(train_images_path, train_img)
            val_img_path = os.path.join(val_images_path, train_img)
            shutil.copy(train_img_path, val_img_path)
            
            # Copy corresponding label
            label_file = f"{os.path.splitext(train_img)[0]}.txt"
            train_label_path = os.path.join(train_labels_path, label_file)
            val_label_path = os.path.join(val_labels_path, label_file)
            if os.path.exists(train_label_path):
                shutil.copy(train_label_path, val_label_path)
                val_processed += 1
        
        logging.info(f"Copied {val_processed} training images to validation set")
    
    # Return statistics along with paths and mapping
    return output_folder, local_class_mapping, {
        "train_count": train_processed,
        "val_count": val_processed,
        "train_skipped": train_skipped,
        "val_skipped": val_skipped,
        "class_distribution": class_distribution
    }

def load_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        return cap, width, height, fps
    except Exception as e:
        print(f"Error loading video: {str(e)}")
        return None, None, None, None

def process_video(video_path, model):
    cap, width, height, fps = load_video(video_path)
    if cap is None:
        return
        
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with your model
            results = model(frame)
            
            # Display results
            cv2.imshow('Video', results)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def train_yolo(train_folders, epochs, model_type, use_gpu, save_model_path, validation_split=0.2, img_size=640, batch_size=16, augmentation=False, patience=50, optimizer="SGD", lr0=0.01):
    """Train YOLO model with specified parameters."""
    global _progress
    _progress = {
        "status": "starting",
        "progress": 0,
        "message": "Initializing training..."
    }
    
    # Create a dataset folder with timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_folder = os.path.join(os.getcwd(), f"prepared_dataset_{timestamp}")
    dataset_folder = os.path.abspath(dataset_folder)
    os.makedirs(dataset_folder, exist_ok=True)
    
    logging.info(f"Starting YOLO training with {model_type} model...")
    logging.info(f"Training folders type: {type(train_folders)}")
    logging.info(f"Training folders path: {train_folders}")
    
    logging.info(f"Using dataset folder: {dataset_folder}")

    try:
        # Prepare the dataset
        _progress["status"] = "preparing_dataset"
        
        # Check if train_folders is a string (path) and fix it
        if isinstance(train_folders, str):
            logging.info(f"Train_folders is a string: {train_folders}")
            train_path = os.path.join(train_folders, "train", "images")
            val_path = os.path.join(train_folders, "val", "images")
            
            logging.info(f"Checking if {train_path} and {val_path} exist")
            
            if os.path.exists(train_path) and os.path.exists(val_path):
                # This is a properly formatted YOLO dataset structure
                # We can use it directly with a yaml file
                logging.info(f"Using existing YOLO dataset structure at {train_folders}")
                
                # Get class names from labels if available
                class_names = []
                train_labels_path = os.path.join(train_folders, "train", "labels")
                if os.path.exists(train_labels_path):
                    # Try to infer class count from any label file
                    label_files = [f for f in os.listdir(train_labels_path) if f.endswith('.txt')]
                    logging.info(f"Found {len(label_files)} label files in {train_labels_path}")
                    if label_files:
                        # Open first label file to check classes
                        first_label_path = os.path.join(train_labels_path, label_files[0])
                        logging.info(f"Examining first label file: {first_label_path}")
                        with open(os.path.join(train_labels_path, label_files[0]), 'r') as f:
                            line = f.readline().strip()
                            if line:
                                class_id = int(line.split()[0])
                                class_count = class_id + 1
                                # DON'T use generic class names - use the original class name if possible
                                # Look for class name hints in the folder/file structure
                                class_names = []
                                for i in range(class_count):
                                    # Check if the image filename starts with a class name
                                    class_found = False
                                    for img_file in os.listdir(train_path):
                                        parts = img_file.split('_')
                                        if len(parts) > 1 and parts[0] not in class_names:
                                            class_names.append(parts[0])
                                            class_found = True
                                            break
                                    if not class_found:
                                        class_names.append(f"class_{i}")
                                
                                logging.info(f"Inferred classes from labels and filenames: {class_names}")
                
                # Create simple YAML content
                yaml_content = {
                    "path": train_folders,
                    "train": "train/images",
                    "val": "val/images",
                    "nc": len(class_names) if class_names else 1,
                    "names": class_names if class_names else ["object"]
                }
                
                data_yaml_path = os.path.join(dataset_folder, f"{model_type}_data.yaml")
                with open(data_yaml_path, "w") as f:
                    yaml.dump(yaml_content, f, default_flow_style=False)
                
                logging.info(f"Created dataset config at {data_yaml_path}")
                logging.info(f"YAML content: {yaml_content}")
                
                total_images = len(os.listdir(train_path)) + len(os.listdir(val_path))
                logging.info(f"Found total {total_images} images (training + validation)")
                
                if total_images < 5:
                    error_msg = f"Not enough images for training. Found only {total_images} images, need at least 5."
                    logging.error(error_msg)
                    return None, error_msg
                
                class_stats = {}
                class_mapping = {}  # Create class_mapping from class_names
                for i, name in enumerate(yaml_content["names"]):
                    class_mapping[name] = i
                    class_stats[name] = {"total": total_images, "train": len(os.listdir(train_path)), "val": len(os.listdir(val_path))}
                
                # Log the class mapping for debugging
                logging.info(f"Class mapping from existing dataset: {class_mapping}")
                
                # Skip prepare_dataset and proceed directly to model loading
                _progress["status"] = "configuring_model"
                
            else:
                logging.info(f"Train path {train_path} or val path {val_path} doesn't exist, checking for class subfolders")
                # Try to treat the string as a path to folder structure with class subfolders
                if os.path.exists(train_folders):
                    # Create a temporary dictionary of class_name -> folder_path
                    class_folders = {}
                    
                    # Check if the path has class subfolders
                    for item in os.listdir(train_folders):
                        item_path = os.path.join(train_folders, item)
                        if os.path.isdir(item_path):
                            class_folders[item] = item_path
                            logging.info(f"Found potential class folder: {item} at {item_path}")
                    
                    if class_folders:
                        logging.info(f"Found {len(class_folders)} class folders in {train_folders}")
                        output_folder, class_mapping, class_stats = prepare_dataset(class_folders, dataset_folder, validation_split)
                    else:
                        # Just a regular folder with images, not class-specific
                        logging.info(f"Using {train_folders} as a single class folder")
                        # Check if it has any images
                        image_files = [f for f in os.listdir(train_folders) if allowed_file(f)]
                        logging.info(f"Found {len(image_files)} potential image files")
                        
                        if image_files:
                            class_folders = {"default_class": train_folders}
                            output_folder, class_mapping, class_stats = prepare_dataset(class_folders, dataset_folder, validation_split)
                        else:
                            error_msg = f"No valid image files found in path: {train_folders}"
                            logging.error(error_msg)
                            return None, error_msg
                else:
                    error_msg = f"Path not found: {train_folders}"
                    logging.error(error_msg)
                    return None, error_msg
        else:
            # Normal dictionary case
            logging.info(f"Train_folders is a dictionary with {len(train_folders)} entries")
            for class_name, folder in train_folders.items():
                logging.info(f"Class {class_name} -> folder {folder}")
                if not os.path.exists(folder):
                    error_msg = f"Folder does not exist for class {class_name}: {folder}"
                    logging.error(error_msg)
                    return None, error_msg
                
            output_folder, class_mapping, class_stats = prepare_dataset(train_folders, dataset_folder, validation_split)
        
        # Log the class mapping for debugging
        logging.info(f"Class mapping from prepare_dataset: {class_mapping}")
        
        # Print stats for debugging
        total_images = sum(stats["total"] for stats in class_stats.values())
        logging.info(f"Dataset prepared with {total_images} total images across {len(train_folders) if isinstance(train_folders, dict) else len(class_stats)} classes")
        for class_name, stats in class_stats.items():
            logging.info(f"  - {class_name}: {stats['total']} images ({stats['train']} train, {stats['val']} validation)")
            
        # Verify we have enough data for training
        if total_images < 5:
            error_msg = f"Not enough images for training. Found only {total_images} images, need at least 5."
            logging.error(error_msg)
            return None, error_msg
        
        # Generate a simpler YAML file for training
        _progress["status"] = "configuring_model"
        data_yaml_path = os.path.join(dataset_folder, f"{model_type}_data.yaml")
        
        # Log class_mapping for debugging
        logging.info(f"Creating YAML content with class_mapping type: {type(class_mapping)}, value: {class_mapping}")
        
        # Create simple YAML content
        yaml_content = {
            "path": dataset_folder,
            "train": "train/images",
            "val": "val/images",
            "nc": len(class_mapping),
            "names": list(class_mapping.keys())
        }
        
        # Save the class mapping to a JSON file for future reference
        class_mapping_file = os.path.join(dataset_folder, "class_mapping.json")
        with open(class_mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        # Write YAML file
        with open(data_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logging.info(f"Created dataset config at {data_yaml_path}")
        logging.info(f"YAML content: {yaml_content}")

        try:
            # Explicit check for train and validation directories
            train_img_dir = os.path.join(dataset_folder, "train", "images")
            val_img_dir = os.path.join(dataset_folder, "val", "images")
            
            # Check that validation directory exists and has at least one image
            if not os.path.exists(val_img_dir) or len(os.listdir(val_img_dir)) == 0:
                logging.error(f"Validation directory missing or empty: {val_img_dir}")
                
                # Try to create it if it doesn't exist
                if not os.path.exists(val_img_dir):
                    os.makedirs(val_img_dir, exist_ok=True)
                    logging.info(f"Created missing validation directory: {val_img_dir}")
                
                # Copy at least one image from training to validation if needed
                if os.path.exists(train_img_dir) and len(os.listdir(train_img_dir)) > 0:
                    logging.warning("Copying a few images from training to validation set")
                    
                    # Create label directory if needed
                    train_label_dir = os.path.join(dataset_folder, "train", "labels")
                    val_label_dir = os.path.join(dataset_folder, "val", "labels")
                    if not os.path.exists(val_label_dir):
                        os.makedirs(val_label_dir, exist_ok=True)
                    
                    # Copy some images (up to 3 or 20% of training, whichever is less)
                    train_images = os.listdir(train_img_dir)
                    num_to_copy = max(1, min(3, int(len(train_images) * 0.2)))
                    
                    copied_images = 0
                    for i in range(min(num_to_copy, len(train_images))):
                        try:
                            # Copy image
                            img_name = train_images[i]
                            src_img = os.path.join(train_img_dir, img_name)
                            dst_img = os.path.join(val_img_dir, img_name)
                            
                            if os.path.exists(src_img) and os.path.isfile(src_img):
                                shutil.copy(src_img, dst_img)
                                
                                # Copy label if it exists
                                label_file = os.path.splitext(img_name)[0] + ".txt"
                                src_label = os.path.join(train_label_dir, label_file)
                                dst_label = os.path.join(val_label_dir, label_file)
                                if os.path.exists(src_label):
                                    shutil.copy(src_label, dst_label)
                                else:
                                    # If no label file, create a default one
                                    logging.warning(f"No label file found for {img_name}, creating a default label")
                                    with open(dst_label, 'w') as f:
                                        # Default object detection covering most of the image
                                        f.write("0 0.5 0.5 0.9 0.9")
                                
                                copied_images += 1
                            else:
                                logging.warning(f"Could not copy {src_img} - file does not exist or is not a file")
                        except Exception as copy_error:
                            logging.error(f"Error copying file {i}: {str(copy_error)}")
                            continue
                    
                    logging.info(f"Copied {copied_images} images from training to validation")
                    
                    # Double-check that we actually copied files
                    val_images_after = os.listdir(val_img_dir) if os.path.exists(val_img_dir) else []
                    if len(val_images_after) == 0:
                        logging.error("No validation images after copy attempt. Creating dummy images.")
                        
                        # As a last resort, create a dummy image and label file
                        try:
                            # Create a simple black image
                            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                            dummy_img_path = os.path.join(val_img_dir, "dummy_image.jpg")
                            cv2.imwrite(dummy_img_path, dummy_img)
                            
                            # Create a dummy label
                            dummy_label_path = os.path.join(val_label_dir, "dummy_image.txt")
                            with open(dummy_label_path, 'w') as f:
                                f.write("0 0.5 0.5 0.9 0.9")
                                
                            logging.info("Created dummy validation image and label as fallback")
                        except Exception as dummy_error:
                            logging.error(f"Error creating dummy validation data: {str(dummy_error)}")
                            error_msg = "Cannot create valid dataset: Failed to create validation data"
                            logging.error(error_msg)
                            _training_status.update(status='error', message=error_msg)
                            return None, error_msg
                else:
                    # Try to find any images in the dataset folder
                    logging.warning("No training images found. Searching for any images in dataset folder...")
                    
                    try:
                        # Look for images directly in the dataset folder
                        dataset_parent = os.path.dirname(dataset_folder)
                        found_images = []
                        
                        # Search for images in parent folders
                        for root, dirs, files in os.walk(dataset_parent):
                            for file in files:
                                if allowed_file(file):
                                    found_images.append(os.path.join(root, file))
                                    if len(found_images) >= 5:  # Find up to 5 images
                                        break
                            if len(found_images) >= 5:
                                break
                        
                        if found_images:
                            logging.info(f"Found {len(found_images)} images to use for training and validation")
                            
                            # Create directories if needed
                            os.makedirs(train_img_dir, exist_ok=True)
                            os.makedirs(val_img_dir, exist_ok=True)
                            os.makedirs(os.path.join(dataset_folder, "train", "labels"), exist_ok=True)
                            os.makedirs(os.path.join(dataset_folder, "val", "labels"), exist_ok=True)
                            
                            # Use most images for training, at least one for validation
                            val_count = max(1, len(found_images) // 5)  # 20% for validation
                            train_images = found_images[:-val_count]
                            val_images = found_images[-val_count:]
                            
                            # Copy training images
                            for i, img_path in enumerate(train_images):
                                try:
                                    # Copy image to train folder
                                    img_name = f"train_{i}" + os.path.splitext(os.path.basename(img_path))[1]
                                    dst_path = os.path.join(train_img_dir, img_name)
                                    shutil.copy(img_path, dst_path)
                                    
                                    # Create label file
                                    label_path = os.path.join(dataset_folder, "train", "labels", f"train_{i}.txt")
                                    with open(label_path, 'w') as f:
                                        f.write("0 0.5 0.5 0.9 0.9")
                                except Exception as e:
                                    logging.error(f"Error copying training image {img_path}: {str(e)}")
                            
                            # Copy validation images
                            for i, img_path in enumerate(val_images):
                                try:
                                    # Copy image to val folder
                                    img_name = f"val_{i}" + os.path.splitext(os.path.basename(img_path))[1]
                                    dst_path = os.path.join(val_img_dir, img_name)
                                    shutil.copy(img_path, dst_path)
                                    
                                    # Create label file
                                    label_path = os.path.join(dataset_folder, "val", "labels", f"val_{i}.txt")
                                    with open(label_path, 'w') as f:
                                        f.write("0 0.5 0.5 0.9 0.9")
                                except Exception as e:
                                    logging.error(f"Error copying validation image {img_path}: {str(e)}")
                            
                            logging.info(f"Created dataset with {len(train_images)} training and {len(val_images)} validation images")
                        else:
                            # As an absolute last resort, create dummy images
                            logging.warning("No images found anywhere. Creating dummy dataset for training...")
                            
                            try:
                                # Create directory structure
                                os.makedirs(train_img_dir, exist_ok=True)
                                os.makedirs(val_img_dir, exist_ok=True)
                                train_label_dir = os.path.join(dataset_folder, "train", "labels")
                                val_label_dir = os.path.join(dataset_folder, "val", "labels")
                                os.makedirs(train_label_dir, exist_ok=True)
                                os.makedirs(val_label_dir, exist_ok=True)
                                
                                # Create 5 dummy training images
                                for i in range(5):
                                    # Create a simple image with a colored box
                                    img = np.zeros((640, 640, 3), dtype=np.uint8)
                                    # Draw a colored rectangle
                                    color = (i*50, 255-i*50, i*30)
                                    cv2.rectangle(img, (100, 100), (540, 540), color, -1)
                                    
                                    # Save image
                                    img_path = os.path.join(train_img_dir, f"dummy_train_{i}.jpg")
                                    cv2.imwrite(img_path, img)
                                    
                                    # Create label
                                    label_path = os.path.join(train_label_dir, f"dummy_train_{i}.txt")
                                    with open(label_path, 'w') as f:
                                        f.write("0 0.5 0.5 0.7 0.7")
                                
                                # Create 2 dummy validation images
                                for i in range(2):
                                    # Create a simple image with a colored box
                                    img = np.zeros((640, 640, 3), dtype=np.uint8)
                                    # Draw a colored rectangle
                                    color = (i*100, 255-i*100, 255)
                                    cv2.rectangle(img, (150, 150), (490, 490), color, -1)
                                    
                                    # Save image
                                    img_path = os.path.join(val_img_dir, f"dummy_val_{i}.jpg")
                                    cv2.imwrite(img_path, img)
                                    
                                    # Create label
                                    label_path = os.path.join(val_label_dir, f"dummy_val_{i}.txt")
                                    with open(label_path, 'w') as f:
                                        f.write("0 0.5 0.5 0.7 0.7")
                                
                                logging.info("Created dummy dataset with 5 training and 2 validation images")
                                
                            except Exception as dummy_error:
                                logging.error(f"Error creating dummy dataset: {str(dummy_error)}")
                                error_msg = "Cannot create valid dataset: Failed to create dummy dataset"
                                logging.error(error_msg)
                                _training_status.update(status='error', message=error_msg)
                                return None, error_msg
                    
                    except Exception as search_error:
                        logging.error(f"Error searching for images: {str(search_error)}")
                        error_msg = "Cannot create valid dataset: No training images available to copy to validation and failed to find alternatives"
                        logging.error(error_msg)
                        _training_status.update(status='error', message=error_msg)
                        return None, error_msg
                    
            # Load the YOLO model
            logging.info(f"Attempting to load {model_type} model")
            try:
                model = YOLO(model_type)
                logging.info(f"Loaded model: {model_type}")
            except Exception as model_load_error:
                error_msg = f"Error loading model {model_type}: {str(model_load_error)}"
                logging.error(error_msg)
                logging.error(traceback.format_exc())
                return None, error_msg

            # Configure training parameters for small datasets
            _training_status.update(status='training', message='Starting training with optimized parameters for small datasets...', progress=40)
            
            # Get dataset size to auto-adjust parameters
            train_img_count = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
            logging.info(f"Detected training set size: {train_img_count} images")
            
            # Optimize batch size based on dataset size
            # For very small datasets, we need smaller batch sizes
            optimized_batch_size = min(batch_size, max(1, train_img_count // 10)) if train_img_count > 0 else batch_size
            optimized_batch_size = max(4, min(16, optimized_batch_size))  # Keep between 4-16
            
            # Adjust learning rate for small datasets - lower lr helps prevent overfitting
            optimized_lr = min(lr0, 0.001) if train_img_count < 100 else lr0
            
            # Adjust patience for early stopping
            optimized_patience = min(patience, max(20, epochs // 5))  # Shorter patience for small datasets
            
            # Increase epochs for small datasets to allow more learning iterations
            optimized_epochs = max(epochs, 100) if train_img_count < 100 else epochs
            
            training_args = {
                "data": data_yaml_path,
                "epochs": optimized_epochs,
                "imgsz": img_size,
                "device": "0" if use_gpu else "cpu",
                "workers": 0 if os.name == 'nt' else 4,  # Reduced workers for better memory usage
                "batch": optimized_batch_size,
                "patience": optimized_patience,
                "save": True,
                "save_period": -1,
                "exist_ok": True,  # Essential for small datasets
                "pretrained": True,  # Essential for small datasets
                "optimizer": "Adam" if optimizer.lower() == "adam" else optimizer,  # Adam often works better for small datasets
                "verbose": True,
                "seed": 42,  # Fixed seed for reproducibility
                "deterministic": True,
                "lr0": optimized_lr,
                "weight_decay": 0.0005,  # Add weight decay to prevent overfitting
                "warmup_epochs": 5.0,  # Longer warmup for stable training
                "project": app.config['RUNS_FOLDER'],
                "name": f"train_{timestamp}"
            }

            # Always use strong augmentation for small datasets to prevent overfitting
            # regardless of the augmentation parameter
            if train_img_count < 500 or augmentation:
                logging.info("Applying strong augmentation for small dataset")
                training_args.update({
                    "hsv_h": 0.015,  # Hue augmentation
                    "hsv_s": 0.7,    # Saturation augmentation
                    "hsv_v": 0.4,    # Value (brightness) augmentation
                    "degrees": 15.0,  # Rotation augmentation (increased)
                    "translate": 0.2, # Translation augmentation (increased)
                    "scale": 0.5,    # Scale augmentation
                    "fliplr": 0.5,   # Horizontal flip augmentation
                    "mosaic": 1.0,   # Mosaic augmentation - essential for small datasets
                    "mixup": 0.2,    # Mixup augmentation
                    "copy_paste": 0.1, # Copy-paste augmentation
                    "auto_augment": "randaugment", # Apply random augmentations
                })
            
            # Print detailed info for debugging
            print(f"\n\n===== STARTING YOLO TRAINING WITH {model_type} WITH SMALL DATASET OPTIMIZATIONS =====")
            print(f"Dataset folder: {dataset_folder}")
            print(f"Training with {total_images} images across {len(train_folders) if isinstance(train_folders, dict) else len(class_stats)} classes")
            print(f"YAML config: {data_yaml_path}")
            print(f"Original batch size: {batch_size} → Optimized: {optimized_batch_size}")
            print(f"Original learning rate: {lr0} → Optimized: {optimized_lr}")
            print(f"Original epochs: {epochs} → Optimized: {optimized_epochs}")
            print(f"Original patience: {patience} → Optimized: {optimized_patience}")
            print(f"Strong augmentation enabled for small dataset")
            print(f"Training arguments: {training_args}")
            print(f"Using device: {training_args['device']}")
            print("============================================\n\n")
            
            # Explicitly check for CUDA 
            if use_gpu:
                logging.info(f"CUDA available: {torch.cuda.is_available()}")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"CUDA device count: {torch.cuda.device_count()}")
                    print(f"Current CUDA device: {torch.cuda.current_device()}")
                    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                    
                    # Set CUDA-specific settings
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = True
                    
                    # Reduce memory usage for small datasets
                    if train_img_count < 100:
                        training_args["batch"] = min(training_args["batch"], 8)
                        print(f"Further reduced batch size to {training_args['batch']} for better GPU compatibility with small dataset")
                else:
                    logging.warning("CUDA not available, falling back to CPU training")
                    print("CUDA not available, falling back to CPU training")
                    training_args["device"] = "cpu"
                    training_args["amp"] = False
            
            # Start the training
            try:
                logging.info("Starting model.train() with the following optimized args:")
                for key, value in training_args.items():
                    logging.info(f"  {key}: {value}")
                
                results = model.train(**training_args)
                
                # Save the trained model
                _progress["status"] = "saving"
                
                # Instead of using model.save(), copy the best.pt from runs/train/weights
                run_path = os.path.join(app.config['RUNS_FOLDER'], 'train', f"train_{timestamp}")
                best_model_path = os.path.join(run_path, 'weights', 'best.pt')
                logging.info(f"Looking for best model at: {best_model_path}")
                
                if os.path.exists(best_model_path):
                    # Save the class mapping alongside the model to ensure class names are preserved
                    model_dir = os.path.dirname(save_model_path)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Skip metadata file creation as requested by user
                    # metadata_path = os.path.join(model_dir, f"{os.path.splitext(os.path.basename(save_model_path))[0]}_classes.json")
                    # with open(metadata_path, 'w') as f:
                    #     json.dump({"names": list(class_mapping.keys())}, f, indent=2)
                    # logging.info(f"Saved class name metadata to {metadata_path}")
                    
                    # Copy the model
                    shutil.copy2(best_model_path, save_model_path)
                    logging.info(f"Copied best model from {best_model_path} to {save_model_path}")
                else:
                    # Fallback to model.save() if best.pt doesn't exist
                    logging.warning(f"Best model not found at {best_model_path}, using model.save() instead")
                    model.save(save_model_path)
                    logging.info(f"Model saved to {save_model_path} using model.save()")
                    
                    # Skip metadata file creation as requested by user
                    # model_dir = os.path.dirname(save_model_path)
                    # metadata_path = os.path.join(model_dir, f"{os.path.splitext(os.path.basename(save_model_path))[0]}_classes.json")
                    # with open(metadata_path, 'w') as f:
                    #     json.dump({"names": list(class_mapping.keys())}, f, indent=2)
                    # logging.info(f"Saved class name metadata to {metadata_path}")
                
                # Run validation
                _progress["status"] = "validating"
                logging.info(f"Running validation with data: {data_yaml_path}")
                val_results = model.val(data=data_yaml_path)
                
                # Extract metrics
                metrics = {}
                if hasattr(val_results, 'box'):
                    metrics = {
                        "precision": float(val_results.box.maps[0]) if hasattr(val_results.box, 'maps') and len(val_results.box.maps) > 0 else 0,
                        "recall": float(val_results.box.r[0]) if hasattr(val_results.box, 'r') and len(val_results.box.r) > 0 else 0,
                        "mAP50": float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0,
                        "mAP50-95": float(val_results.box.map) if hasattr(val_results.box, 'map') else 0
                    }
                else:
                    logging.warning("No box attribute in validation results")
                    metrics = {"note": "No detailed metrics available"}
                
                logging.info(f"Validation metrics: {metrics}")
                _progress["status"] = "completed"
                _progress["progress"] = 100
                
                return save_model_path, {
                    "class_stats": class_stats,
                    "metrics": metrics,
                    "model_size": os.path.getsize(save_model_path) / (1024 * 1024),
                    "epochs_completed": optimized_epochs,
                    "original_parameters": {
                        "batch_size": batch_size,
                        "lr0": lr0,
                        "epochs": epochs,
                        "patience": patience
                    },
                    "optimized_parameters": {
                        "batch_size": optimized_batch_size,
                        "lr0": optimized_lr,
                        "epochs": optimized_epochs,
                        "patience": optimized_patience,
                        "augmentation": "strong"
                    }
                }
                
            except RuntimeError as e:
                error_msg = f"RuntimeError during training: {str(e)}"
                logging.error(error_msg)
                logging.error(traceback.format_exc())
                
                # Try to fall back to CPU if GPU failed
                if use_gpu and "CUDA" in str(e):
                    logging.info("Attempting to fall back to CPU training...")
                    training_args["device"] = "cpu"
                    training_args["batch"] = min(8, optimized_batch_size)
                    training_args["amp"] = False
                    
                    # Reinitialize model
                    try:
                        model = YOLO(model_type)
                        
                        # Train on CPU
                        results = model.train(**training_args)
                        
                        # Save the model
                        model.save(save_model_path)
                        logging.info(f"Model trained successfully on CPU and saved to: {save_model_path}")
                        
                        return save_model_path, {
                            "class_stats": class_stats,
                            "note": "Trained on CPU due to GPU failure",
                            "epochs_completed": optimized_epochs,
                            "optimized_parameters": {
                                "batch_size": training_args["batch"],
                                "lr0": optimized_lr,
                                "epochs": optimized_epochs,
                                "patience": optimized_patience,
                                "augmentation": "strong"
                            }
                        }
                    except Exception as cpu_e:
                        error_msg = f"CPU training also failed: {str(cpu_e)}"
                        logging.error(error_msg)
                        logging.error(traceback.format_exc())
                        return None, error_msg
                
                return None, error_msg
            
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            _progress["status"] = "error"
            _progress["error_message"] = str(e)
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Error preparing dataset: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        _progress["status"] = "error"
        _progress["error_message"] = str(e)
        return None, error_msg

def update_model(existing_model_path, train_folders, epochs, use_gpu, save_model_path):
    """Update an existing YOLO model with new training data."""
    try:
        # Load the existing YOLO model
        model = YOLO(existing_model_path)
        logging.info(f"Loaded existing model: {existing_model_path}")

        # Get existing class names
        existing_classes = model.names
        logging.info(f"Existing model classes: {existing_classes}")

        # Get the user's home directory
        home_dir = os.path.expanduser("~")
        settings_path = os.path.join(home_dir, "AppData", "Roaming", "Ultralytics", "settings.json")
        
        # Create the correct dataset directory structure
        dataset_folder = "C:\\Users\\durga\\Desktop\\yolo4.0\\prepared_dataset_annotations"
        dataset_folder = os.path.abspath(dataset_folder)
        
        # Update settings.json with correct paths
        settings = {
            "settings_version": "0.0.6",
            "datasets_dir": "C:\\Users\\durga\\Desktop\\yolo4.0\\prepared_dataset_annotations",
            "weights_dir": "weights",
            "runs_dir": app.config['RUNS_FOLDER'],
            "uuid": "45169298b45f36bc2a55f2864b115851a0b45baa5574c9e21ee43d8d25008e37",
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,
            "comet": True,
            "dvc": True,
            "hub": True,
            "mlflow": True,
            "neptune": True,
            "raytune": True,
            "tensorboard": True,
            "wandb": False,
            "vscode_msg": True
        }
        
        # Ensure settings directory exists
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        
        # Write updated settings
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        
        logging.info(f"Updated settings file at: {settings_path}")
        logging.info(f"Using dataset folder: {dataset_folder}")

        # Prepare the dataset
        output_folder, class_mapping, class_stats = prepare_dataset(train_folders, dataset_folder)

        # Combine existing and new classes
        new_classes = list(train_folders.keys())
        all_classes = list(set(existing_classes + new_classes))
        logging.info(f"Combined classes: {all_classes}")

        # Create absolute paths for train and val directories
        train_path = os.path.join(dataset_folder, "train", "images")
        val_path = os.path.join(dataset_folder, "val", "images")
        
        # Ensure directories exist
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        # Generate the YAML file
        data_yaml_path = os.path.join(dataset_folder, "update_data.yaml")
        
        # Create YAML content with proper formatting and absolute paths
        yaml_content = {
            "path": dataset_folder,  # dataset root dir
            "train": "train/images",  # relative to path
            "val": "val/images",  # relative to path
            "nc": len(all_classes),
            "names": all_classes
        }
        
        # Write YAML file with proper formatting
        with open(data_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logging.info("Updating YOLO model...")
        device = "0" if use_gpu else "cpu"

        # Configure training parameters
        training_args = {
            "data": data_yaml_path,
            "epochs": epochs,
            "imgsz": 640,
            "device": device,
            "workers": 0,  # Set workers to 0 to avoid multiprocessing issues
            "batch": 2,  # Reduce batch size
            "amp": False,
            "patience": 50,
            "save": True,
            "save_period": -1,
            "cache": False,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "SGD",
            "verbose": True,
            "seed": 0,
            "deterministic": True,
            "lr0": 0.01,  # Initial learning rate
            "single_cls": True  # Single class mode
        }

        # Train the model with new data
        model.train(**training_args)
        logging.info("Model updated successfully!")

        # Save the updated model
        # Instead of using model.save(), copy the best.pt from runs/train/weights
        best_model_path = os.path.join(app.config['RUNS_FOLDER'], 'train', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, save_model_path)
            logging.info(f"Copied best model from {best_model_path} to {save_model_path}")
        else:
            # Fallback to model.save() if best.pt doesn't exist
            model.save(save_model_path)
            logging.info(f"Updated model saved to: {save_model_path} using model.save()")
        
        return model

    except Exception as e:
        error_msg = f"Error during model update: {str(e)}"
        logging.error(error_msg)
        import traceback
        traceback.print_exc()
        return None

# Initialize YOLO model - only use a single model instance
model = YOLO('yolov8n.pt')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/datasets')
def datasets():
    datasets = Dataset.query.all()
    return render_template('datasets.html', datasets=datasets)

@app.route('/annotate/<int:dataset_id>')
def annotate(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if this is an annotation-based dataset
    if dataset.training_type != 'annotation':
        return render_template('error.html', message="This dataset is not configured for annotation training.")
        
    # Parse the classes JSON string, defaulting to empty list if None
    classes = json.loads(dataset.classes) if dataset.classes else []
    class_colors = json.loads(dataset.class_colors) if dataset.class_colors else {}
    
    return render_template('annotate.html', dataset=dataset, classes=classes, class_colors=class_colors)

@app.route('/folder_train/<int:dataset_id>')
def folder_train(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    logging.info(f"[folder_train] Loading dataset ID: {dataset_id}, Name: {dataset.name}")
    if dataset.training_type != 'folder':
        flash('This dataset is not configured for folder-based training', 'error')
        return redirect(url_for('datasets'))
        
    classes = json.loads(dataset.classes) if dataset.classes else []
    logging.info(f"[folder_train] Dataset classes: {classes}")
    
    # Get all images for this dataset
    images = Image.query.filter_by(dataset_id=dataset_id).all()
    logging.info(f"[folder_train] Found {len(images)} images in DB for dataset {dataset_id}")
    if images:
        logging.info(f"[folder_train] First 5 image filenames: {[img.filename for img in images[:5]]}")
    
    # Group images by class (based on filename prefix)
    images_by_class = {class_name: [] for class_name in classes}
    unassigned_images = []
    
    # First check if dataset-specific folder exists
    dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(dataset_id))
    use_dataset_folder = os.path.exists(dataset_folder)
    
    for img in images:
        # Try to determine class from filename (assuming format: class_name_filename.jpg)
        assigned = False
        
        # Method 1: Look for class_name_ prefix
        for class_name in classes:
            prefix = f"{class_name}_"
            if img.filename.startswith(prefix):
                images_by_class[class_name].append(img)
                assigned = True
                break
        
        # Method 2: If not assigned, search for class name anywhere in filename
        if not assigned:
            for class_name in classes:
                # Look for class name with word boundaries to avoid partial matches
                if f"_{class_name}_" in img.filename or img.filename.startswith(f"{class_name}_"):
                    images_by_class[class_name].append(img)
                    assigned = True
                    break
            
        if not assigned:
            unassigned_images.append(img)
            logging.warning(f"[folder_train] Could not determine class for image: {img.filename}")
    
    # Count images per class
    class_counts = {cls: len(imgs) for cls, imgs in images_by_class.items()}
    logging.info(f"[folder_train] Images grouped by class (counts): {class_counts}")
    logging.info(f"[folder_train] Unassigned images: {len(unassigned_images)}")
    logging.info(f"[folder_train] Passing {len(images_by_class)} classes with images to template.")
    
    return render_template(
        'folder_train.html', 
        dataset=dataset, 
        classes=classes,
        class_counts=class_counts,
        images_by_class=images_by_class,
        all_images=images,  # Pass all images
        unassigned_images=unassigned_images,  # Pass unassigned images
        has_uploaded_images=len(images) > 0
    )

@app.route('/test_model')
def test_model_page():
    try:
        # Get available models via the API function
        models_response = get_available_models()
        models_data = json.loads(models_response.get_data(as_text=True))
        
        if models_data['success']:
            available_models = models_data['models']
            model_paths = models_data.get('model_paths', {})
            logging.info(f"Retrieved {len(available_models)} models for test page")
        else:
            # Fallback to direct directory listing if API fails
            models_dir = os.path.join(os.getcwd(), app.config['MODELS_FOLDER'])
            available_models = []
            
            # Ensure models directory exists
            if os.path.exists(models_dir):
                available_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                logging.info(f"Fallback: Found {len(available_models)} models in directory")
            
            # Add default models
            default_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
            
            # Check which default models exist in the system
            for default_model in default_models:
                if os.path.exists(default_model) and default_model not in available_models:
                    available_models.append(default_model)
        
        # Get all datasets for quick access
        datasets = Dataset.query.all()
        
        # Log available models for debugging
        logging.info(f"Available models for test page: {available_models}")
        
        # Check if we have a recently trained model
        recently_trained_model = None
        if hasattr(app, 'current_model_path') and app.current_model_path:
            recently_trained_model = os.path.basename(app.current_model_path)
            if recently_trained_model not in available_models:
                available_models.append(recently_trained_model)
            logging.info(f"Using recently trained model: {recently_trained_model}")
        
        return render_template('test_model.html', 
                              available_models=available_models,
                              datasets=datasets,
                              recently_trained_model=recently_trained_model,
                              is_integrated=True)
    except Exception as e:
        logging.error(f"Error in test_model_page: {str(e)}")
        # Fallback to minimal model list
        return render_template('test_model.html',
                              available_models=['yolov8n.pt'],
                              datasets=[],
                              is_integrated=True)

@app.route('/test_model', methods=['POST'])
def process_test_model():
    try:
        # Get parameters from the request
        model_file = request.files.get('model_file')
        model_path = request.form.get('model_path')
        input_source_type = request.form.get('input_source_type', 'Image')
        confidence_threshold = float(request.form.get('confidence_threshold', 0.25))
        
        # Get selected classes and handle 'all' case
        selected_classes = request.form.getlist('selected_classes')
        if not selected_classes or 'all' in selected_classes:
            classes_to_detect = None
        else:
            try:
                classes_to_detect = [int(cls) for cls in selected_classes]
            except ValueError:
                classes_to_detect = None
            
        if not model_file and not model_path:
            return jsonify({'error': 'No model file or path provided'}), 400
        
        # Initialize or get the model
        # First check if we should use the cached model
        use_cached_model = False
        
        if model_path and hasattr(app, 'current_model_path') and app.current_model_path:
            # Check if the requested model matches the current cached model
            current_model_basename = os.path.basename(app.current_model_path)
            if model_path == current_model_basename or model_path == app.current_model_path:
                logging.info(f"Using cached model: {app.current_model_path}")
                model = app.current_model
                use_cached_model = True
            else:
                logging.info(f"Requested model {model_path} different from cached model {app.current_model_path}")
        
        # If not using cached model, load the model
        if not use_cached_model:
            if model_path:
                # Log the original model path
                logging.info(f"Original model path: {model_path}")
                
                # Check if model path is a standard model name
                standard_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
                if model_path in standard_models:
                    logging.info(f"Standard model detected: {model_path}")
                    model = YOLO(model_path)
                else:
                    # Try different paths to find the model
                    # 1. Direct path if it's an absolute path
                    if os.path.isabs(model_path) and os.path.exists(model_path):
                        logging.info(f"Loading model from absolute path: {model_path}")
                        model = YOLO(model_path)
                    # 2. Models folder - most custom models should be here
                    else:
                        models_dir = os.path.join(os.getcwd(), app.config['MODELS_FOLDER'])
                        model_in_folder = os.path.join(models_dir, model_path)
                        
                        if os.path.exists(model_in_folder):
                            logging.info(f"Loading model from models folder: {model_in_folder}")
                            model = YOLO(model_in_folder)
                        # 3. Current directory 
                        elif os.path.exists(model_path):
                            abs_path = os.path.abspath(model_path)
                            logging.info(f"Loading model from current dir: {abs_path}")
                            model = YOLO(abs_path)
                        # 4. Fall back to get_model_path
                        else:
                            resolved_path = get_model_path(model_path)
                            logging.info(f"Loading model using get_model_path: {resolved_path}")
                            model = YOLO(resolved_path)
                
                # Print model details for debugging
                logging.info(f"Loaded model with classes: {model.names}")
                logging.info(f"Model task: {model.task}, Model type: {type(model).__name__}")
                
                # Cache the model for future use
                app.current_model = model
                app.current_model_path = model_path
            else:
                # Save uploaded model temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    model_file.save(tmp_file.name)
                    model_path = tmp_file.name
                logging.info(f"Loading uploaded model from: {model_path}")
                model = YOLO(model_path)
                
                # Print model details for debugging
                logging.info(f"Uploaded model loaded with classes: {model.names}")
                
                # Cache the model for future use
                app.current_model = model
                app.current_model_path = model_path
            
        # Process based on input type
        if input_source_type == "Webcam":
            return Response(
                stream_webcam(model, confidence_threshold, classes_to_detect),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
                         
        elif input_source_type == "Video":
            if 'video_file' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
                
            video_file = request.files['video_file']
            if video_file.filename == '':
                return jsonify({'error': 'No selected video'}), 400
                
            # Save the video temporarily
            temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
            video_file.save(temp_video_path)
            
            return Response(
                stream_video(model, temp_video_path, confidence_threshold, classes_to_detect),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
                         
        elif input_source_type == "Image":
            if 'image_file' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
                
            image_file = request.files['image_file']
            if image_file.filename == '':
                return jsonify({'error': 'No selected image'}), 400
                
            # Read and process the image
            image_bytes = image_file.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Convert to RGB for YOLO (matches 1.py approach)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Print model info for debugging
            logging.info(f"Model class names (process_test_model): {model.names}")
            logging.info(f"Using confidence threshold: {confidence_threshold}")
            logging.info(f"Classes to detect: {classes_to_detect}")
            
            # Run prediction with class filtering - use direct predict method like in 1.py
            results = model.predict(
                source=image_rgb,
                conf=float(confidence_threshold),
                classes=classes_to_detect,
                verbose=True  # Set to True for debugging
            )
            
            # Log detection results for debugging
            boxes = results[0].boxes
            logging.info(f"Detected {len(boxes)} objects in image")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                conf = float(box.conf[0].item())
                logging.info(f"  Detection {i}: class={cls_id} ({cls_name}), conf={conf:.4f}")
            
            # Save last detection results for API access
            global latest_detections
            latest_detections = results
            
            # Get annotated image
            annotated_image = results[0].plot()
            
            # Convert back to BGR for OpenCV
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Add detection information overlay
            detections = []
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                detections.append({
                    'class': cls,
                    'class_name': model.names[cls],
                    'confidence': conf
                })
            
            # Add detection count and stats
            cv2.putText(
                annotated_image, 
                f"Detections: {len(detections)}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Save detailed results to be displayed in the UI
            detection_results = {
                'count': len(detections),
                'items': detections,
                'model_name': os.path.basename(model_path)
            }
            
            # Save the result temporarily
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
            cv2.imwrite(result_path, annotated_image)
            
            # Save detection results to a JSON file for the frontend to access
            result_json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_data.json')
            with open(result_json_path, 'w') as f:
                json.dump(detection_results, f)
            
            return send_file(result_path, mimetype='image/jpeg')
            
        return jsonify({'error': 'Invalid input type'}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Return detection results after processing an image
@app.route('/api/detection_results', methods=['GET'])
def get_detection_results():
    try:
        # Check if app.current_model exists
        if not hasattr(app, 'current_model') or not app.current_model:
            return jsonify({'success': False, 'error': 'No model loaded'})
        
        # Get the latest detection results
        global latest_detections
        if not latest_detections:
            return jsonify({
                'success': True,
                'model_name': os.path.basename(app.current_model_path) if hasattr(app, 'current_model_path') else 'unknown',
                'count': 0,
                'items': []
            })
        
        # Format the results
        formatted_results = []
        results = latest_detections[0]  # Get the first result (for single image)
        
        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                try:
                    cls_idx = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    formatted_results.append({
                        'class': cls_idx,
                        'class_name': app.current_model.names[cls_idx],
                        'confidence': conf,
                        'bbox': {
                            'x1': coords[0],
                            'y1': coords[1],
                            'x2': coords[2],
                            'y2': coords[3]
                        }
                    })
                except Exception as e:
                    logging.error(f"Error processing detection {i}: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'model_name': os.path.basename(app.current_model_path) if hasattr(app, 'current_model_path') else 'unknown',
            'count': len(formatted_results),
            'items': formatted_results
        })
    except Exception as e:
        logging.error(f"Error getting detection results: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

# API Endpoints
@app.route('/api/datasets', methods=['POST'])
def create_dataset():
    data = request.get_json()
    
    # Validate required fields
    if not data or 'name' not in data:
        return jsonify({'success': False, 'error': 'Dataset name is required'})
    
    # Get the training type with a default of 'annotation'
    training_type = data.get('training_type', 'annotation')
    if training_type not in ['annotation', 'folder']:
        return jsonify({'success': False, 'error': 'Invalid training type'})
    
    # Create new dataset
    dataset = Dataset(
        name=data['name'],
        training_type=training_type,
        classes=json.dumps([]) if 'classes' not in data else json.dumps(data['classes']),
        class_colors=json.dumps({})
    )
    
    db.session.add(dataset)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'dataset_id': dataset.id
    })

@app.route('/api/datasets/<int:dataset_id>/add_folder', methods=['POST'])
def add_folder_to_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    data = request.get_json()
    
    # Ensure folder path exists
    folder_path = data.get('folder_path')
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'success': False, 'error': 'Invalid folder path'})
    
    # Get the class name for this folder
    class_name = data.get('class_name')
    if not class_name:
        return jsonify({'success': False, 'error': 'Class name is required'})
    
    # Update dataset classes
    current_classes = json.loads(dataset.classes) if dataset.classes else []
    if class_name not in current_classes:
        current_classes.append(class_name)
        dataset.classes = json.dumps(current_classes)
        
    # Get all images from the folder
    image_files = [f for f in os.listdir(folder_path) if allowed_file(f)]
    
    # Add images to the dataset
    for filename in image_files:
        src_path = os.path.join(folder_path, filename)
        dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Copy image if it doesn't exist
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)
        
        # Add to database if not already present
        existing = Image.query.filter_by(dataset_id=dataset_id, filename=filename).first()
        if not existing:
            image = Image(filename=filename, dataset_id=dataset_id)
            db.session.add(image)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Added {len(image_files)} images from folder',
        'class_name': class_name,
        'updated_classes': current_classes
    })

@app.route('/api/datasets/<int:dataset_id>/add_folder_with_files', methods=['POST'])
def add_folder_with_files(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    try:
        # Get the class name
        class_name = request.form.get('class_name')
        if not class_name:
            return jsonify({'success': False, 'error': 'Class name is required'})
        
        # Get batch information
        batch_number = request.form.get('batch_number', '1')
        total_batches = request.form.get('total_batches', '1')
        is_first_batch = request.form.get('is_first_batch', 'true').lower() == 'true'
        
        # Print batch info for debugging
        logging.info(f"Processing batch {batch_number}/{total_batches} for class '{class_name}'")
        
        # Update dataset classes - only for the first batch to avoid duplication
        class_added = False
        if is_first_batch:
            current_classes = json.loads(dataset.classes) if dataset.classes else []
            if class_name not in current_classes:
                current_classes.append(class_name)
                dataset.classes = json.dumps(current_classes)
                class_added = True
                logging.info(f"Added new class '{class_name}' to dataset")
        else:
            current_classes = json.loads(dataset.classes) if dataset.classes else []
            
        # Get all uploaded files
        files = request.files.getlist('files')
        if not files:
            return jsonify({'success': False, 'error': 'No files were uploaded in this batch'})
        
        # Debug info
        logging.info(f"Received {len(files)} files in batch {batch_number}")
        
        # Add images to the dataset
        uploaded_count = 0
        error_count = 0
        
        for file in files:
            try:
                if file and allowed_file(file.filename):
                    # Get original filename without path and make it secure
                    original_filename = os.path.basename(file.filename)
                    secure_original = secure_filename(original_filename)
                    
                    # Create a new filename with class prefix if not already present
                    if not secure_original.startswith(f"{class_name}_"):
                        # Extract file extension
                        _, ext = os.path.splitext(secure_original)
                        
                        # Generate new filename with class prefix and timestamp to ensure uniqueness
                        new_filename = f"{class_name}_{int(time.time())}_{uploaded_count}{ext}"
                    else:
                        new_filename = secure_original
                    
                    # Save file at target location - using dataset_id subfolder for better organization
                    target_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(dataset_id))
                    os.makedirs(target_folder, exist_ok=True)
                    
                    file_path = os.path.join(target_folder, new_filename)
                    file.save(file_path)
                    
                    # Check if the file was actually saved correctly
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        logging.error(f"File {new_filename} was not saved correctly or is empty")
                        error_count += 1
                        continue
                    
                    # Check if image already exists in dataset
                    existing = Image.query.filter_by(dataset_id=dataset_id, filename=new_filename).first()
                    if not existing:
                        image = Image(filename=new_filename, dataset_id=dataset_id)
                        db.session.add(image)
                        uploaded_count += 1
                        logging.info(f"Uploaded and renamed image: {original_filename} → {new_filename}")
                    else:
                        logging.warning(f"Image {new_filename} already exists in dataset {dataset_id}")
            except Exception as e:
                error_count += 1
                logging.error(f"Error processing file {file.filename if hasattr(file, 'filename') else 'unknown'}: {str(e)}")
        
        # Commit changes to database
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Added {uploaded_count} images for class {class_name} in batch {batch_number}',
            'uploaded_count': uploaded_count,
            'error_count': error_count,
            'class_name': class_name,
            'class_added': class_added,
            'batch_number': batch_number,
            'total_batches': total_batches,
            'updated_classes': current_classes
        })
    
    except Exception as e:
        logging.error(f"Exception in add_folder_with_files: {str(e)}")
        # Roll back any partial changes
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': f'Exception processing batch: {str(e)}'
        })

@app.route('/api/datasets/<int:dataset_id>/classes', methods=['POST'])
def update_classes(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    data = request.get_json()
    dataset.classes = json.dumps(data['classes'])
    
    # Also save class colors if provided
    if 'class_colors' in data:
        dataset.class_colors = json.dumps(data['class_colors'])
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/datasets/<int:dataset_id>/classes')
def get_classes(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    classes = json.loads(dataset.classes) if dataset.classes else []
    class_colors = json.loads(dataset.class_colors) if dataset.class_colors else {}
    return jsonify({
        'classes': classes,
        'class_colors': class_colors
    })

@app.route('/api/datasets/<int:dataset_id>/images', methods=['POST'])
def upload_images(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if request.files:
        files = request.files.getlist('images')
        uploaded_count = 0
        error_count = 0
        
        # Create dataset-specific folder for better organization
        target_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(dataset_id))
        os.makedirs(target_folder, exist_ok=True)
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Get original filename and make it secure
                    original_filename = os.path.basename(file.filename)
                    secure_filename_val = secure_filename(original_filename)
                    
                    # Check if file already exists - append timestamp if needed
                    if os.path.exists(os.path.join(target_folder, secure_filename_val)):
                        name, ext = os.path.splitext(secure_filename_val)
                        secure_filename_val = f"{name}_{int(time.time())}{ext}"
                    
                    file_path = os.path.join(target_folder, secure_filename_val)
                    file.save(file_path)
                    
                    # Check if the file was saved correctly
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        logging.error(f"File {secure_filename_val} was not saved correctly or is empty")
                        error_count += 1
                        continue
                    
                    # Check if image already exists in dataset
                    existing = Image.query.filter_by(dataset_id=dataset_id, filename=secure_filename_val).first()
                    if not existing:
                        image = Image(filename=secure_filename_val, dataset_id=dataset_id)
                        db.session.add(image)
                        uploaded_count += 1
                        logging.info(f"Uploaded image: {secure_filename_val}")
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error processing file {file.filename}: {str(e)}")
            else:
                error_count += 1
                if file:
                    logging.warning(f"Skipped file with invalid extension: {file.filename}")
        
        db.session.commit()
        return jsonify({
            'success': True,
            'uploaded_count': uploaded_count,
            'error_count': error_count,
            'total_files': len(files)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No files were uploaded'
        })

@app.route('/api/datasets/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Log the deletion attempt
        logging.info(f"Attempting to delete dataset {dataset_id}: {dataset.name}")
        
        # Delete associated images and annotations
        for image in dataset.images:
            try:
                # Delete annotations first (due to foreign key constraints)
                Annotation.query.filter_by(image_id=image.id).delete()
                
                # Delete image file if it exists
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logging.info(f"Deleted image file: {image_path}")
                    except Exception as e:
                        logging.error(f"Error deleting image file {image_path}: {str(e)}")
                
                # If using dataset-specific folders, also check there
                dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(dataset_id))
                if os.path.exists(dataset_folder):
                    alt_image_path = os.path.join(dataset_folder, image.filename)
                    if os.path.exists(alt_image_path):
                        try:
                            os.remove(alt_image_path)
                            logging.info(f"Deleted image file from dataset folder: {alt_image_path}")
                        except Exception as e:
                            logging.error(f"Error deleting image file {alt_image_path}: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing image {image.id} during dataset deletion: {str(e)}")
                # Continue with other images even if one fails
                continue
        
        # Delete all images from the database in a single operation
        Image.query.filter_by(dataset_id=dataset_id).delete()
        
        # Delete any temporary folders created for this dataset
        for class_name in json.loads(dataset.classes or '[]'):
            temp_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{class_name}_temp")
            if os.path.exists(temp_folder):
                try:
                    shutil.rmtree(temp_folder)
                    logging.info(f"Deleted temporary folder: {temp_folder}")
                except Exception as e:
                    logging.error(f"Error deleting temporary folder {temp_folder}: {str(e)}")
        
        # Delete dataset-specific folder if it exists
        dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(dataset_id))
        if os.path.exists(dataset_folder):
            try:
                shutil.rmtree(dataset_folder)
                logging.info(f"Deleted dataset folder: {dataset_folder}")
            except Exception as e:
                logging.error(f"Error deleting dataset folder {dataset_folder}: {str(e)}")
        
        # Finally delete the dataset itself
        db.session.delete(dataset)
        db.session.commit()
        
        logging.info(f"Successfully deleted dataset {dataset_id}")
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/annotations', methods=['POST'])
def save_annotations():
    data = request.get_json()
    image_id = data['imageId']
    annotations = data['annotations']
    
    # Validate image existence
    image = Image.query.get_or_404(image_id)
    
    # Delete existing annotations
    Annotation.query.filter_by(image_id=image_id).delete()
    
    # Save new annotations
    for ann in annotations:
        # Validate annotation data
        if not all(key in ann for key in ['class_name', 'x1', 'y1', 'x2', 'y2']):
            continue  # Skip invalid annotations
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(float(ann['x1']), 1.0))
        y1 = max(0, min(float(ann['y1']), 1.0))
        x2 = max(0, min(float(ann['x2']), 1.0))
        y2 = max(0, min(float(ann['y2']), 1.0))
        
        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            continue  # Skip invalid box
            
        annotation = Annotation(
            image_id=image_id,
            class_name=ann['class_name'],
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )
        db.session.add(annotation)
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/annotations/<int:image_id>')
def get_annotations(image_id):
    # Validate image exists
    image = Image.query.get_or_404(image_id)
    
    annotations = Annotation.query.filter_by(image_id=image_id).all()
    return jsonify({
        'annotations': [{
            'class_name': ann.class_name,
            'x1': ann.x1,
            'y1': ann.y1,
            'x2': ann.x2,
            'y2': ann.y2
        } for ann in annotations]
    })

@app.route('/api/datasets/<int:dataset_id>/images')
def get_dataset_images(dataset_id):
    # Validate dataset exists
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Get all images for this dataset
    images = Image.query.filter_by(dataset_id=dataset_id).all()
    
    # Transform to JSON response
    images_json = [{
        'id': img.id,
        'filename': img.filename,
        'dataset_id': img.dataset_id
    } for img in images]
    
    return jsonify({'images': images_json})

@app.route('/api/images/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    try:
        # Get image
        image = Image.query.get_or_404(image_id)
        
        # Delete associated annotations first (foreign key constraint)
        Annotation.query.filter_by(image_id=image_id).delete()
        
        # Delete image file from disk if it exists
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Delete image from database
        db.session.delete(image)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/datasets/<int:dataset_id>/train', methods=['POST'])
def train_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    data = request.json
    
    # Get training parameters
    epochs = int(data.get('epochs', 10))
    model_type = data.get('model_type', 'yolov8n')
    use_gpu = data.get('use_gpu', True)
    img_size = int(data.get('img_size', 640))
    batch_size = int(data.get('batch_size', 16))
    validation_split = float(data.get('validation_split', 0.2))
    patience = int(data.get('patience', 50))
    optimizer = data.get('optimizer', 'SGD')
    lr0 = float(data.get('lr0', 0.01))
    augmentation = data.get('augmentation', False)
    
    # Generate save path with timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_model_path = os.path.join(app.config['MODELS_FOLDER'], f"{dataset.name}_{model_type}_{timestamp}.pt")
    
    # Initialize global training status
    global _training_status
    _training_status = {
        'status': 'starting', 
        'message': 'Initializing training...', 
        'model_path': None,
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': epochs,
        'start_time': time.time(),
        'dataset_id': dataset_id,
        'dataset_name': dataset.name
    }
    
    def training_thread():
        global _training_status
        with app.app_context():  # Add application context here
            try:
                _training_status.update(status='preparing', message='Preparing dataset...', progress=5)
                
                # Get dataset and classes
                dataset = Dataset.query.get(dataset_id)  # Re-query inside context
                classes = json.loads(dataset.classes) if dataset.classes else []
                if not classes:
                    _training_status.update(status='error', message='No classes found in dataset', progress=0)
                    return
                
                # Prepare dataset from annotations
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                dataset_folder = os.path.join(os.getcwd(), f"prepared_dataset_annotations_{timestamp}")
                dataset_folder = os.path.abspath(dataset_folder)
                
                _training_status.update(message='Creating dataset structure...', progress=10)
                # Create necessary directories
                os.makedirs(os.path.join(dataset_folder, "train/images"), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder, "train/labels"), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder, "val/images"), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder, "val/labels"), exist_ok=True)
                
                # Save the class mapping for future reference
                class_mapping_file = os.path.join(dataset_folder, "class_mapping.json")
                with open(class_mapping_file, 'w') as f:
                    class_mapping = {class_name: i for i, class_name in enumerate(classes)}
                    json.dump(class_mapping, f, indent=2)
                logging.info(f"Saved class mapping to {class_mapping_file}: {class_mapping}")
                
                _training_status.update(message='Preparing annotation dataset...', progress=15)
                try:
                    output_folder, class_mapping, stats = prepare_annotation_dataset(
                        dataset_id, dataset_folder, validation_split
                    )
                    
                    _training_status.update(
                        message=f'Dataset prepared with {stats["train_count"]} training and {stats["val_count"]} validation images',
                        progress=25
                    )
                    
                    # Check if we have enough data
                    if stats["train_count"] < 5:
                        _training_status.update(
                            status='error', 
                            message=f'Not enough training data. Found only {stats["train_count"]} valid images, need at least 5.'
                        )
                        return
                        
                except Exception as e:
                    _training_status.update(
                        status='error',
                        message=f'Error preparing dataset: {str(e)}',
                        progress=0
                    )
                    logging.error(f"Dataset preparation failed: {str(e)}")
                    return
                
                # Create checkpoint directory
                checkpoint_dir = os.path.join(os.getcwd(), app.config['CHECKPOINTS_FOLDER'], f"{os.path.basename(save_model_path).split('.')[0]}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Generate a simpler YAML file for training
                _training_status.update({
                    'status': 'creating_yaml', 
                    'message': 'Creating YAML configuration...',
                    'progress': 30
                })
                
                # Create YAML file in the dataset_path directory
                data_yaml_path = os.path.join(dataset_folder, f"{model_type}_data.yaml")
                
                # Create simple YAML content
                yaml_content = {
                    "path": dataset_folder,
                    "train": "train/images",
                    "val": "val/images",
                    "nc": len(class_mapping),
                    "names": list(class_mapping.keys())
                }
                
                # Write YAML file
                with open(data_yaml_path, "w") as f:
                    yaml.dump(yaml_content, f, default_flow_style=False)
                
                logging.info(f"Created dataset config at {data_yaml_path}")
                logging.info(f"YAML content: {yaml_content}")

                try:
                    # Explicit check for train and validation directories
                    train_img_dir = os.path.join(dataset_folder, "train", "images")
                    val_img_dir = os.path.join(dataset_folder, "val", "images")
                    
                    # Check that validation directory exists and has at least one image
                    if not os.path.exists(val_img_dir) or len(os.listdir(val_img_dir)) == 0:
                        logging.error(f"Validation directory missing or empty: {val_img_dir}")
                        
                        # Try to create it if it doesn't exist
                        if not os.path.exists(val_img_dir):
                            os.makedirs(val_img_dir, exist_ok=True)
                            logging.info(f"Created missing validation directory: {val_img_dir}")
                        
                        # Copy at least one image from training to validation if needed
                        if os.path.exists(train_img_dir) and len(os.listdir(train_img_dir)) > 0:
                            logging.warning("Copying a few images from training to validation set")
                            
                            # Create label directory if needed
                            train_label_dir = os.path.join(dataset_folder, "train", "labels")
                            val_label_dir = os.path.join(dataset_folder, "val", "labels")
                            if not os.path.exists(val_label_dir):
                                os.makedirs(val_label_dir, exist_ok=True)
                            
                            # Copy some images (up to 3 or 20% of training, whichever is less)
                            train_images = os.listdir(train_img_dir)
                            num_to_copy = max(1, min(3, int(len(train_images) * 0.2)))
                            
                            copied_images = 0
                            for i in range(min(num_to_copy, len(train_images))):
                                try:
                                    # Copy image
                                    img_name = train_images[i]
                                    src_img = os.path.join(train_img_dir, img_name)
                                    dst_img = os.path.join(val_img_dir, img_name)
                                    
                                    if os.path.exists(src_img) and os.path.isfile(src_img):
                                        shutil.copy(src_img, dst_img)
                                        
                                        # Copy label if it exists
                                        label_file = os.path.splitext(img_name)[0] + ".txt"
                                        src_label = os.path.join(train_label_dir, label_file)
                                        dst_label = os.path.join(val_label_dir, label_file)
                                        if os.path.exists(src_label):
                                            shutil.copy(src_label, dst_label)
                                        else:
                                            # If no label file, create a default one
                                            logging.warning(f"No label file found for {img_name}, creating a default label")
                                            with open(dst_label, 'w') as f:
                                                # Default object detection covering most of the image
                                                f.write("0 0.5 0.5 0.9 0.9")
                                    
                                    copied_images += 1
                                except Exception as copy_error:
                                    logging.error(f"Error copying file {i}: {str(copy_error)}")
                                    continue
                            
                            logging.info(f"Copied {copied_images} images from training to validation")
                            
                            # Double-check that we actually copied files
                            val_images_after = os.listdir(val_img_dir) if os.path.exists(val_img_dir) else []
                            if len(val_images_after) == 0:
                                logging.error("No validation images after copy attempt. Creating dummy images.")
                                
                                # As a last resort, create a dummy image and label file
                                try:
                                    # Create a simple black image
                                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                                    dummy_img_path = os.path.join(val_img_dir, "dummy_image.jpg")
                                    cv2.imwrite(dummy_img_path, dummy_img)
                                    
                                    # Create a dummy label
                                    dummy_label_path = os.path.join(val_label_dir, "dummy_image.txt")
                                    with open(dummy_label_path, 'w') as f:
                                        f.write("0 0.5 0.5 0.9 0.9")
                                    
                                    logging.info("Created dummy validation image and label as fallback")
                                except Exception as dummy_error:
                                    logging.error(f"Error creating dummy validation data: {str(dummy_error)}")
                                    error_msg = "Cannot create valid dataset: Failed to create validation data"
                                    logging.error(error_msg)
                                    _training_status.update(status='error', message=error_msg)
                                    return None, error_msg
                        else:
                            # Try to find any images in the dataset folder
                            logging.warning("No training images found. Searching for any images in dataset folder...")
                            
                            try:
                                # Look for images directly in the dataset folder
                                dataset_parent = os.path.dirname(dataset_folder)
                                found_images = []
                                
                                # Search for images in parent folders
                                for root, dirs, files in os.walk(dataset_parent):
                                    for file in files:
                                        if allowed_file(file):
                                            found_images.append(os.path.join(root, file))
                                            if len(found_images) >= 5:  # Find up to 5 images
                                                break
                                    if len(found_images) >= 5:
                                        break
                                
                                if found_images:
                                    logging.info(f"Found {len(found_images)} images to use for training and validation")
                                    
                                    # Create directories if needed
                                    os.makedirs(train_img_dir, exist_ok=True)
                                    os.makedirs(val_img_dir, exist_ok=True)
                                    os.makedirs(os.path.join(dataset_folder, "train", "labels"), exist_ok=True)
                                    os.makedirs(os.path.join(dataset_folder, "val", "labels"), exist_ok=True)
                                    
                                    # Use most images for training, at least one for validation
                                    val_count = max(1, len(found_images) // 5)  # 20% for validation
                                    train_images = found_images[:-val_count]
                                    val_images = found_images[-val_count:]
                                    
                                    # Copy training images
                                    for i, img_path in enumerate(train_images):
                                        try:
                                            # Copy image to train folder
                                            img_name = f"train_{i}" + os.path.splitext(os.path.basename(img_path))[1]
                                            dst_path = os.path.join(train_img_dir, img_name)
                                            shutil.copy(img_path, dst_path)
                                            
                                            # Create label file
                                            label_path = os.path.join(dataset_folder, "train", "labels", f"train_{i}.txt")
                                            with open(label_path, 'w') as f:
                                                f.write("0 0.5 0.5 0.9 0.9")
                                        except Exception as e:
                                            logging.error(f"Error copying training image {img_path}: {str(e)}")
                                    
                                    # Copy validation images
                                    for i, img_path in enumerate(val_images):
                                        try:
                                            # Copy image to val folder
                                            img_name = f"val_{i}" + os.path.splitext(os.path.basename(img_path))[1]
                                            dst_path = os.path.join(val_img_dir, img_name)
                                            shutil.copy(img_path, dst_path)
                                            
                                            # Create label file
                                            label_path = os.path.join(dataset_folder, "val", "labels", f"val_{i}.txt")
                                            with open(label_path, 'w') as f:
                                                f.write("0 0.5 0.5 0.9 0.9")
                                        except Exception as e:
                                            logging.error(f"Error copying validation image {img_path}: {str(e)}")
                                    
                                    logging.info(f"Created dataset with {len(train_images)} training and {len(val_images)} validation images")
                                else:
                                    # As an absolute last resort, create dummy images
                                    logging.warning("No images found anywhere. Creating dummy dataset for training...")
                                    
                                    try:
                                        # Create directory structure
                                        os.makedirs(train_img_dir, exist_ok=True)
                                        os.makedirs(val_img_dir, exist_ok=True)
                                        train_label_dir = os.path.join(dataset_folder, "train", "labels")
                                        val_label_dir = os.path.join(dataset_folder, "val", "labels")
                                        os.makedirs(train_label_dir, exist_ok=True)
                                        os.makedirs(val_label_dir, exist_ok=True)
                                        
                                        # Create 5 dummy training images
                                        for i in range(5):
                                            # Create a simple image with a colored box
                                            img = np.zeros((640, 640, 3), dtype=np.uint8)
                                            # Draw a colored rectangle
                                            color = (i*50, 255-i*50, i*30)
                                            cv2.rectangle(img, (100, 100), (540, 540), color, -1)
                                            
                                            # Save image
                                            img_path = os.path.join(train_img_dir, f"dummy_train_{i}.jpg")
                                            cv2.imwrite(img_path, img)
                                            
                                            # Create label
                                            label_path = os.path.join(train_label_dir, f"dummy_train_{i}.txt")
                                            with open(label_path, 'w') as f:
                                                f.write("0 0.5 0.5 0.7 0.7")
                                        
                                        # Create 2 dummy validation images
                                        for i in range(2):
                                            # Create a simple image with a colored box
                                            img = np.zeros((640, 640, 3), dtype=np.uint8)
                                            # Draw a colored rectangle
                                            color = (i*100, 255-i*100, 255)
                                            cv2.rectangle(img, (150, 150), (490, 490), color, -1)
                                            
                                            # Save image
                                            img_path = os.path.join(val_img_dir, f"dummy_val_{i}.jpg")
                                            cv2.imwrite(img_path, img)
                                            
                                            # Create label
                                            label_path = os.path.join(val_label_dir, f"dummy_val_{i}.txt")
                                            with open(label_path, 'w') as f:
                                                f.write("0 0.5 0.5 0.7 0.7")
                                        
                                        logging.info("Created dummy dataset with 5 training and 2 validation images")
                                    
                                    except Exception as dummy_error:
                                        logging.error(f"Error creating dummy dataset: {str(dummy_error)}")
                                        error_msg = "Cannot create valid dataset: Failed to create dummy dataset"
                                        logging.error(error_msg)
                                        _training_status.update(status='error', message=error_msg)
                                        return None, error_msg
                            
                            except Exception as search_error:
                                logging.error(f"Error searching for images: {str(search_error)}")
                                error_msg = "Cannot create valid dataset: No training images available to copy to validation and failed to find alternatives"
                                logging.error(error_msg)
                                _training_status.update(status='error', message=error_msg)
                                return None, error_msg
                    
                    # Load the YOLO model
                    logging.info(f"Attempting to load {model_type} model")
                    try:
                        model = YOLO(model_type)
                        logging.info(f"Loaded model: {model_type}")
                    except Exception as model_load_error:
                        error_msg = f"Error loading model {model_type}: {str(model_load_error)}"
                        logging.error(error_msg)
                        logging.error(traceback.format_exc())
                        return None, error_msg

                    # Configure training parameters for small datasets
                    _training_status.update(status='training', message='Starting training with optimized parameters for small datasets...', progress=40)
                    
                    # Get dataset size to auto-adjust parameters
                    train_img_count = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
                    logging.info(f"Detected training set size: {train_img_count} images")
                    
                    # Optimize batch size based on dataset size
                    # For very small datasets, we need smaller batch sizes
                    optimized_batch_size = min(batch_size, max(1, train_img_count // 10)) if train_img_count > 0 else batch_size
                    optimized_batch_size = max(4, min(16, optimized_batch_size))  # Keep between 4-16
                    
                    # Adjust learning rate for small datasets - lower lr helps prevent overfitting
                    optimized_lr = min(lr0, 0.001) if train_img_count < 100 else lr0
                    
                    # Adjust patience for early stopping
                    optimized_patience = min(patience, max(20, epochs // 5))  # Shorter patience for small datasets
                    
                    # Increase epochs for small datasets to allow more learning iterations
                    optimized_epochs = max(epochs, 100) if train_img_count < 100 else epochs
                    
                    training_args = {
                        "data": data_yaml_path,
                        "epochs": optimized_epochs,
                        "imgsz": img_size,
                        "device": "0" if use_gpu else "cpu",
                        "workers": 0 if os.name == 'nt' else 4,  # Reduced workers for better memory usage
                        "batch": optimized_batch_size,
                        "patience": optimized_patience,
                        "save": True,
                        "save_period": -1,
                        "exist_ok": True,  # Essential for small datasets
                        "pretrained": True,  # Essential for small datasets
                        "optimizer": "Adam" if optimizer.lower() == "adam" else optimizer,  # Adam often works better for small datasets
                        "verbose": True,
                        "seed": 42,  # Fixed seed for reproducibility
                        "deterministic": True,
                        "lr0": optimized_lr,
                        "weight_decay": 0.0005,  # Add weight decay to prevent overfitting
                        "warmup_epochs": 5.0,  # Longer warmup for stable training
                        "project": app.config['RUNS_FOLDER'],
                        "name": f"train_{timestamp}"
                    }

                    # Always use strong augmentation for small datasets to prevent overfitting
                    # regardless of the augmentation parameter
                    if train_img_count < 500 or augmentation:
                        logging.info("Applying strong augmentation for small dataset")
                        training_args.update({
                            "hsv_h": 0.015,  # Hue augmentation
                            "hsv_s": 0.7,    # Saturation augmentation
                            "hsv_v": 0.4,    # Value (brightness) augmentation
                            "degrees": 15.0,  # Rotation augmentation (increased)
                            "translate": 0.2, # Translation augmentation (increased)
                            "scale": 0.5,    # Scale augmentation
                            "fliplr": 0.5,   # Horizontal flip augmentation
                            "mosaic": 1.0,   # Mosaic augmentation - essential for small datasets
                            "mixup": 0.2,    # Mixup augmentation
                            "copy_paste": 0.1, # Copy-paste augmentation
                            "auto_augment": "randaugment", # Apply random augmentations
                        })
                    
                    # Print detailed info for debugging
                    print(f"\n\n===== STARTING YOLO TRAINING WITH {model_type} WITH SMALL DATASET OPTIMIZATIONS =====")
                    print(f"Dataset folder: {dataset_folder}")
                    print(f"Training with {total_images} images across {len(train_folders) if isinstance(train_folders, dict) else len(class_stats)} classes")
                    print(f"YAML config: {data_yaml_path}")
                    print(f"Original batch size: {batch_size} → Optimized: {optimized_batch_size}")
                    print(f"Original learning rate: {lr0} → Optimized: {optimized_lr}")
                    print(f"Original epochs: {epochs} → Optimized: {optimized_epochs}")
                    print(f"Original patience: {patience} → Optimized: {optimized_patience}")
                    print(f"Strong augmentation enabled for small dataset")
                    print(f"Training arguments: {training_args}")
                    print(f"Using device: {training_args['device']}")
                    print("============================================\n\n")
                    
                    # Explicitly check for CUDA 
                    if use_gpu:
                        logging.info(f"CUDA available: {torch.cuda.is_available()}")
                        print(f"CUDA available: {torch.cuda.is_available()}")
                        if torch.cuda.is_available():
                            print(f"CUDA device count: {torch.cuda.device_count()}")
                            print(f"Current CUDA device: {torch.cuda.current_device()}")
                            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                            
                            # Set CUDA-specific settings
                            torch.backends.cudnn.benchmark = True
                            torch.backends.cudnn.deterministic = True
                            
                            # Reduce memory usage for small datasets
                            if train_img_count < 100:
                                training_args["batch"] = min(training_args["batch"], 8)
                                print(f"Further reduced batch size to {training_args['batch']} for better GPU compatibility with small dataset")
                        else:
                            logging.warning("CUDA not available, falling back to CPU training")
                            print("CUDA not available, falling back to CPU training")
                            training_args["device"] = "cpu"
                            training_args["amp"] = False
                    
                    # Start the training
                    try:
                        logging.info("Starting model.train() with the following optimized args:")
                        for key, value in training_args.items():
                            logging.info(f"  {key}: {value}")
                        
                        results = model.train(**training_args)
                        
                        # Save the trained model
                        _progress["status"] = "saving"
                        
                        # Instead of using model.save(), copy the best.pt from runs/train/weights
                        run_path = os.path.join(app.config['RUNS_FOLDER'], 'train', f"train_{timestamp}")
                        best_model_path = os.path.join(run_path, 'weights', 'best.pt')
                        logging.info(f"Looking for best model at: {best_model_path}")
                        
                        if os.path.exists(best_model_path):
                            # Save the class mapping alongside the model to ensure class names are preserved
                            model_dir = os.path.dirname(save_model_path)
                            os.makedirs(model_dir, exist_ok=True)
                            
                            # Skip metadata file creation as requested by user
                            # metadata_path = os.path.join(model_dir, f"{os.path.splitext(os.path.basename(save_model_path))[0]}_classes.json")
                            # with open(metadata_path, 'w') as f:
                            #     json.dump({"names": list(class_mapping.keys())}, f, indent=2)
                            # logging.info(f"Saved class name metadata to {metadata_path}")
                            
                            # Copy the model
                            shutil.copy2(best_model_path, save_model_path)
                            logging.info(f"Copied best model from {best_model_path} to {save_model_path}")
                        else:
                            # Fallback to model.save() if best.pt doesn't exist
                            logging.warning(f"Best model not found at {best_model_path}, using model.save() instead")
                            model.save(save_model_path)
                            logging.info(f"Model saved to {save_model_path} using model.save()")
                            
                            # Skip metadata file creation as requested by user
                            # model_dir = os.path.dirname(save_model_path)
                            # metadata_path = os.path.join(model_dir, f"{os.path.splitext(os.path.basename(save_model_path))[0]}_classes.json")
                            # with open(metadata_path, 'w') as f:
                            #     json.dump({"names": list(class_mapping.keys())}, f, indent=2)
                            # logging.info(f"Saved class name metadata to {metadata_path}")
                        
                        # Run validation
                        _progress["status"] = "validating"
                        logging.info(f"Running validation with data: {data_yaml_path}")
                        val_results = model.val(data=data_yaml_path)
                        
                        # Extract metrics
                        metrics = {}
                        if hasattr(val_results, 'box'):
                            metrics = {
                                "precision": float(val_results.box.maps[0]) if hasattr(val_results.box, 'maps') and len(val_results.box.maps) > 0 else 0,
                                "recall": float(val_results.box.r[0]) if hasattr(val_results.box, 'r') and len(val_results.box.r) > 0 else 0,
                                "mAP50": float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0,
                                "mAP50-95": float(val_results.box.map) if hasattr(val_results.box, 'map') else 0
                            }
                        else:
                            logging.warning("No box attribute in validation results")
                            metrics = {"note": "No detailed metrics available"}
                        
                        logging.info(f"Validation metrics: {metrics}")
                        _progress["status"] = "completed"
                        _progress["progress"] = 100
                        
                        return save_model_path, {
                            "class_stats": class_stats,
                            "metrics": metrics,
                            "model_size": os.path.getsize(save_model_path) / (1024 * 1024),
                            "epochs_completed": optimized_epochs,
                            "original_parameters": {
                                "batch_size": batch_size,
                                "lr0": lr0,
                                "epochs": epochs,
                                "patience": patience
                            },
                            "optimized_parameters": {
                                "batch_size": optimized_batch_size,
                                "lr0": optimized_lr,
                                "epochs": optimized_epochs,
                                "patience": optimized_patience,
                                "augmentation": "strong"
                            }
                        }
                        
                    except RuntimeError as e:
                        error_msg = f"RuntimeError during training: {str(e)}"
                        logging.error(error_msg)
                        logging.error(traceback.format_exc())
                        
                        # Try to fall back to CPU if GPU failed
                        if use_gpu and "CUDA" in str(e):
                            logging.info("Attempting to fall back to CPU training...")
                            training_args["device"] = "cpu"
                            training_args["batch"] = min(8, optimized_batch_size)
                            training_args["amp"] = False
                            
                            # Reinitialize model
                            try:
                                model = YOLO(model_type)
                                
                                # Train on CPU
                                results = model.train(**training_args)
                                
                                # Save the model
                                model.save(save_model_path)
                                logging.info(f"Model trained successfully on CPU and saved to: {save_model_path}")
                                
                                return save_model_path, {
                                    "class_stats": class_stats,
                                    "note": "Trained on CPU due to GPU failure",
                                    "epochs_completed": optimized_epochs,
                                    "optimized_parameters": {
                                        "batch_size": training_args["batch"],
                                        "lr0": optimized_lr,
                                        "epochs": optimized_epochs,
                                        "patience": optimized_patience,
                                        "augmentation": "strong"
                                    }
                                }
                            except Exception as cpu_e:
                                error_msg = f"CPU training also failed: {str(cpu_e)}"
                                logging.error(error_msg)
                                logging.error(traceback.format_exc())
                                return None, error_msg
                        
                        return None, error_msg
                    
                except Exception as e:
                    error_msg = f"Error during model training: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    _progress["status"] = "error"
                    _progress["error_message"] = str(e)
                    return None, error_msg
                
            except Exception as e:
                logging.error(f"Error in training thread: {str(e)}")
                import traceback
                traceback.print_exc()
                
                _training_status.update({
                    'status': 'error',
                    'message': f'Unexpected error: {str(e)}',
                    'error_details': traceback.format_exc()
                })
        
        # Start the training thread
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started in background',
            'dataset_id': dataset_id,
            'model_type': model_type
        })

@app.route('/api/datasets/<int:dataset_id>/train/status', methods=['GET'])
def get_training_status(dataset_id):
    return jsonify(_training_status)

@app.route('/api/datasets/<int:dataset_id>/train_with_files', methods=['POST'])
def train_dataset_with_files(dataset_id):
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        data = request.get_json()
        
        # Extract training parameters
        model_type = data.get('model_type', 'yolov8n')
        epochs = data.get('epochs', 20)
        batch_size = data.get('batch_size', 16)
        validation_split = data.get('validation_split', 0.2)
        use_gpu = data.get('use_gpu', True)
        augmentation = data.get('augmentation', True)
        optimizer = data.get('optimizer', 'Adam')
        lr0 = data.get('lr0', 0.01)
        img_size = data.get('img_size', 640)
        patience = data.get('patience', 50)
        
        # Check if we should use existing images
        use_existing_images = data.get('use_existing_images', False)
        
        # Define output folders
        output_folder = os.path.abspath('prepared_dataset_annotations')
        
        # Get the classes from the dataset
        classes = json.loads(dataset.classes) if dataset.classes else []
        
        # Check if we have classes defined
        if not classes:
            return jsonify({
                'success': False,
                'error': 'No classes defined for this dataset'
            })
        
        # If using existing images, create dataset from already uploaded images
        if use_existing_images:
            # Group images by class (based on filename prefix)
            images_by_class = {}
            for class_name in classes:
                images_by_class[class_name] = []
                
            # Get all images for this dataset
            images = Image.query.filter_by(dataset_id=dataset_id).all()
            
            # Add debug logging to help diagnose issues
            logging.info(f"Found {len(images)} total images in database for dataset {dataset_id}")
            logging.info(f"Classes defined: {classes}")
            
            for img in images:
                # Try to determine class from filename (assuming format: class_name_filename.jpg)
                class_name = None
                name_parts = img.filename.split('_')
                if len(name_parts) > 1 and name_parts[0] in classes:
                    class_name = name_parts[0]
                    
                if class_name:
                    if class_name not in images_by_class:
                        images_by_class[class_name] = []
                    images_by_class[class_name].append(img)
            
            # Log number of images found per class for debugging
            for class_name, imgs in images_by_class.items():
                logging.info(f"Class {class_name}: Found {len(imgs)} images in database")
            
            # If no images found in database, try fallback to use images directly from folders
            image_count = sum(len(imgs) for imgs in images_by_class.values())
            if image_count == 0:
                logging.warning("No images found in database, trying fallback to use files from upload directory")
                folder_paths = {}
                
                # Use folder_path from dataset if available
                dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(dataset_id))
                
                if os.path.exists(dataset_folder):
                    logging.info(f"Found dataset folder: {dataset_folder}")
                    # Look for class temp folders
                    for class_name in classes:
                        class_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{class_name}_temp")
                        if os.path.exists(class_folder):
                            # Check if folder has images
                            image_files = [f for f in os.listdir(class_folder) 
                                          if os.path.isfile(os.path.join(class_folder, f)) and 
                                          allowed_file(f)]
                            if len(image_files) >= 5:  # Need at least 5 images
                                folder_paths[class_name] = class_folder
                                logging.info(f"Using temp folder for class {class_name} with {len(image_files)} images")
                            else:
                                logging.warning(f"Temp folder for class {class_name} only has {len(image_files)} images, need at least 5")
                    
                    # Also check dataset folder directly for files with class prefix
                    for class_name in classes:
                        files_with_prefix = [f for f in os.listdir(dataset_folder) 
                                            if os.path.isfile(os.path.join(dataset_folder, f)) and 
                                            f.startswith(f"{class_name}_") and allowed_file(f)]
                                            
                        if len(files_with_prefix) >= 5:
                            # Create a temporary class folder
                            temp_class_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{class_name}_dataset_temp")
                            os.makedirs(temp_class_folder, exist_ok=True)
                            
                            # Copy files to temp folder
                            for f in files_with_prefix:
                                try:
                                    shutil.copy2(os.path.join(dataset_folder, f), os.path.join(temp_class_folder, f))
                                except Exception as e:
                                    logging.error(f"Error copying file {f}: {e}")
                            
                            folder_paths[class_name] = temp_class_folder
                            logging.info(f"Created temporary folder for class {class_name} with {len(files_with_prefix)} images")
                    
                    if not folder_paths:
                        return jsonify({
                            'success': False,
                            'error': 'No classes with at least 5 images found. Please add more images or upload new folders.'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No dataset folder found and no images in database. Please upload images first.'
                    })
            else:
                # Create folder paths dictionary for prepare_dataset
                folder_paths = {}
                
                # Create temporary directories for each class
                for class_name, class_images in images_by_class.items():
                    if not class_images:
                        continue
                        
                    # Create a temporary folder for this class
                    # Use os.path.join to ensure correct path separators
                    class_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{class_name}_temp")
                    
                    # Make sure the directory exists and is empty
                    if os.path.exists(class_folder):
                        # Clean the directory before using it
                        for file in os.listdir(class_folder):
                            file_path = os.path.join(class_folder, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                            except Exception as e:
                                logging.error(f"Error deleting {file_path}: {e}")
                    
                    os.makedirs(class_folder, exist_ok=True)
                    logging.info(f"Created temporary class folder: {class_folder}")
                    
                    # Count how many images were actually copied
                    copied_count = 0
                    
                    # Copy images to the temporary folder
                    for img in class_images:
                        # Get source path - using dataset_id to properly locate the image
                        dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(dataset_id))
                        src_path = os.path.join(dataset_folder, img.filename)
                        
                        if os.path.exists(src_path):
                            dst_path = os.path.join(class_folder, os.path.basename(img.filename))
                            try:
                                shutil.copy2(src_path, dst_path)
                                copied_count += 1
                                logging.info(f"Copied image {img.filename} to {dst_path}")
                            except Exception as e:
                                logging.error(f"Error copying {src_path} to {dst_path}: {e}")
                        else:
                            # Try alternate path without dataset_id
                            alt_src_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
                            if os.path.exists(alt_src_path):
                                dst_path = os.path.join(class_folder, os.path.basename(img.filename))
                                try:
                                    shutil.copy2(alt_src_path, dst_path)
                                    copied_count += 1
                                    logging.info(f"Copied image {img.filename} (alternate path) to {dst_path}")
                                except Exception as e:
                                    logging.error(f"Error copying {alt_src_path} to {dst_path}: {e}")
                            else:
                                logging.warning(f"Image file not found: {src_path} or {alt_src_path}")
                    
                    # Only add to folder paths if images were copied
                    if copied_count >= 5:  # Need at least 5 images per class
                        folder_paths[class_name] = class_folder
                        logging.info(f"Added class folder with {copied_count} images: {class_name} -> {class_folder}")
                    else:
                        logging.warning(f"Only {copied_count} images were copied for class '{class_name}', need at least 5. This class will be skipped.")
                
                # Verify we have at least one class with images
                if not folder_paths:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to find any class with at least 5 images. Please add more images to each class.'
                    })
        else:
            # Process folder paths for server-side training or browser uploads
            folder_paths = data.get('folder_paths', {})
            
            # Validate folder paths
            for class_name, folder_path in folder_paths.items():
                if not os.path.exists(folder_path):
                    return jsonify({
                        'success': False,
                        'error': f'Folder path not found: {folder_path}'
                    })
        
        # Update global training status
        _training_status.update({
            'status': 'preparing', 
            'message': 'Preparing dataset files...', 
            'model_path': None,
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': epochs
        })
        
        # Define the training thread function
        def training_thread():
            try:
                # Import traceback at the beginning of the function
                import traceback
                
                # Define model save path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_name = f"{model_type}_{dataset.name}_{timestamp}.pt"
                model_path = os.path.join(app.config['MODELS_FOLDER'], model_name)
                
                # Ensure models directory exists
                os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
                
                # Make sure we have everything we need before proceeding
                if not isinstance(folder_paths, dict) or len(folder_paths) == 0:
                    error_message = f"Invalid folder paths data structure: {folder_paths}"
                    logging.error(error_message)
                    _training_status.update({
                        'status': 'error',
                        'message': error_message,
                        'error_details': traceback.format_exc()
                    })
                    raise ValueError(error_message)
                
                # Update status
                _training_status.update({
                    'status': 'preparing', 
                    'message': 'Preparing training data...',
                    'progress': 0
                })
                
                # Validate all folders before proceeding
                for class_name, folder_path in folder_paths.items():
                    if not os.path.exists(folder_path):
                        error_message = f"Folder path for class '{class_name}' does not exist: {folder_path}"
                        logging.error(error_message)
                        _training_status.update({
                            'status': 'error',
                            'message': error_message
                        })
                        raise ValueError(error_message)
                
                # Create output directory with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = os.path.join(os.getcwd(), f"prepared_dataset_{timestamp}")
                logging.info(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    # Log what we're about to do
                    logging.info("Calling prepare_dataset function")
                    logging.info(f"Folder paths: {folder_paths}")
                    logging.info(f"Output directory: {output_dir}")
                    
                    # Store the result in fresh variables with different names to avoid variable scope issues
                    result = prepare_dataset(
                        folder_paths,
                        output_dir,
                        validation_split=validation_split
                    )
                    
                    # Unpack results explicitly with proper scoping
                    dataset_path = result[0]
                    dataset_class_mapping = result[1]  # Use this instead of class_mapping
                    dataset_class_stats = result[2]
                    
                    # Log results for debugging
                    logging.info(f"Dataset preparation succeeded. Path: {dataset_path}")
                    logging.info(f"Class mapping: {dataset_class_mapping}")
                    
                except Exception as e:
                    error_message = f"Error preparing dataset: {str(e)}"
                    logging.error(error_message)
                    logging.error(traceback.format_exc())
                    _training_status.update({
                        'status': 'error',
                        'message': error_message,
                        'error_details': traceback.format_exc()
                    })
                    raise Exception(error_message)
                
                # Update status
                _training_status.update({
                    'status': 'training', 
                    'message': 'Training started...',
                    'progress': 0,
                    'current_epoch': 0,
                    'total_epochs': epochs
                })
                
                # Start training
                try:
                    # Make sure the dataset_path is valid
                    if not dataset_path or not os.path.exists(dataset_path):
                        raise ValueError(f"Invalid dataset path: {dataset_path}")
                    
                    # Log what we're about to do
                    logging.info(f"Starting training with folder: {dataset_path}")
                    
                    # Call train_yolo and store result first
                    result = train_yolo(
                        dataset_path,
                        epochs=epochs,
                        model_type=model_type,
                        use_gpu=use_gpu,
                        save_model_path=model_path,
                        validation_split=validation_split,
                        img_size=img_size,
                        batch_size=batch_size,
                        augmentation=augmentation,
                        patience=patience,
                        optimizer=optimizer,
                        lr0=lr0
                    )
                    
                    # Check if result is None or a tuple with None as first element
                    if result is None:
                        error_message = "Training failed with unknown error"
                        _training_status.update({
                            'status': 'error',
                            'message': error_message,
                            'error_details': error_message
                        })
                        raise Exception(error_message)
                    
                    # Now safely unpack
                    trained_model_path, training_stats = result
                    
                    if trained_model_path is None:
                        error_message = training_stats if isinstance(training_stats, str) else "Training failed with unknown error"
                        _training_status.update({
                            'status': 'error',
                            'message': error_message,
                            'error_details': error_message
                        })
                        raise Exception(error_message)
                    
                    # Training completed successfully
                    _training_status.update({
                        'status': 'completed',
                        'message': 'Training completed successfully.',
                        'model_path': trained_model_path,
                        'progress': 100,
                        'current_epoch': epochs,
                        'total_epochs': epochs,
                        'training_stats': training_stats
                    })
                    
                except Exception as e:
                    _training_status.update({
                        'status': 'error',
                        'message': f'Error during training: {str(e)}',
                        'error_details': traceback.format_exc()
                    })
                    raise
                
            except Exception as e:
                logging.error(f"Error in training thread: {str(e)}")
                import traceback
                traceback.print_exc()
                _training_status.update({
                    'status': 'error',
                    'message': f'Unexpected error: {str(e)}',
                    'error_details': traceback.format_exc()
                })
        
        # Start training in a background thread
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error in train_dataset_with_files: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.pt'):
            return jsonify({'success': False, 'error': 'Invalid file type. Only .pt files are allowed'})
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.getcwd(), app.config['MODELS_FOLDER'])
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the model file with a unique name if it already exists
        base_name = os.path.splitext(file.filename)[0]
        ext = os.path.splitext(file.filename)[1]
        model_path = os.path.join(models_dir, file.filename)
        
        # If file exists, append timestamp
        if os.path.exists(model_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(models_dir, f"{base_name}_{timestamp}{ext}")
        
        file.save(model_path)
        
        # Load the model to verify it and get its classes
        model = YOLO(model_path)
        if not model:
            return jsonify({'success': False, 'error': 'Failed to load model'})
        
        # Get class names from the model
        class_names = model.names
        if not class_names:
            return jsonify({'success': False, 'error': 'No classes found in model'})
        
        # Convert class names to a dictionary with class IDs as keys
        classes_dict = {str(i): name for i, name in class_names.items()}
        
        return jsonify({
            'success': True,
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'classes': classes_dict
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_model_classes', methods=['POST'])
def get_model_classes():
    try:
        model_path = request.form.get('model_path')
        if not model_path:
            return jsonify({'success': False, 'error': 'No model path provided'})
        
        # Try to get absolute paths of available models
        try:
            available_models_response = get_available_models()
            available_models_data = json.loads(available_models_response.get_data(as_text=True))
            
            if available_models_data.get('success'):
                model_paths = available_models_data.get('model_paths', {})
                
                # Check if the requested model is in our available models
                if model_path in model_paths:
                    # Use the absolute path we found during listing
                    absolute_model_path = model_paths[model_path]
                    logging.info(f"Found absolute path for {model_path}: {absolute_model_path}")
                    model_path = absolute_model_path
        except Exception as e:
            logging.error(f"Error retrieving model paths: {str(e)}")
            # Continue with get_model_path as fallback
        
        # Get the absolute path to the model
        model_path = get_model_path(model_path)
        
        # Load the model to get its classes
        logging.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        if not model:
            return jsonify({'success': False, 'error': 'Failed to load model'})
        
        # Get class names from the model
        class_names = model.names
        if not class_names:
            return jsonify({'success': False, 'error': 'No classes found in model'})
        
        # Convert class names to a dictionary with class IDs as keys
        classes_dict = {str(i): name for i, name in class_names.items()}
        
        return jsonify({
            'success': True,
            'classes': classes_dict,
            'model_name': os.path.basename(model_path)
        })
    except Exception as e:
        logging.error(f"Error getting model classes: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_detection_info')
def get_detection_info():
    try:
        # Get the latest detection results from the global variable
        global latest_detections
        if latest_detections is None:
            return jsonify({'success': True, 'detections': []})
            
        # Format detections for the frontend
        formatted_detections = []
        for det in latest_detections:
            formatted_detections.append({
                'class_id': int(det.boxes.cls.item()),
                'class_name': det.names[int(det.boxes.cls.item())],
                'confidence': float(det.boxes.conf.item()),
                'bbox': det.boxes.xyxy[0].tolist()
            })
            
        return jsonify({
            'success': True,
            'detections': formatted_detections
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/check_camera')
def check_camera():
    """Check if a camera is available and working"""
    try:
        # Try with DirectShow first (Windows specific)
        for camera_index in range(3):
            try:
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cap.release()
                        return jsonify({
                            'success': True,
                            'message': f'Camera available at index {camera_index} using DirectShow',
                            'index': camera_index,
                            'backend': 'dshow'
                        })
                if cap:
                    cap.release()
            except Exception as e:
                print(f"Error with DirectShow camera {camera_index}: {e}")
        
        # Try default backend
        for camera_index in range(3):
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cap.release()
                        return jsonify({
                            'success': True,
                            'message': f'Camera available at index {camera_index} using default backend',
                            'index': camera_index,
                            'backend': 'default'
                        })
                if cap:
                    cap.release()
            except Exception as e:
                print(f"Error with default backend camera {camera_index}: {e}")
        
        # No camera found
        return jsonify({
            'success': False,
            'message': 'No working camera found. Please check if your webcam is properly connected and not being used by another application.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error checking camera: {str(e)}'
        })

@app.route('/video_feed')
def video_feed():
    model_path = request.args.get('model_path')
    confidence_threshold = float(request.args.get('confidence_threshold', 0.25))
    source_type = request.args.get('source_type', 'webcam')
    video_path = request.args.get('video_path')
    selected_classes = request.args.get('selected_classes', '').split(',')
    selected_classes = [int(c) for c in selected_classes if c]  # Convert to integers and filter empty strings
    
    if not model_path:
        return "No model path provided", 400
        
    try:
        # First, check if we can use the cached model
        use_cached_model = False
        
        if hasattr(app, 'current_model_path') and app.current_model_path:
            # Check if it's the same model
            current_basename = os.path.basename(app.current_model_path)
            if model_path == current_basename or model_path == app.current_model_path:
                logging.info(f"Using cached model for video feed: {app.current_model_path}")
                use_cached_model = True
        
        # If not using cached model, load it
        if not use_cached_model:
            # Get the actual model path
            model_path = get_model_path(model_path)
            logging.info(f"Loading model for video feed: {model_path}")
            
            # Initialize model and cache it
            app.current_model = YOLO(model_path)
            app.current_model_path = model_path
        
        # Get source type and video path from stored values if not provided in request
        if hasattr(app, 'current_source_type') and not source_type:
            source_type = app.current_source_type
            
        if hasattr(app, 'current_video_path') and not video_path and source_type == 'video':
            video_path = app.current_video_path
            
        # Save current settings for future reference
        app.current_source_type = source_type
        if video_path:
            app.current_video_path = video_path
            
        # Get video source (webcam or file)
        if source_type == 'video' and video_path:
            source = video_path  # Use uploaded video file
            logging.info(f"Using video file as source: {video_path}")
        else:
            source = 0  # Default to webcam
            logging.info("Using webcam as source")
        
        def generate_frames():
            cap = None
            try:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video source: {source}")
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        # If video file ends, loop back to the beginning
                        if source != 0:  # Only rewind if it's a file, not webcam
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            success, frame = cap.read()
                            if not success:
                                break
                        else:
                            break
                    else:
                        try:
                            # Run YOLOv8 inference
                            results = app.current_model(frame, conf=confidence_threshold)
                            
                            # Filter results based on selected classes
                            if selected_classes:
                                # Get the class indices from the results
                                class_indices = results[0].boxes.cls.cpu().numpy()
                                # Create a mask for selected classes
                                mask = np.isin(class_indices, selected_classes)
                                # Apply the mask to filter boxes
                                results[0].boxes = results[0].boxes[mask]
                            
                            # Save results for API access
                            global latest_detections
                            latest_detections = results
                            
                            # Visualize results
                            annotated_frame = results[0].plot()
                            
                            # Convert frame to JPEG
                            ret, buffer = cv2.imencode('.jpg', annotated_frame)
                            frame = buffer.tobytes()
                            
                            # Yield frame in MJPEG format
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        except Exception as e:
                            logging.error(f"Error processing frame: {str(e)}")
                            # Output error frame
                            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(error_frame, f"Error: {str(e)}", (10, 240), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            ret, buffer = cv2.imencode('.jpg', error_frame)
                            frame = buffer.tobytes()
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            continue
            except Exception as e:
                logging.error(f"Error in video feed: {str(e)}")
                # Generate error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)}", (10, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            finally:
                if cap is not None:
                    cap.release()
            
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
                        
    except Exception as e:
        logging.error(f"Error in video_feed: {str(e)}")
        return str(e), 500

@app.route('/api/update_detection_classes', methods=['POST'])
def update_detection_classes():
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        selected_classes = data.get('selected_classes', [])
        confidence_threshold = float(data.get('confidence_threshold', 0.25))
        source_type = data.get('source_type', 'webcam')
        video_path = data.get('video_path')
        
        if not model_path:
            return jsonify({'success': False, 'error': 'No model path provided'}), 400
            
        # Validate selected classes
        try:
            selected_classes = [int(c) for c in selected_classes if c]
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid class IDs'}), 400
            
        # Use our model path helper to get the actual path
        model_path = get_model_path(model_path)
        logging.info(f"Update detection classes: using model {model_path}")
            
        # Update model if needed
        if not hasattr(app, 'current_model') or app.current_model_path != model_path:
            try:
                app.current_model = YOLO(model_path)
                app.current_model_path = model_path
                logging.info(f"Loaded model: {model_path}")
            except Exception as e:
                logging.error(f"Error loading model {model_path}: {str(e)}")
                return jsonify({'success': False, 'error': f'Error loading model: {str(e)}'}), 500
            
        # Store additional stream parameters
        app.current_source_type = source_type
        app.current_video_path = video_path
            
        return jsonify({
            'success': True,
            'message': 'Detection classes updated successfully',
            'selected_classes': selected_classes,
            'confidence_threshold': confidence_threshold,
            'source_type': source_type,
            'video_path': video_path
        })
        
    except Exception as e:
        logging.error(f"Error in update_detection_classes: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def stream_video(model, video_path, confidence=0.25, classes=None):
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Error: Could not open video", (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
        
        # Store the model for test API access
        app.current_model = model
        if not hasattr(app, 'current_model_path'):
            app.current_model_path = 'uploaded_model.pt'
            
        # Print model class names for debugging
        logging.info(f"Model class names (stream_video): {model.names}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                # Loop back to start of video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Convert to RGB for prediction (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with YOLO using direct predict method like in 1.py
            results = model.predict(
                source=frame_rgb,
                conf=confidence,
                classes=classes,
                verbose=True  # Set to True for debugging
            )
            
            # Save results for API access
            global latest_detections
            latest_detections = results
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Log detection results for debugging
            boxes = results[0].boxes
            logging.info(f"Detected {len(boxes)} objects")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                conf = float(box.conf[0].item())
                logging.info(f"  Detection {i}: class={cls_id} ({cls_name}), conf={conf:.4f}")
            
            # Add detection count
            num_detections = len(results[0].boxes)
            cv2.putText(
                annotated_frame, 
                f"Detections: {num_detections}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    except Exception as e:
        logging.error(f"Error in stream_video: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: {str(e)}", (10, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    finally:
        cap.release()

def stream_webcam(model, confidence=0.25, classes=None):
    """Stream webcam with YOLOv8 detection"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Error: Could not open webcam", (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
        
        # Store the model for test API access
        app.current_model = model
        if not hasattr(app, 'current_model_path'):
            app.current_model_path = 'uploaded_model.pt'
            
        # Print model class names for debugging
        logging.info(f"Model class names (stream_webcam): {model.names}")
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with YOLO - use same approach as 1.py
            # For debugging, print confidence threshold and classes
            logging.info(f"Detection with confidence: {confidence}, classes: {classes}")
            
            # Convert to RGB for prediction (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with YOLO
            results = model.predict(
                source=frame_rgb,
                conf=confidence,
                classes=classes,
                verbose=True  # Set to True for debugging
            )
            
            # Save results for API access
            global latest_detections
            latest_detections = results
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Log detection results for debugging
            boxes = results[0].boxes
            logging.info(f"Detected {len(boxes)} objects")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                conf = float(box.conf[0].item())
                logging.info(f"  Detection {i}: class={cls_id} ({cls_name}), conf={conf:.4f}")
            
            # Add detection count
            num_detections = len(results[0].boxes)
            cv2.putText(
                annotated_frame, 
                f"Detections: {num_detections}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
    except Exception as e:
        logging.error(f"Error in stream_webcam: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: {str(e)}", (10, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
              
    finally:
        cap.release()

@app.route('/api/process_video', methods=['POST'])
def process_video():
    try:
        # Get parameters from the request
        if 'video_file' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video_file']
        if video_file.filename == '':
            return jsonify({'error': 'No selected video'}), 400

        # Get file extension
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        if file_ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return jsonify({'error': f'Unsupported video format: {file_ext}. Please use .mp4, .avi, .mov, .mkv, or .webm'}), 400

        model_path = request.form.get('model_path')
        confidence_threshold = float(request.form.get('confidence_threshold', 0.25))
        selected_classes = request.form.getlist('selected_classes')

        # Load model if needed
        if not model_path:
            return jsonify({'error': 'No model path provided'}), 400

        try:
            model_path = get_model_path(model_path)
            print(f"Loading model from {model_path}")
            model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500

        # Create a unique filename for the uploaded video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_video_filename = f"temp_video_{timestamp}{file_ext}"
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_video_filename)
        
        # Create upload folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the video temporarily
        try:
            video_file.save(temp_video_path)
            print(f"Video saved to {temp_video_path}")
            
            # Verify the file was saved correctly
            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                return jsonify({'error': 'Failed to save video file'}), 500
                
            # Check if file is a valid video by attempting to open it
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                cap.release()
                return jsonify({'error': 'Uploaded file is not a valid video'}), 400
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            print(f"Valid video file: {temp_video_path}, FPS: {fps}, Frames: {frame_count}")
            
            # Return success response with video path that will be used in video_feed
            return jsonify({
                'success': True, 
                'video_path': temp_video_path,
                'fps': fps,
                'frame_count': frame_count
            })
            
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            return jsonify({'error': f'Error saving video: {str(e)}'}), 500

        # Convert selected_classes to integers if present
        if selected_classes:
            try:
                selected_classes = [int(c) for c in selected_classes if c]
            except ValueError:
                return jsonify({'error': 'Invalid class IDs provided'}), 400

    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<int:dataset_id>/rename_images', methods=['POST'])
def rename_images(dataset_id):
    """Rename images to associate them with a class."""
    try:
        # Check if dataset exists
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        class_name = data.get('class_name')
        filenames = data.get('filenames', [])
        
        if not class_name or not filenames:
            return jsonify({'success': False, 'error': 'Missing class name or filenames'}), 400
        
        # Check if the class exists in the dataset
        classes = json.loads(dataset.classes) if dataset.classes else []
        if class_name not in classes:
            return jsonify({'success': False, 'error': f'Class "{class_name}" does not exist in this dataset'}), 400
        
        # Get the upload folder path for this dataset
        upload_folder = app.config['UPLOAD_FOLDER']
        dataset_folder = os.path.join(upload_folder, str(dataset_id))
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Track renamed image count
        renamed_count = 0
        errors = []
        
        # Process each image
        for filename in filenames:
            try:
                # Get the image from database
                image = Image.query.filter_by(dataset_id=dataset_id, filename=filename).first()
                
                # If image not in database, check if file exists and create record
                if not image:
                    file_path = os.path.join(dataset_folder, filename)
                    if os.path.exists(file_path):
                        # Create new image record in database
                        image = Image(filename=filename, dataset_id=dataset_id)
                        db.session.add(image)
                        db.session.commit()
                        logging.info(f"Created new database record for existing file: {filename}")
                    else:
                        alt_file_path = os.path.join(upload_folder, filename)
                        if os.path.exists(alt_file_path):
                            # Copy file to dataset folder
                            dst_path = os.path.join(dataset_folder, filename)
                            shutil.copy2(alt_file_path, dst_path)
                            
                            # Create new image record in database
                            image = Image(filename=filename, dataset_id=dataset_id)
                            db.session.add(image)
                            db.session.commit()
                            logging.info(f"Copied file and created database record: {filename}")
                        else:
                            logging.warning(f"Image file not found: {filename}")
                            errors.append(f"Image file not found: {filename}")
                            continue
                
                # Generate a new filename with class prefix if it doesn't already have it
                if not filename.startswith(f"{class_name}_"):
                    # Extract file extension
                    _, ext = os.path.splitext(filename)
                    
                    # Create a new filename with the class prefix
                    new_filename = f"{class_name}_{int(time.time())}_{renamed_count}{ext}"
                    
                    # Get file paths
                    old_path = os.path.join(dataset_folder, filename)
                    new_path = os.path.join(dataset_folder, new_filename)
                    
                    # Rename the file if it exists on disk
                    if os.path.exists(old_path):
                        try:
                            shutil.move(old_path, new_path)
                            logging.info(f"Renamed file {old_path} to {new_path}")
                        except Exception as e:
                            logging.error(f"Error renaming file {old_path} to {new_path}: {e}")
                            errors.append(f"Error renaming file {filename}: {str(e)}")
                            continue
                    else:
                        # Try alternate locations
                        alt_path = os.path.join(upload_folder, filename)
                        if os.path.exists(alt_path):
                            try:
                                shutil.copy2(alt_path, new_path)
                                logging.info(f"Copied file {alt_path} to {new_path}")
                            except Exception as e:
                                logging.error(f"Error copying file {alt_path} to {new_path}: {e}")
                                errors.append(f"Error copying file {filename}: {str(e)}")
                                continue
                        else:
                            logging.warning(f"Image file not found for renaming: {filename}")
                            errors.append(f"Image file not found for renaming: {filename}")
                            continue
                    
                    # Update the database record
                    image.filename = new_filename
                    db.session.commit()
                    renamed_count += 1
            except Exception as e:
                logging.error(f"Error processing image {filename}: {e}")
                errors.append(f"Error processing {filename}: {str(e)}")
                db.session.rollback()
        
        if renamed_count > 0:
            result = {
                'success': True, 
                'message': f'Successfully renamed {renamed_count} images'
            }
            if errors:
                result['warnings'] = errors
            return jsonify(result)
        else:
            if errors:
                return jsonify({
                    'success': False, 
                    'error': 'No images were renamed due to errors',
                    'details': errors
                }), 400
            else:
                return jsonify({
                    'success': False, 
                    'error': 'No images were renamed. They may already have the correct prefix.'
                }), 400
            
    except Exception as e:
        logging.error(f"Error in rename_images: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_available_models')
def get_available_models():
    """Get a list of available models in the models directory"""
    try:
        # List of models with their paths
        available_models = []
        available_model_paths = {}
        
        # Check in the models directory
        models_dir = os.path.join(os.getcwd(), app.config['MODELS_FOLDER'])
        logging.info(f"Looking for models in: {models_dir}")
        
        # Also check the user's specified absolute path if different
        hardcoded_models_dir = "C:\\Users\\durga\\Desktop\\yolo4.0\\models"
        model_dirs = [models_dir]
        if hardcoded_models_dir != models_dir:
            model_dirs.append(hardcoded_models_dir)
            logging.info(f"Also checking for models in: {hardcoded_models_dir}")
        
        # Check all model directories
        for dir_path in model_dirs:
            # Ensure models directory exists
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    if f.endswith('.pt') and f not in available_models:
                        available_models.append(f)
                        available_model_paths[f] = os.path.join(dir_path, f)
                        logging.info(f"Found model: {f} at {available_model_paths[f]}")
        
        # Add default models that exist in the system
        default_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        for default_model in default_models:
            if default_model not in available_models:
                # Check current directory
                if os.path.exists(default_model):
                    available_models.append(default_model)
                    available_model_paths[default_model] = os.path.abspath(default_model)
                    logging.info(f"Found default model: {default_model} at {available_model_paths[default_model]}")
        
        # Sort models for consistent display
        available_models.sort()
        
        return jsonify({
            'success': True,
            'models': available_models,
            'model_paths': available_model_paths
        })
    except Exception as e:
        logging.error(f"Error in get_available_models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def create_app():
    global app
    
    # Initialize app attributes for model caching
    if not hasattr(app, 'current_model'):
        app.current_model = None
    
    if not hasattr(app, 'current_model_path'):
        app.current_model_path = None
    
    # Initialize YOLO model with default settings if available
    if app.current_model is None:
        initialize_model()
    
    return app

# Call create_app directly instead of using before_first_request
create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("Flask server starting... open http://127.0.0.1:5000 in your browser")
    app.run(host='0.0.0.0', debug=False)  # Using host 0.0.0.0 to make it accessible from other devices
