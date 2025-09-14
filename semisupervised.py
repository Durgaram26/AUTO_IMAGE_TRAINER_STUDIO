import os
import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import shutil
import json
from flask import current_app
from models import Image, Annotation, Class, db

class SemiSupervisedAnnotator:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5, pseudo_label_threshold=0.8):
        """
        Initialize the semi-supervised annotation system.
        
        Args:
            model_path: Path to a pre-trained YOLO model
            confidence_threshold: Threshold for model predictions during annotation
            pseudo_label_threshold: Higher threshold for pseudo-labeling (more confident predictions)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.pseudo_label_threshold = pseudo_label_threshold
        self.logger = logging.getLogger(__name__)
        
    def predict_single_image(self, image_path):
        """Predict on a single image using the model"""
        results = self.model(image_path)[0]
        return results
    
    def create_pseudo_labels(self, unlabeled_images, project_id, class_mapping):
        """
        Create pseudo-labels for unlabeled images.
        
        Args:
            unlabeled_images: List of Image objects without annotations
            project_id: The project ID 
            class_mapping: Dictionary mapping class indices to class names
            
        Returns:
            count of images successfully pseudo-labeled
        """
        pseudo_labeled_count = 0
        
        for image in unlabeled_images:
            try:
                img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image.filename)
                
                if not os.path.exists(img_path):
                    self.logger.warning(f"Image file not found: {img_path}")
                    continue
                
                # Run prediction with high confidence threshold
                results = self.predict_single_image(img_path)
                
                # Get image dimensions
                img = cv2.imread(img_path)
                img_height, img_width = img.shape[:2]
                
                # Process detections and create annotations
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                # Only keep high confidence predictions for pseudo-labeling
                mask = scores >= self.pseudo_label_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                class_ids = class_ids[mask]
                
                if len(boxes) == 0:
                    self.logger.info(f"No high-confidence predictions for {image.filename}")
                    continue
                    
                # Create annotations for the image
                for box, score, class_id in zip(boxes, scores, class_ids):
                    if class_id not in class_mapping:
                        continue
                        
                    class_name = class_mapping[class_id]
                    class_obj = Class.query.filter_by(name=class_name, project_id=project_id).first()
                    
                    if not class_obj:
                        self.logger.warning(f"Class not found in database: {class_name}")
                        continue
                        
                    # Create annotation in database
                    annotation = Annotation(
                        image_id=image.id,
                        class_id=class_obj.id,
                        x_min=float(box[0]),
                        y_min=float(box[1]),
                        x_max=float(box[2]),
                        y_max=float(box[3]),
                        confidence=float(score)
                    )
                    
                    db.session.add(annotation)
                
                db.session.commit()
                image.is_annotated = True
                image.auto_annotated = True
                db.session.commit()
                pseudo_labeled_count += 1
                
            except Exception as e:
                self.logger.error(f"Error creating pseudo-labels for image {image.filename}: {str(e)}")
                db.session.rollback()
                
        return pseudo_labeled_count
    
    def train_from_existing_annotations(self, project_id, epochs=10, img_size=640, batch_size=8):
        """
        Fine-tune the model on existing annotations before pseudo-labeling.
        
        Args:
            project_id: The project ID
            epochs: Number of epochs for fine-tuning
            img_size: Image size for training
            batch_size: Batch size for training
            
        Returns:
            Path to the trained model weights
        """
        from utils import create_yolo_dataset
        
        # Create a temporary YOLO dataset from annotated images
        dataset_yaml = create_yolo_dataset(project_id)
        
        if not os.path.exists(dataset_yaml):
            raise ValueError(f"Failed to create dataset for training: {dataset_yaml}")
            
        # Train the model on existing annotations
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=5,
            verbose=True
        )
        
        # Return path to best model weights
        return self.model.ckpt_path
        
    def semi_supervised_workflow(self, project_id, class_names, active_learning_iterations=3):
        """
        Full semi-supervised workflow:
        1. Train model on initial labeled data
        2. Pseudo-label unlabeled data
        3. Retrain model on combined data
        4. Repeat if needed
        
        Args:
            project_id: The project ID
            class_names: List of class names in the project
            active_learning_iterations: Number of training/pseudo-labeling cycles
            
        Returns:
            Dict with results summary
        """
        # Create class mapping (name -> id)
        class_mapping = {i: name for i, name in enumerate(class_names)}
        inv_class_mapping = {name: i for i, name in enumerate(class_names)}
        
        results = {
            "iterations": [],
            "final_model_path": None,
            "total_pseudo_labeled": 0
        }
        
        # Get all images in the project
        all_images = Image.query.filter_by(project_id=project_id).all()
        labeled_images = [img for img in all_images if img.is_annotated and not img.auto_annotated]
        unlabeled_images = [img for img in all_images if not img.is_annotated]
        
        self.logger.info(f"Starting semi-supervised learning with {len(labeled_images)} labeled images and {len(unlabeled_images)} unlabeled images")
        
        if len(labeled_images) == 0:
            raise ValueError("No manually labeled images found. Please annotate some images first.")
            
        for iteration in range(active_learning_iterations):
            # Skip training on first iteration if using pre-trained model
            if iteration > 0 or len(labeled_images) >= 10:
                self.logger.info(f"Iteration {iteration+1}: Training model on {len(labeled_images)} labeled images")
                model_path = self.train_from_existing_annotations(project_id)
                self.model = YOLO(model_path)
            
            # Get remaining unlabeled images
            unlabeled_images = [img for img in all_images if not img.is_annotated]
            
            if not unlabeled_images:
                self.logger.info("No unlabeled images remaining")
                break
                
            # Pseudo-label some images
            batch_size = min(len(unlabeled_images), max(5, len(unlabeled_images) // (active_learning_iterations - iteration)))
            batch_to_label = unlabeled_images[:batch_size]
            
            self.logger.info(f"Pseudo-labeling {len(batch_to_label)} images")
            pseudo_labeled = self.create_pseudo_labels(batch_to_label, project_id, inv_class_mapping)
            
            results["total_pseudo_labeled"] += pseudo_labeled
            results["iterations"].append({
                "iteration": iteration + 1,
                "images_pseudo_labeled": pseudo_labeled,
                "total_labeled_images": len(labeled_images) + results["total_pseudo_labeled"]
            })
            
            self.logger.info(f"Iteration {iteration+1} complete: Pseudo-labeled {pseudo_labeled} images")
            labeled_images = [img for img in all_images if img.is_annotated]
            
        results["final_model_path"] = self.model.ckpt_path
        return results

def auto_annotate_project(project_id, confidence_threshold=0.5, pseudo_label_threshold=0.8):
    """
    Entry point for auto-annotation with semi-supervised learning.
    
    Args:
        project_id: The project ID
        confidence_threshold: Threshold for model predictions
        pseudo_label_threshold: Higher threshold for pseudo-labels
        
    Returns:
        Dict with results summary
    """
    from models import Project, Class
    
    project = Project.query.get_or_404(project_id)
    classes = Class.query.filter_by(project_id=project_id).all()
    class_names = [cls.name for cls in classes]
    
    # Check if we have class names
    if not class_names:
        raise ValueError("No classes defined for this project")
        
    # Initialize the semi-supervised annotator
    annotator = SemiSupervisedAnnotator(
        model_path="yolov8n.pt",
        confidence_threshold=confidence_threshold,
        pseudo_label_threshold=pseudo_label_threshold
    )
    
    # Run the semi-supervised workflow
    results = annotator.semi_supervised_workflow(
        project_id=project_id,
        class_names=class_names,
        active_learning_iterations=3
    )
    
    return results

# If this script is run directly, example usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        project_id = int(sys.argv[1])
        print(f"Auto-annotating project {project_id}")
        results = auto_annotate_project(project_id)
        print(f"Results: {results}")
    else:
        print("Usage: python semisupervised.py <project_id>") 