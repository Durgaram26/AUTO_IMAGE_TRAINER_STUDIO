import os
import shutil
import yaml
import json
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from flask import current_app

# Import needed models
from models import Class, Model

# Global variable to track training status
_training_status = {
    'status': 'none',
    'message': 'No training in progress',
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0
}

def create_yolo_dataset(project, split_ratios=(0.7, 0.2, 0.1)):
    """
    Create a YOLO dataset from project images and annotations
    """
    # Create dataset directories
    dataset_path = Path(current_app.config['YOLO_MODELS_DIR']) / f"project_{project.id}_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    # Create directory structure
    (dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # Create dataset.yaml
    classes = [cls.name for cls in project.classes]
    yaml_data = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open(dataset_path / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    # Distribute images according to split ratios
    splits = {'train': [], 'val': [], 'test': []}
    
    # Use existing splits if available
    for image in project.images:
        if image.split in ['train', 'val', 'test']:
            splits[image.split].append(image)
        else:
            splits['train'].append(image)
    
    # If no splits are defined, create them now
    if not any(len(s) > 0 for s in splits.values()):
        # Convert SQLAlchemy InstrumentedList to Python list before slicing
        images = list(project.images)
        train_count = int(len(images) * split_ratios[0])
        val_count = int(len(images) * split_ratios[1])
        
        splits['train'] = images[:train_count]
        splits['val'] = images[train_count:train_count+val_count]
        splits['test'] = images[train_count+val_count:]
        
        # Update database with split info
        for split, imgs in splits.items():
            for img in imgs:
                img.split = split
    
    # Process each split
    class_indices = {cls.name: i for i, cls in enumerate(project.classes)}
    
    for split, images in splits.items():
        for image in images:
            # Copy the image
            img_src = Path(current_app.config['UPLOAD_FOLDER']) / image.filename
            img_dst = dataset_path / "images" / split / Path(image.filename).name
            
            if img_src.exists():
                shutil.copy(img_src, img_dst)
                
                # Create label file (YOLO format)
                label_path = dataset_path / "labels" / split / f"{Path(image.filename).stem}.txt"
                
                # Get image dimensions for normalization
                img = Image.open(img_src)
                img_width, img_height = img.size
                
                # Write annotations in YOLO format
                with open(label_path, 'w') as f:
                    for ann in image.annotations:
                        # Skip if no bounding box
                        if ann.x_min is None or ann.y_min is None or ann.x_max is None or ann.y_max is None:
                            continue
                        
                        # Convert to YOLO format (center_x, center_y, width, height)
                        x_center = (ann.x_min + ann.x_max) / 2 / img_width
                        y_center = (ann.y_min + ann.y_max) / 2 / img_height
                        width = (ann.x_max - ann.x_min) / img_width
                        height = (ann.y_max - ann.y_min) / img_height
                        
                        # Get class name by querying Class model directly
                        class_obj = Class.query.get(ann.class_id)
                        class_idx = class_indices.get(class_obj.name, 0)
                        
                        # Write to file: class_idx x_center y_center width height
                        f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
    
    return str(dataset_path / "dataset.yaml")

def train_yolo_model(project, model_id, model_params):
    """
    Train a YOLOv8 model on the project dataset
    
    Args:
        project: Project object with annotations and images
        model_id: ID of the model being trained
        model_params: Dictionary with training parameters
    
    Returns:
        Dictionary with training results and weights path
    """
    import os
    from pathlib import Path
    
    # Create dataset
    dataset_yaml = create_yolo_dataset(project)
    
    # Set up model params
    model_type = model_params.get('model_type', 'yolov8n.pt')
    epochs = model_params.get('epochs', 100)
    batch_size = model_params.get('batch_size', 16)
    img_size = model_params.get('img_size', 640)
    patience = model_params.get('patience', 50)
    optimizer = model_params.get('optimizer', 'SGD')
    
    # Set up save directory
    save_dir = Path(current_app.config['YOLO_MODELS_DIR']) / f"project_{project.id}_model_{model_id}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    # Initialize model
    from ultralytics import YOLO
    model = YOLO(model_type)
    
    # Check for existing weights to resume training
    last_weights = save_dir / "weights" / "last.pt"
    if last_weights.exists():
        current_app.logger.info(f"Resuming training from {last_weights}")
        model = YOLO(last_weights)
    
    # Set up device
    device = model_params.get('device', None)  # None = auto detection of GPU
    
    # Create checkpoint callback to track progress
    class CheckpointCallback:
        def __init__(self, model_id, session):
            from models import Model
            self.model_id = model_id
            self.session = session
        
        def on_train_epoch_end(self, trainer):
            """Update the model's progress in the database after each epoch"""
            try:
                # Get model from database
                model_record = self.session.query(Model).filter_by(id=self.model_id).first()
                if model_record:
                    # Update current epoch
                    current_epoch = trainer.epoch + 1  # 0-indexed to 1-indexed
                    current_app.logger.info(f"Training epoch {current_epoch}/{model_record.epochs} completed")
                    
                    # Update the current_epoch in the model record
                    model_record.current_epoch = current_epoch
                    self.session.commit()
                    
                    # Log metrics if available
                    if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                        metrics = {}
                        if hasattr(trainer.metrics, 'box'):
                            metrics['map50'] = float(trainer.metrics.box.map50)
                            metrics['map'] = float(trainer.metrics.box.map)
                            current_app.logger.info(f"Current metrics: mAP50={metrics['map50']}, mAP={metrics['map']}")
                
            except Exception as e:
                current_app.logger.error(f"Error updating model progress: {str(e)}")
                import traceback
                current_app.logger.error(traceback.format_exc())
    
    # Train model
    try:
        # Import SQLAlchemy components
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create a new session for the callback
        engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
        Session = sessionmaker(bind=engine)
        callback_session = Session()
        
        # Define callbacks
        checkpoint_callback = CheckpointCallback(model_id, callback_session)
        
        # Train the model
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=str(Path(current_app.config['YOLO_MODELS_DIR'])),
            name=f"project_{project.id}_model_{model_id}",
            exist_ok=True,
            patience=patience,
            optimizer=optimizer,
            device=device,
            verbose=True,
            callbacks=[checkpoint_callback]
        )
        
        # Get best weights path
        weights_path = str(save_dir / "weights" / "best.pt")
        
        # Clean up
        callback_session.close()
        
        return {
            'weights_path': weights_path,
            'results': results
        }
        
    except Exception as e:
        current_app.logger.error(f"Error during model training: {str(e)}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        raise

def detect_with_yolo(model_path, image_path, conf=0.25):
    """
    Run inference on an image using a trained YOLOv8 model
    """
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf)
    
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(confidence),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    
    return detections

def export_annotations(project, format='coco'):
    """
    Export project annotations in specified format
    """
    if format == 'coco':
        return export_coco(project)
    elif format == 'yolo':
        return create_yolo_dataset(project)
    else:
        raise ValueError(f"Unsupported format: {format}")

def export_coco(project):
    """
    Export project annotations in COCO format
    """
    output_path = Path(current_app.config['UPLOAD_FOLDER']) / f"exports" / f"project_{project.id}_coco.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    coco = {
        "info": {
            "description": f"COCO dataset for {project.name}",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": project.owner.name,
            "date_created": project.created_at.isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i, cls in enumerate(project.classes):
        coco["categories"].append({
            "id": i + 1,
            "name": cls.name,
            "supercategory": ""
        })
    
    # Map class names to IDs
    class_map = {cls.name: i + 1 for i, cls in enumerate(project.classes)}
    
    # Add images and annotations
    annotation_id = 1
    for image in project.images:
        img_path = Path(current_app.config['UPLOAD_FOLDER']) / image.filename
        
        if not img_path.exists():
            continue
            
        img = Image.open(img_path)
        width, height = img.size
        
        img_id = image.id
        coco["images"].append({
            "id": img_id,
            "license": 1,
            "file_name": Path(image.filename).name,
            "height": height,
            "width": width,
            "date_captured": image.uploaded_at.isoformat()
        })
        
        # Add annotations
        for ann in image.annotations:
            # Skip if no bounding box
            if ann.x_min is None or ann.y_min is None or ann.x_max is None or ann.y_max is None:
                continue
                
            width = ann.x_max - ann.x_min
            height = ann.y_max - ann.y_min
            
            # Get class name by querying Class model directly
            class_obj = Class.query.get(ann.class_id)
            coco_ann = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": class_map.get(class_obj.name, 1),
                "bbox": [ann.x_min, ann.y_min, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            }
            
            # Add segmentation if available
            if ann.polygon_points:
                try:
                    coco_ann["segmentation"] = [ann.get_polygon_points()]
                except:
                    pass
                    
            coco["annotations"].append(coco_ann)
            annotation_id += 1
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(coco, f)
    
    return str(output_path)

def train_model_async(project, model_id, model_params, db):
    """
    Train a YOLOv8 model in a background thread and update model status in database
    
    Args:
        project: Project object
        model_id: ID of the model to train
        model_params: Dictionary with training parameters
        db: SQLAlchemy database instance
    """
    import threading
    from datetime import datetime
    from flask import current_app
    
    # Get the current application instance before starting the thread
    app = current_app._get_current_object()
    
    # Create global training status dict to track progress without using database
    global _training_status
    _training_status = {
        'status': 'starting', 
        'message': 'Initializing training...', 
        'model_path': None,
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': model_params.get('epochs', 100),
        'start_time': datetime.now().timestamp(),
        'model_id': model_id
    }
    
    def training_thread():
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import json
        import traceback
        
        global _training_status
        
        # Use the app instance passed from the parent function
        with app.app_context():
            try:
                _training_status['status'] = 'preparing'
                _training_status['message'] = 'Preparing dataset...'
                _training_status['progress'] = 10
                
                # Get model parameters
                model_type = model_params.get('model_type', 'yolov8n.pt')
                epochs = model_params.get('epochs', 100)
                batch_size = model_params.get('batch_size', 16)
                img_size = model_params.get('img_size', 640)
                patience = model_params.get('patience', 50)
                optimizer = model_params.get('optimizer', 'SGD')
                
                # Create dataset
                try:
                    dataset_yaml = create_yolo_dataset(project)
                    _training_status['status'] = 'training'
                    _training_status['message'] = 'Training model...'
                    _training_status['progress'] = 25
                except Exception as e:
                    _training_status['status'] = 'error'
                    _training_status['message'] = f'Dataset preparation failed: {str(e)}'
                    app.logger.error(f"Error preparing dataset: {str(e)}")
                    app.logger.error(traceback.format_exc())
                    return
                
                # Initialize model
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_type)
                    _training_status['progress'] = 30
                except Exception as e:
                    _training_status['status'] = 'error'
                    _training_status['message'] = f'Model initialization failed: {str(e)}'
                    app.logger.error(f"Error initializing model: {str(e)}")
                    app.logger.error(traceback.format_exc())
                    return
                
                # Set up save directory
                save_dir = app.config['YOLO_MODELS_DIR'] / f"project_{project.id}_model_{model_id}"
                if not save_dir.exists():
                    save_dir.mkdir(parents=True, exist_ok=True)
                
                # Define progress callback for epochs
                class ProgressCallback:
                    def __init__(self):
                        pass
                    
                    def on_train_epoch_end(self, trainer):
                        global _training_status
                        try:
                            current_epoch = trainer.epoch + 1  # 0-indexed to 1-indexed
                            _training_status['current_epoch'] = current_epoch
                            _training_status['progress'] = int(30 + (current_epoch / epochs) * 60)  # 30-90% progress during training
                            
                            app.logger.info(f"Training epoch {current_epoch}/{epochs} completed")
                            
                            # Log metrics if available
                            if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                                metrics = {}
                                if hasattr(trainer.metrics, 'box'):
                                    metrics['map50'] = float(trainer.metrics.box.map50)
                                    metrics['map'] = float(trainer.metrics.box.map)
                                    app.logger.info(f"Current metrics: mAP50={metrics['map50']}, mAP={metrics['map']}")
                                    _training_status['metrics'] = metrics
                        
                        except Exception as e:
                            app.logger.error(f"Error updating training progress: {str(e)}")
                
                # Start training
                try:
                    callback = ProgressCallback()
                    
                    # Train the model
                    results = model.train(
                        data=dataset_yaml,
                        epochs=epochs,
                        batch=batch_size,
                        imgsz=img_size,
                        project=str(app.config['YOLO_MODELS_DIR']),
                        name=f"project_{project.id}_model_{model_id}",
                        exist_ok=True,
                        patience=patience,
                        optimizer=optimizer,
                        verbose=True,
                        callbacks=[callback]
                    )
                    
                    # Training completed successfully
                    _training_status['status'] = 'saving'
                    _training_status['message'] = 'Saving trained model...'
                    _training_status['progress'] = 90
                    
                    # Get best weights path
                    weights_path = f"{save_dir}/weights/best.pt"
                    
                    # Update model status via SQLAlchemy session
                    try:
                        engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
                        Session = sessionmaker(bind=engine)
                        session = Session()
                        
                        # Using native SQL to avoid model import issues
                        sql = "UPDATE model SET status = 'complete', weights_path = :weights_path, completed_at = :completed_at WHERE id = :model_id"
                        session.execute(sql, {
                            'weights_path': weights_path,
                            'completed_at': datetime.now().isoformat(),
                            'model_id': model_id
                        })
                        session.commit()
                        session.close()
                    except Exception as db_err:
                        app.logger.error(f"Database update error: {str(db_err)}")
                        app.logger.error(traceback.format_exc())
                    
                    # Final status update
                    _training_status['status'] = 'complete'
                    _training_status['message'] = 'Training completed successfully!'
                    _training_status['progress'] = 100
                    _training_status['model_path'] = weights_path
                    
                    app.logger.info(f"Model {model_id} training completed successfully")
                    
                except Exception as e:
                    # Update status to failed
                    _training_status['status'] = 'error'
                    _training_status['message'] = f'Training failed: {str(e)}'
                    _training_status['progress'] = 0
                    
                    # Try to update database
                    try:
                        engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
                        Session = sessionmaker(bind=engine)
                        session = Session()
                        
                        # Using native SQL to avoid model import issues
                        sql = "UPDATE model SET status = 'failed' WHERE id = :model_id"
                        session.execute(sql, {'model_id': model_id})
                        session.commit()
                        session.close()
                    except Exception as db_err:
                        app.logger.error(f"Database update error: {str(db_err)}")
                    
                    app.logger.error(f"Error training model {model_id}: {str(e)}")
                    app.logger.error(traceback.format_exc())
            
            except Exception as e:
                _training_status['status'] = 'error'
                _training_status['message'] = f'Unexpected error: {str(e)}'
                app.logger.error(f"Error in training thread: {str(e)}")
                app.logger.error(traceback.format_exc())
    
    # Start training in a background thread
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    return thread 