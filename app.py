import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, send_file, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
import threading
import traceback
import torch  # Import PyTorch
import shutil
import cv2
import numpy as np
from flask import copy_current_request_context
import time

# Load environment variables
load_dotenv()

# Import models
from models import db, User, Project, Image as ImageModel, Annotation, Class, Model, ImageTag

# Import utility functions
import utils

# Import the new semisupervised module
from semisupervised import auto_annotate_project

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///griffonder.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 20 * 1024 * 1024))  # 20MB
app.config['YOLO_MODELS_DIR'] = os.getenv('YOLO_MODELS_DIR', 'static/models')

# Initialize database
db.init_app(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Ensure upload directories exist
for directory in [app.config['UPLOAD_FOLDER'], app.config['YOLO_MODELS_DIR'], os.path.join(app.config['UPLOAD_FOLDER'], 'temp')]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Add this after other global variables
_training_status = {'status': 'idle', 'message': '', 'model_path': None, 'progress': 0}

# Create a dict to store training status per project
_training_status_by_project = {}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    # Auto-login as admin
    if not current_user.is_authenticated:
        admin = User.query.filter_by(username='admin').first()
        if admin:
            login_user(admin)
        else:
            # Create admin user if it doesn't exist
            admin = User(username='admin', email='admin@example.com', name='Admin User')
            admin.set_password('password')
            db.session.add(admin)
            db.session.commit()
            login_user(admin)
    
    return redirect(url_for('projects_page'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in or if direct access, redirect to home
    if current_user.is_authenticated or request.method == 'GET':
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('projects_page'))
        flash('Invalid username or password')
        
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email, name=name)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('projects_page'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/projects')
@login_required
def projects_page():
    projects = Project.query.filter_by(user_id=current_user.id).all()
    
    # Add preview image to each project if available
    for project in projects:
        preview_image = None
        if project.images:
            # Get the first image from the project to use as preview
            preview_image = project.images[0]
            project.preview_image = preview_image.filename
        else:
            project.preview_image = None
            
    return render_template('projects.html', projects=projects)

@app.route('/monitoring')
@login_required
def monitoring():
    return render_template('monitoring.html')

@app.route('/deployments')
@login_required
def deployments():
    # Get available models for the user
    models = Model.query.join(Project).filter(Project.user_id == current_user.id, Model.status == 'complete').all()
    
    # Create a list of model information for the template
    model_list = []
    for model in models:
        model_info = {
            'id': model.id,
            'name': model.name,
            'project_name': model.project.name,
            'project_id': model.project_id,
            'created_at': model.created_at,
            'type': model.model_type
        }
        model_list.append(model_info)
    
    return render_template('deployments.html', models=model_list)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/universe')
@login_required
def universe():
    return render_template('universe.html')

@app.route('/help')
@login_required
def help_docs():
    return render_template('help.html')

@app.route('/notifications')
@login_required
def notifications():
    return render_template('notifications.html')

# Project management routes
@app.route('/projects/new')
@login_required
def new_project():
    return render_template('create_project.html')

@app.route('/projects/create', methods=['POST'])
@login_required
def create_project():
    # Get form data
    project_name = request.form.get('project_name')
    project_type = request.form.get('project_type')
    annotation_group = request.form.get('annotation_group')
    license = request.form.get('license')
    
    # Create new project
    project = Project(
        name=project_name,
        type=project_type.replace('-', ' ').title(),
        visibility='Private',
        annotation_group=annotation_group,
        license=license,
        user_id=current_user.id
    )
    
    db.session.add(project)
    db.session.commit()
    
    # Add a default class for the project
    default_class = Class(
        name=annotation_group if annotation_group else "default",
        color="#6c5ce7",
        project_id=project.id
    )
    db.session.add(default_class)
    db.session.commit()
    
    return redirect(url_for('project_detail', project_id=project.id))

@app.route('/projects/<int:project_id>')
@login_required
def project_detail(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Format dates for display
    project.edited = format_last_edited(project.updated_at)
    
    return render_template('project_detail.html', project=project)

@app.route('/projects/<int:project_id>/upload', methods=['GET', 'POST'])
@login_required
def upload(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        app.logger.info("Upload request received")
        batch_name = request.form.get('batch_name', f'Uploaded on {datetime.now().strftime("%m/%d/%y at %I:%M %p")}')
        tags = request.form.get('tags', '').split(',')
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        # Debug request information
        app.logger.info(f"Form data: batch_name={batch_name}, tags={tags}")
        app.logger.info(f"Files in request: {list(request.files.keys())}")
        
        # Check if files were uploaded
        if 'files[]' not in request.files:
            app.logger.warning("No 'files[]' in request.files")
            flash('No files selected')
            return redirect(request.url)
            
        files = request.files.getlist('files[]')
        
        # Debug information
        app.logger.info(f"Received {len(files)} files for upload")
        app.logger.info(f"File names: {[f.filename for f in files if f.filename]}")
        
        if len(files) == 0:
            flash('No files were selected')
            return redirect(request.url)
            
        uploaded_count = 0
        error_count = 0
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    app.logger.info(f"Processing file: {file.filename}")
                    # Generate secure filename with UUID to avoid collisions
                    filename = secure_filename(file.filename)
                    
                    # Handle folder paths by preserving directory structure
                    relative_path = os.path.dirname(filename)
                    upload_dir = app.config['UPLOAD_FOLDER']
                    
                    if relative_path:
                        # Create the directory structure if it doesn't exist
                        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], relative_path)
                        app.logger.info(f"Creating directory: {upload_dir}")
                        os.makedirs(upload_dir, exist_ok=True)
                    
                    unique_filename = f"{uuid.uuid4().hex}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    
                    # Save the file
                    app.logger.info(f"Saving file to: {file_path}")
                    file.save(file_path)
                    
                    # Get image dimensions
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                            format = img.format.lower()
                            app.logger.info(f"Image dimensions: {width}x{height}, format: {format}")
                    except Exception as img_err:
                        app.logger.error(f"Error processing image {filename}: {str(img_err)}")
                        os.remove(file_path)  # Remove invalid file
                        error_count += 1
                        continue
                        
                    # Create database record
                    image = ImageModel(
                        filename=unique_filename,
                        original_filename=filename,
                        width=width,
                        height=height,
                        format=format,
                        filesize=os.path.getsize(file_path),
                        batch_name=batch_name,
                        project_id=project.id
                    )
                    
                    db.session.add(image)
                    db.session.commit()
                    app.logger.info(f"Added image to database with ID: {image.id}")
                    
                    # Add tags if any
                    for tag_name in tags:
                        tag = ImageTag(name=tag_name, image_id=image.id)
                        db.session.add(tag)
                    
                    db.session.commit()
                    uploaded_count += 1
                except Exception as e:
                    error_count += 1
                    app.logger.error(f"Error uploading file {file.filename}: {str(e)}")
                    # Log the full exception traceback
                    import traceback
                    app.logger.error(traceback.format_exc())
            else:
                if file and file.filename:
                    app.logger.warning(f"Invalid file type: {file.filename}")
                    error_count += 1
        
        app.logger.info(f"Upload complete: {uploaded_count} successful, {error_count} failed")
        
        if uploaded_count > 0:
            flash(f'{uploaded_count} file(s) uploaded successfully')
        if error_count > 0:
            flash(f'{error_count} file(s) failed to upload')
        if uploaded_count == 0 and error_count == 0:
            flash('No files were uploaded. Make sure your files are valid images.')
        return redirect(url_for('project_detail', project_id=project.id))
    
    current_date = datetime.now().strftime('%m/%d/%y at %I:%M %p')
    return render_template('upload.html', project=project, current_date=current_date)

@app.route('/projects/<int:project_id>/annotate')
@login_required
def annotate(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Get the current image to annotate
    image_id = request.args.get('image_id', None)
    
    if image_id:
        current_image = ImageModel.query.filter_by(id=image_id, project_id=project.id).first()
    else:
        # Get the first image without annotations or the first image
        current_image = ImageModel.query.outerjoin(Annotation).filter(
            ImageModel.project_id == project.id,
            Annotation.id.is_(None)
        ).first() or ImageModel.query.filter_by(project_id=project.id).first()
    
    if not current_image:
        flash('No images available for annotation')
        return redirect(url_for('project_detail', project_id=project.id))
    
    # Get all images for this project with annotation counts
    images = []
    for img in ImageModel.query.filter_by(project_id=project.id).all():
        # Get annotation count for each image
        ann_count = Annotation.query.filter_by(image_id=img.id).count()
        img.annotations_count = ann_count
        images.append(img)
    
    # Get project classes
    classes = Class.query.filter_by(project_id=project.id).all()
    
    # Get image count and position
    image_count = len(images)
    
    # Find position of current image in list
    image_position = 1
    for i, img in enumerate(images):
        if img.id == current_image.id:
            image_position = i + 1
            break
    
    return render_template('annotate.html', 
                          project=project, 
                          current_image=current_image,
                          images=images,
                          classes=classes,
                          image_count=image_count,
                          image_position=image_position)

@app.route('/projects/<int:project_id>/analytics')
@login_required
def analytics(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    generation_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    
    # Get analytics data
    image_count = ImageModel.query.filter_by(project_id=project.id).count()
    annotation_count = db.session.query(Annotation).join(ImageModel).filter(ImageModel.project_id == project.id).count()
    
    # Get class distribution
    class_distribution = db.session.query(
        Class.name, Class.color, db.func.count(Annotation.id)
    ).join(
        Annotation, Class.id == Annotation.class_id
    ).filter(
        Class.project_id == project.id
    ).group_by(Class.id).all()
    
    # Format class distribution data
    classes_data = []
    for name, color, count in class_distribution:
        classes_data.append({
            'name': name,
            'color': color,
            'count': count
        })
    
    # Calculate average annotations per image
    avg_annotations = annotation_count / image_count if image_count > 0 else 0
    
    # Get image size statistics
    image_sizes = ImageModel.query.filter_by(project_id=project.id).with_entities(
        ImageModel.width, ImageModel.height
    ).all()
    
    if image_sizes:
        megapixels = [(w * h) / 1000000 for w, h in image_sizes]
        avg_mp = sum(megapixels) / len(megapixels)
        min_mp = min(megapixels)
        max_mp = max(megapixels)
        
        # Get median image size for aspect ratio
        sorted_by_area = sorted(image_sizes, key=lambda x: x[0] * x[1])
        median_width, median_height = sorted_by_area[len(sorted_by_area) // 2]
    else:
        avg_mp = min_mp = max_mp = 0
        median_width = median_height = 0
    
    analytics_data = {
        'image_count': image_count,
        'annotation_count': annotation_count,
        'avg_annotations': avg_annotations,
        'classes': classes_data,
        'avg_mp': avg_mp,
        'min_mp': min_mp,
        'max_mp': max_mp,
        'median_width': median_width,
        'median_height': median_height,
    }
    
    return render_template('analytics.html', 
                          project=project, 
                          generation_date=generation_date,
                          analytics=analytics_data)

@app.route('/projects/<int:project_id>/classes')
@login_required
def classes(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Get all classes for this project with annotation counts
    classes = db.session.query(
        Class, db.func.count(Annotation.id).label('count')
    ).outerjoin(
        Annotation, Class.id == Annotation.class_id
    ).filter(
        Class.project_id == project.id
    ).group_by(Class.id).all()
    
    # Format class data
    classes_data = []
    for cls, count in classes:
        classes_data.append({
            'id': cls.id,
            'name': cls.name,
            'color': cls.color,
            'count': count
        })
    
    # Get all tags
    tags = db.session.query(
        ImageTag, db.func.count(db.distinct(ImageTag.image_id)).label('count')
    ).join(
        ImageModel, ImageTag.image_id == ImageModel.id
    ).filter(
        ImageModel.project_id == project.id
    ).group_by(ImageTag.name).all()
    
    # Format tag data
    tags_data = []
    for tag, count in tags:
        tags_data.append({
            'id': tag.id,
            'name': tag.name,
            'count': count,
            'created_at': tag.created_at.strftime('%B %d, %Y') if tag.created_at else ''
        })
    
    return render_template('classes.html', 
                          project=project,
                          classes=classes_data,
                          tags=tags_data)

@app.route('/projects/<int:project_id>/models')
@login_required
def project_models(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Get all models for this project
    models = Model.query.filter_by(project_id=project_id).order_by(Model.created_at.desc()).all()
    
    # Process models to add metrics for display
    processed_models = []
    for model in models:
        model_data = {
            'id': model.id,
            'name': model.name,
            'model_type': model.model_type,
            'status': model.status,
            'created_at': model.created_at,
            'completed_at': model.completed_at,
            'current_epoch': model.current_epoch,
            'epochs': model.epochs,
            'version': model.version,
            'weights_path': model.weights_path
        }
        
        if model.status == 'complete':
            # Get metrics using the get_metrics method which handles JSON serialization
            try:
                model_data['metrics'] = model.get_metrics()
            except:
                model_data['metrics'] = {
                    'map50': '0.00',
                    'map': '0.00',
                    'precision': '0.00',
                    'recall': '0.00'
                }
        elif model.status == 'training':
            # For demonstration, set a random progress value
            import random
            model_data['current_epoch'] = random.randint(1, model.epochs)
        
        processed_models.append(model_data)
    
    return render_template('models.html', project=project, models=processed_models)

@app.route('/projects/<int:project_id>/train', methods=['GET', 'POST'])
@login_required
def train_model(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Get all models for this project for the real-time training section
    models = Model.query.filter_by(project_id=project_id).order_by(Model.created_at.desc()).all()
    
    if request.method == 'POST':
        # Create a new model record
        model_name = request.form.get('model_name', f"{project.name} Model")
        model_type = request.form.get('model_type', 'yolov8n.pt')
        epochs = int(request.form.get('epochs', 100))
        batch_size = int(request.form.get('batch_size', 16))
        img_size = int(request.form.get('img_size', 640))
        device = request.form.get('device', 'auto')
        
        model = Model(
            name=model_name,
            model_type=model_type,
            status='creating',
            project_id=project.id,
            epochs=epochs,
            batch_size=batch_size,
            image_size=img_size,
            created_at=datetime.now()
        )
        
        db.session.add(model)
        db.session.commit()
        
        # Start training in background thread using reference_sample.py approach
        try:
            # Initialize global training status
            global _training_status
            _training_status = {
                'status': 'starting', 
                'message': 'Initializing training...', 
                'model_path': None,
                'progress': 0,
                'current_epoch': 0,
                'total_epochs': epochs,
                'start_time': datetime.now().timestamp(),
                'model_id': model.id,
                'project_id': project_id
            }
            
            # Get the Project and all its data within this session
            project_data = {
                'id': project.id,
                'name': project.name,
                'classes': [{'id': cls.id, 'name': cls.name} for cls in project.classes],
                'images': []
            }
            
            # Get all project images with their annotations
            for image in project.images:
                img_data = {
                    'id': image.id,
                    'filename': image.filename,
                    'width': image.width,
                    'height': image.height,
                    'split': image.split or 'train',
                    'annotations': []
                }
                
                for ann in image.annotations:
                    class_obj = Class.query.get(ann.class_id)
                    ann_data = {
                        'class_id': ann.class_id,
                        'class_name': class_obj.name if class_obj else 'unknown',
                        'x_min': ann.x_min,
                        'y_min': ann.y_min,
                        'x_max': ann.x_max,
                        'y_max': ann.y_max
                    }
                    img_data['annotations'].append(ann_data)
                
                project_data['images'].append(img_data)
            
            # Start training thread with project data
            train_thread = threading.Thread(
                target=train_yolo_thread,
                args=(project_data, model.id, {
                    'model_type': model_type,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'img_size': img_size,
                    'device': device
                })
            )
            train_thread.daemon = True
            train_thread.start()
            
            flash('Model training started successfully!')
        except Exception as e:
            app.logger.error(f"Error starting model training: {str(e)}")
            app.logger.error(traceback.format_exc())
            model.status = 'failed'
            db.session.commit()
            flash('Error starting model training: ' + str(e), 'error')
        
        return redirect(url_for('train_model', project_id=project.id))
            
    # Count images by split
    training_images = ImageModel.query.filter_by(project_id=project_id, split='train').count()
    val_images = ImageModel.query.filter_by(project_id=project_id, split='val').count()
    test_images = ImageModel.query.filter_by(project_id=project_id, split='test').count()
    
    return render_template('train.html', 
                          project=project,
                          models=models,
                          training_images=training_images,
                          val_images=val_images,
                          test_images=test_images,
                          training_status=_training_status)

@app.route('/api/projects/<int:project_id>/training_status')
@login_required
def get_training_status(project_id):
    """Get the current training status for a project"""
    global _training_status
    global _training_status_by_project
    
    # Check if user has access to this project
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Get active model training for this project
    active_model = Model.query.filter_by(project_id=project_id, status='training').order_by(Model.created_at.desc()).first()
    
    # Check if this is a training in progress for this project in the global status dict
    if project_id in _training_status_by_project:
        status = _training_status_by_project[project_id]
        
        # Add timestamp to help with frontend updates
        status['timestamp'] = datetime.now().timestamp()
        
        # Add model info if available
        if 'model_id' in status and status['model_id']:
            model = Model.query.get(status['model_id'])
            if model:
                status['model_name'] = model.name
                status['model_type'] = model.model_type
        
        return jsonify(status)
    
    # Backward compatibility for older training jobs
    if _training_status.get('project_id') == project_id:
        # Add timestamp to help with frontend updates
        _training_status['timestamp'] = datetime.now().timestamp()
        return jsonify(_training_status)
    
    # If there's an active model being trained but not in our status dictionary
    # (could happen if the server was restarted during training)
    if active_model:
        status = {
            'status': 'training',
            'message': f'Training model... Epoch {active_model.current_epoch or 0}/{active_model.epochs}',
            'progress': min(90, int(30 + ((active_model.current_epoch or 0) / active_model.epochs) * 60)),
            'current_epoch': active_model.current_epoch or 0,
            'total_epochs': active_model.epochs,
            'model_id': active_model.id,
            'project_id': project_id,
            'model_name': active_model.name,
            'model_type': active_model.model_type,
            'timestamp': datetime.now().timestamp()
        }
        return jsonify(status)
    
    # No active training
    return jsonify({
        'status': 'idle', 
        'message': 'No active training', 
        'progress': 0,
        'timestamp': datetime.now().timestamp()
    })

# API endpoints
@app.route('/api/projects/<int:project_id>/images')
@login_required
def api_get_images(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    images = ImageModel.query.filter_by(project_id=project.id).all()
    
    result = []
    for image in images:
        result.append({
            'id': image.id,
            'filename': image.filename,
            'original_filename': image.original_filename,
            'width': image.width,
            'height': image.height,
            'uploaded_at': image.uploaded_at.isoformat(),
            'url': url_for('get_image', filename=image.filename)
        })
    
    return jsonify({'images': result})

@app.route('/api/projects/<int:project_id>/classes')
@login_required
def api_get_classes(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    classes = Class.query.filter_by(project_id=project.id).all()
    
    result = []
    for cls in classes:
        result.append({
            'id': cls.id,
            'name': cls.name,
            'color': cls.color
        })
    
    return jsonify({'classes': result})

@app.route('/api/projects/<int:project_id>/classes', methods=['POST'])
@login_required
def api_create_class(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    data = request.json
    name = data.get('name')
    color = data.get('color', '#6c5ce7')
    
    if not name:
        return jsonify({'error': 'Class name is required'}), 400
    
    # Check if class already exists
    if Class.query.filter_by(name=name, project_id=project.id).first():
        return jsonify({'error': 'Class already exists'}), 400
    
    # Create new class
    cls = Class(name=name, color=color, project_id=project.id)
    db.session.add(cls)
    db.session.commit()
    
    return jsonify({
        'id': cls.id,
        'name': cls.name,
        'color': cls.color
    })

@app.route('/api/images/<int:image_id>/annotations')
@login_required
def api_get_annotations(image_id):
    image = ImageModel.query.get_or_404(image_id)
    project = Project.query.get_or_404(image.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    annotations = Annotation.query.filter_by(image_id=image_id).all()
    
    result = []
    for ann in annotations:
        class_obj = Class.query.get(ann.class_id)
        ann_data = {
            'id': ann.id,
            'class_id': ann.class_id,
            'class_name': class_obj.name,
            'class_color': class_obj.color
        }
        
        # Add specific annotation data based on type
        if ann.x_min is not None:  # Bounding box
            ann_data.update({
                'type': 'box',
                'x_min': ann.x_min,
                'y_min': ann.y_min,
                'x_max': ann.x_max,
                'y_max': ann.y_max
            })
        elif ann.polygon_points:  # Segmentation mask
            ann_data.update({
                'type': 'polygon',
                'points': ann.get_polygon_points()
            })
        elif ann.keypoints:  # Keypoints
            ann_data.update({
                'type': 'keypoints',
                'points': ann.get_keypoints()
            })
        elif ann.label:  # Text annotation
            ann_data.update({
                'type': 'text',
                'label': ann.label
            })
        
        result.append(ann_data)
    
    return jsonify({'annotations': result})

@app.route('/api/images/<int:image_id>/annotations', methods=['POST'])
@login_required
def api_create_annotation(image_id):
    image = ImageModel.query.get_or_404(image_id)
    project = Project.query.get_or_404(image.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    class_id = data.get('class_id')
    
    if not class_id:
        return jsonify({'error': 'Class ID is required'}), 400
    
    # Create new annotation
    annotation = Annotation(
        image_id=image_id,
        class_id=class_id
    )
    
    # Set annotation data based on type
    ann_type = data.get('type', 'box')
    
    if ann_type == 'box':
        annotation.x_min = data.get('x_min')
        annotation.y_min = data.get('y_min')
        annotation.x_max = data.get('x_max')
        annotation.y_max = data.get('y_max')
    elif ann_type == 'polygon':
        annotation.set_polygon_points(data.get('points', []))
    elif ann_type == 'keypoints':
        annotation.set_keypoints(data.get('points', []))
    elif ann_type == 'text':
        annotation.label = data.get('label', '')
    
    db.session.add(annotation)
    db.session.commit()
    
    return jsonify({
        'id': annotation.id,
        'type': ann_type,
        'class_id': annotation.class_id
    })

@app.route('/api/annotations/<int:annotation_id>', methods=['PUT'])
@login_required
def api_update_annotation(annotation_id):
    annotation = Annotation.query.get_or_404(annotation_id)
    image = ImageModel.query.get_or_404(annotation.image_id)
    project = Project.query.get_or_404(image.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    
    # Update class if provided
    if 'class_id' in data:
        annotation.class_id = data['class_id']
    
    # Update annotation data based on type
    ann_type = data.get('type')
    
    if ann_type == 'box':
        if 'x_min' in data: annotation.x_min = data['x_min']
        if 'y_min' in data: annotation.y_min = data['y_min']
        if 'x_max' in data: annotation.x_max = data['x_max']
        if 'y_max' in data: annotation.y_max = data['y_max']
    elif ann_type == 'polygon' and 'points' in data:
        annotation.set_polygon_points(data['points'])
    elif ann_type == 'keypoints' and 'points' in data:
        annotation.set_keypoints(data['points'])
    elif ann_type == 'text' and 'label' in data:
        annotation.label = data['label']
    
    db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/api/annotations/<int:annotation_id>', methods=['DELETE'])
@login_required
def api_delete_annotation(annotation_id):
    annotation = Annotation.query.get_or_404(annotation_id)
    image = ImageModel.query.get_or_404(annotation.image_id)
    project = Project.query.get_or_404(image.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(annotation)
    db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/api/images/<int:image_id>/annotations/batch', methods=['POST'])
@login_required
def api_create_annotations_batch(image_id):
    image = ImageModel.query.get_or_404(image_id)
    project = Project.query.get_or_404(image.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    annotations_data = data.get('annotations', [])
    
    if not annotations_data:
        return jsonify({'error': 'No annotations provided'}), 400
    
    # First, delete existing annotations for this image
    Annotation.query.filter_by(image_id=image_id).delete()
    
    # Create new annotations
    created_annotations = []
    for ann_data in annotations_data:
        class_id = ann_data.get('class_id')
        if not class_id:
            continue
            
        # Create new annotation
        annotation = Annotation(
            image_id=image_id,
            class_id=class_id
        )
        
        # Set annotation data based on type
        ann_type = ann_data.get('type', 'box')
        
        if ann_type == 'box':
            annotation.x_min = ann_data.get('x_min')
            annotation.y_min = ann_data.get('y_min')
            annotation.x_max = ann_data.get('x_max')
            annotation.y_max = ann_data.get('y_max')
        elif ann_type == 'polygon':
            annotation.set_polygon_points(ann_data.get('points', []))
        elif ann_type == 'keypoints':
            annotation.set_keypoints(ann_data.get('points', []))
        elif ann_type == 'text':
            annotation.label = ann_data.get('label', '')
        
        db.session.add(annotation)
    
    # Commit all annotations at once
    db.session.commit()
    
    # Return the newly created annotations in the response
    annotations = Annotation.query.filter_by(image_id=image_id).all()
    result = []
    for ann in annotations:
        class_obj = Class.query.get(ann.class_id)
        ann_data = {
            'id': ann.id,
            'class_id': ann.class_id,
            'class_name': class_obj.name,
            'class_color': class_obj.color,
            'type': 'box'  # Default type
        }
        
        # Add specific annotation data based on type
        if ann.x_min is not None:  # Bounding box
            ann_data.update({
                'type': 'box',
                'x_min': ann.x_min,
                'y_min': ann.y_min,
                'x_max': ann.x_max,
                'y_max': ann.y_max
            })
        elif ann.polygon_points:  # Segmentation mask
            ann_data.update({
                'type': 'polygon',
                'points': ann.get_polygon_points()
            })
        
        result.append(ann_data)
    
    return jsonify({'status': 'success', 'annotations': result})

@app.route('/api/projects/<int:project_id>/models')
@login_required
def api_get_models(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    models = Model.query.filter_by(project_id=project.id).all()
    
    result = []
    for model in models:
        result.append({
            'id': model.id,
            'name': model.name,
            'version': model.version,
            'status': model.status,
            'created_at': model.created_at.isoformat(),
            'completed_at': model.completed_at.isoformat() if model.completed_at else None,
        })
    
    return jsonify({'models': result})

@app.route('/api/projects/<int:project_id>/export', methods=['POST'])
@login_required
def api_export_project(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    format = request.json.get('format', 'coco')
    
    try:
        output_path = utils.export_annotations(project, format)
        
        return jsonify({
            'status': 'success',
            'format': format,
            'file_path': output_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/images/<int:image_id>')
@login_required
def api_get_image(image_id):
    image = ImageModel.query.get_or_404(image_id)
    project = Project.query.get_or_404(image.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify({
        'image': {
            'id': image.id,
            'filename': image.filename,
            'original_filename': image.original_filename,
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'filesize': image.filesize,
            'batch_name': image.batch_name,
            'split': image.split,
            'uploaded_at': image.uploaded_at.isoformat(),
            'url': url_for('get_image', filename=image.filename)
        }
    })

@app.route('/api/images/<int:image_id>/split', methods=['PUT'])
@login_required
def update_image_split(image_id):
    image = ImageModel.query.filter_by(id=image_id).first_or_404()
    
    # Check if the user has access to this image
    project = Project.query.filter_by(id=image.project_id, user_id=current_user.id).first_or_404()
    
    # Get the split parameter from request
    data = request.get_json()
    if not data or 'split' not in data:
        return jsonify({"error": "Split type is required"}), 400
    
    split_type = data['split']
    
    # Validate split type
    if split_type not in ['train', 'val', 'test']:
        return jsonify({"error": "Invalid split type. Must be 'train', 'val', or 'test'"}), 400
    
    try:
        # Update the image split
        image.split = split_type
        db.session.commit()
    
        return jsonify({
                "message": "Image split updated successfully",
                "image_id": image_id,
                "split": split_type
            })
    except Exception as e:
        db.session.rollback()
        print(f"Error updating image split: {str(e)}")
        return jsonify({"error": "An error occurred while updating image split"}), 500

@app.route('/api/models/<int:model_id>/predict', methods=['POST'])
@login_required
def api_predict(model_id):
    model = Model.query.get_or_404(model_id)
    project = Project.query.get_or_404(model.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Check if model is trained
    if model.status != 'completed' or not model.weights_path:
        return jsonify({'error': 'Model is not trained'}), 400
    
    # Get image from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Run inference
        try:
            detections = utils.detect_with_yolo(model.weights_path, temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return jsonify({
                'status': 'success',
                'detections': detections
            })
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

# Utility routes
@app.route('/uploads/<path:filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Utility functions
def allowed_file(filename):
    """Check if file has an allowed extension for image uploads"""
    if not filename:
        return False
        
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'avif'}
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    app.logger.info(f"Checking file: {filename}, extension: {extension}")
    
    is_allowed = extension in ALLOWED_EXTENSIONS
    if not is_allowed:
        app.logger.warning(f"File type not allowed: {filename} with extension {extension}")
    
    return is_allowed

def format_last_edited(date):
    """Format the last edited date in a human-readable format"""
    now = datetime.now()
    diff = now - date
    
    if diff.days == 0:
        hours = diff.seconds // 3600
        if hours == 0:
            minutes = diff.seconds // 60
            if minutes == 0:
                return 'just now'
            elif minutes == 1:
                return '1 minute ago'
            else:
                return f'{minutes} minutes ago'
        elif hours == 1:
            return '1 hour ago'
        else:
            return f'{hours} hours ago'
    elif diff.days == 1:
        return 'yesterday'
    elif diff.days < 7:
        return f'{diff.days} days ago'
    elif diff.days < 30:
        weeks = diff.days // 7
        if weeks == 1:
            return '1 week ago'
        else:
            return f'{weeks} weeks ago'
    elif diff.days < 365:
        months = diff.days // 30
        if months == 1:
            return '1 month ago'
        else:
            return f'{months} months ago'
    else:
        years = diff.days // 365
        if years == 1:
            return '1 year ago'
        else:
            return f'{years} years ago'

@app.context_processor
def inject_globals():
    """Inject global variables into all templates."""
    return {
        'current_date': datetime.now().strftime('%m/%d/%y at %I:%M %p')
    }

# Command to create the database
@app.cli.command("init-db")
def init_db():
    db.create_all()
    print('Database initialized.')

# Command to seed the database with sample data
@app.cli.command("seed-db")
def seed_db():
    # Create admin user if it doesn't exist
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', name='Admin User')
        admin.set_password('password')
        db.session.add(admin)
        db.session.commit()
        print('Admin user created successfully.')
    else:
        print('Admin user already exists.')
    
    print('Database seeding completed.')

@app.route('/projects/<int:project_id>/delete', methods=['POST'])
@login_required
def delete_project(project_id):
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Get project name for flash message
    project_name = project.name
    
    # Delete the project (cascade will handle related records)
    db.session.delete(project)
    db.session.commit()
    
    flash(f'Project "{project_name}" has been deleted')
    return redirect(url_for('projects_page'))

@app.route('/models/<int:model_id>/download/<format>')
@login_required
def download_model(model_id, format):
    model = Model.query.filter_by(id=model_id).first_or_404()
    project = Project.query.filter_by(id=model.project_id, user_id=current_user.id).first_or_404()
    
    # Check if format is valid
    if format not in ['pt', 'onnx', 'tflite']:
        return jsonify({"error": "Invalid format. Must be 'pt', 'onnx', or 'tflite'"}), 400
    
    # Get the path to the model file
    if format == 'pt':
        file_path = model.weights_path
        content_type = 'application/octet-stream'
        filename = f"{model.name}.pt"
    elif format == 'onnx':
        # In a real app, you would convert the model to ONNX format or load an existing ONNX file
        file_path = model.weights_path.replace('.pt', '.onnx')
        content_type = 'application/octet-stream'
        filename = f"{model.name}.onnx"
    elif format == 'tflite':
        # In a real app, you would convert the model to TFLite format or load an existing TFLite file
        file_path = model.weights_path.replace('.pt', '.tflite')
        content_type = 'application/octet-stream'
        filename = f"{model.name}.tflite"
    
    # Check if file exists
    if not os.path.exists(file_path):
        # For demonstration, we'll return a dummy file
        return Response(
            "This is a placeholder for the model file download.",
            mimetype='text/plain',
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )
    
    # Return the file for download
    return send_file(
        file_path,
        mimetype=content_type,
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
@login_required
def delete_model(model_id):
    """Delete a model and associated files"""
    try:
        model = Model.query.get_or_404(model_id)
        project = Project.query.get_or_404(model.project_id)
        
        # Check if user has access to this project
        if project.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Save model name for response message
        model_name = model.name
        project_id = project.id
        
        # Delete model weights if they exist
        if model.weights_path and os.path.exists(model.weights_path):
            try:
                os.remove(model.weights_path)
                app.logger.info(f"Deleted model weights file: {model.weights_path}")
            except Exception as e:
                app.logger.error(f"Error deleting model weights file {model.weights_path}: {str(e)}")
        
        # Delete model directory if it exists
        model_dir = os.path.dirname(model.weights_path) if model.weights_path else None
        if model_dir and os.path.exists(model_dir):
            try:
                import shutil
                shutil.rmtree(model_dir)
                app.logger.info(f"Deleted model directory: {model_dir}")
            except Exception as e:
                app.logger.error(f"Error deleting model directory {model_dir}: {str(e)}")
        
        # Clean up related dataset directories
        try:
            # Look for dataset folders with pattern "project_{project_id}_dataset*"
            models_dir = app.config['YOLO_MODELS_DIR']
            if os.path.exists(models_dir):
                # Find the timestamp from the model
                model_timestamp = None
                # Extract timestamp from directory like "project_1_model_12"
                model_path_parts = model_dir.split('_') if model_dir else []
                for part in model_path_parts:
                    if len(part) > 10 and part.isdigit():
                        model_timestamp = part
                        break
                
                # Look for matching dataset folders
                for item in os.listdir(models_dir):
                    # If we have a timestamp, look for exact match; otherwise match any dataset for this project
                    if (model_timestamp and f"project_{project_id}_dataset_{model_timestamp}" == item) or \
                       (not model_timestamp and item.startswith(f"project_{project_id}_dataset")):
                        item_path = os.path.join(models_dir, item)
                        if os.path.isdir(item_path):
                            try:
                                shutil.rmtree(item_path)
                                app.logger.info(f"Deleted dataset directory associated with model: {item_path}")
                            except Exception as e:
                                app.logger.error(f"Error deleting dataset directory {item_path}: {str(e)}")
        except Exception as e:
            app.logger.error(f"Error cleaning up dataset directories: {str(e)}")
        
        # Clear training status if it exists for this project
        if project_id in _training_status_by_project and _training_status_by_project[project_id].get('model_id') == model_id:
            _training_status_by_project.pop(project_id, None)
            app.logger.info(f"Cleared training status for project {project_id}, model {model_id}")
        
        # Delete the model record from the database
        db.session.delete(model)
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': f'Model "{model_name}" has been deleted'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to delete model {model_id}: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/models/<int:model_id>/metrics')
@login_required
def api_get_model_metrics(model_id):
    model = Model.query.get_or_404(model_id)
    project = Project.query.get_or_404(model.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Check if model is trained
    if model.status != 'completed' or not model.weights_path:
        return jsonify({'error': 'Model is not trained'}), 400
    
    # Get model metrics
    metrics = model.get_metrics()
    
    return jsonify({'metrics': metrics})

@app.route('/api/models/<int:model_id>/status')
@login_required
def api_get_model_status(model_id):
    """Get the current status and metrics of a model"""
    model = Model.query.get_or_404(model_id)
    project = Project.query.get_or_404(model.project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    response = {
        'id': model.id,
        'name': model.name,
        'status': model.status,
        'created_at': model.created_at.isoformat() if model.created_at else None,
        'completed_at': model.completed_at.isoformat() if model.completed_at else None,
        'epochs': model.epochs,
        'current_epoch': model.current_epoch,
    }
    
    # Add metrics if available and training is complete
    if model.status == 'complete' and model.metrics:
        response['metrics'] = model.get_metrics()
    
    return jsonify(response)

@app.route('/api/models/<int:model_id>/retry', methods=['POST'])
@login_required
def retry_model_training(model_id):
    """Retry training for a failed model"""
    try:
        model = Model.query.get_or_404(model_id)
        project = Project.query.get_or_404(model.project_id)
        
        # Check if user has access to this project
        if project.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Check if model is in failed status
        if model.status != 'failed':
            return jsonify({'error': 'Only failed models can be retried'}), 400
        
        # Reset model status to 'created'
        model.status = 'created'
        model.current_epoch = 0
        db.session.commit()
        
        return jsonify({
            'success': True,
            'redirect': url_for('train_model', project_id=project.id)
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to retry model training {model_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<int:model_id>/cancel', methods=['POST'])
@login_required
def cancel_model_training(model_id):
    """Cancel training for a model in progress"""
    try:
        model = Model.query.get_or_404(model_id)
        project = Project.query.get_or_404(model.project_id)
        
        # Check if user has access to this project
        if project.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Check if model is in training status
        if model.status != 'training':
            return jsonify({'error': 'Only models in training can be cancelled'}), 400
        
        # Update model status to 'cancelled'
        model.status = 'cancelled'
        db.session.commit()
        
        # Clear training status if it exists for this project
        project_id = project.id
        if project_id in _training_status_by_project:
            _training_status_by_project[project_id]['status'] = 'cancelled'
            _training_status_by_project[project_id]['message'] = 'Training cancelled by user'
        
        return jsonify({
            'success': True,
            'message': 'Training cancelled successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to cancel model training {model_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/projects/<int:project_id>/annotation_count')
@login_required
def get_annotation_count(project_id):
    """Get the total annotation count for a project"""
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Count annotations for all images in this project
    count = db.session.query(db.func.count(Annotation.id)).join(
        ImageModel, Annotation.image_id == ImageModel.id
    ).filter(
        ImageModel.project_id == project_id
    ).scalar()
    
    return jsonify({'count': count})

@app.route('/api/projects/<int:project_id>/images_by_split')
@login_required
def api_get_images_by_split(project_id):
    """Get images for a project with optional filtering by split"""
    project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
    
    # Get query parameters
    split = request.args.get('split')
    limit = request.args.get('limit', 12, type=int)
    
    # Build query
    query = ImageModel.query.filter_by(project_id=project.id)
    
    # Filter by split if specified
    if split in ['train', 'val', 'test']:
        query = query.filter_by(split=split)
    
    # Get images with limit
    images = query.limit(limit).all()
    
    result = []
    for image in images:
        # Count annotations for this image
        ann_count = Annotation.query.filter_by(image_id=image.id).count()
        
        result.append({
            'id': image.id,
            'filename': image.filename,
            'original_filename': image.original_filename,
            'width': image.width,
            'height': image.height,
            'uploaded_at': image.uploaded_at.isoformat(),
            'annotations_count': ann_count,
            'url': url_for('get_image', filename=image.filename)
        })
    
    return jsonify({'images': result})

def create_yolo_dataset_from_data(project_data, output_dir, split_ratios=(0.7, 0.2, 0.1)):
    """
    Create a YOLO dataset from project data without SQLAlchemy dependencies
    """
    import os
    import yaml
    import shutil
    from pathlib import Path
    
    # Create dataset directories
    dataset_path = Path(output_dir).resolve()  # Convert to absolute path
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    # Create directory structure
    (dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # Extract classes
    classes = [cls['name'] for cls in project_data['classes']]
    
    # Create dataset.yaml with absolute paths
    yaml_data = {
        'path': str(dataset_path),  # Use absolute path
        'train': str(dataset_path / 'images/train'),  # Use absolute paths for train/val/test
        'val': str(dataset_path / 'images/val'),
        'test': str(dataset_path / 'images/test'),
        'nc': len(classes),
        'names': classes
    }
    
    with open(dataset_path / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    # Collect images by split
    splits = {'train': [], 'val': [], 'test': []}
    
    # Use existing splits if available
    for image in project_data['images']:
        if image['split'] in ['train', 'val', 'test']:
            splits[image['split']].append(image)
        else:
            splits['train'].append(image)
    
    # If no splits are defined, create them now
    if not any(len(s) > 0 for s in splits.values()):
        images = project_data['images']
        train_count = int(len(images) * split_ratios[0])
        val_count = int(len(images) * split_ratios[1])
        
        splits['train'] = images[:train_count]
        splits['val'] = images[train_count:train_count+val_count]
        splits['test'] = images[train_count+val_count:]
    
    # Ensure at least one image in each split to avoid errors
    # If any split is empty, move one image from train to that split
    for split_name in ['val', 'test']:
        if len(splits[split_name]) == 0 and len(splits['train']) > 0:
            app.logger.warning(f"No images in {split_name} split, moving one from train")
            splits[split_name].append(splits['train'].pop())
    
    # Map class names to indices
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    # Process each split
    for split, images in splits.items():
        for image in images:
            # Copy the image with absolute paths
            img_src = Path(app.config['UPLOAD_FOLDER']).resolve() / image['filename']
            img_dst = dataset_path / "images" / split / Path(image['filename']).name
            
            if img_src.exists():
                shutil.copy(img_src, img_dst)
                
                # Create label file (YOLO format)
                label_path = dataset_path / "labels" / split / f"{Path(image['filename']).stem}.txt"
                
                # Use image dimensions for normalization
                img_width = image['width']
                img_height = image['height']
                
                # Track valid annotations
                valid_annotations = 0
                
                # Write annotations in YOLO format
                with open(label_path, 'w') as f:
                    for ann in image['annotations']:
                        # Skip if no bounding box
                        if not all(k in ann for k in ['x_min', 'y_min', 'x_max', 'y_max']) or \
                           any(ann[k] is None for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                            continue
                        
                        try:
                            # Ensure coordinates are within image bounds
                            x_min = max(0, min(ann['x_min'], img_width))
                            y_min = max(0, min(ann['y_min'], img_height))
                            x_max = max(0, min(ann['x_max'], img_width))
                            y_max = max(0, min(ann['y_max'], img_height))
                            
                            # Skip invalid boxes (zero width/height)
                            if x_min >= x_max or y_min >= y_max:
                                app.logger.warning(f"Skipping invalid box in {image['filename']}: coords ({x_min},{y_min},{x_max},{y_max})")
                                continue
                            
                            # Convert to YOLO format (center_x, center_y, width, height)
                            x_center = (x_min + x_max) / 2 / img_width
                            y_center = (y_min + y_max) / 2 / img_height
                            width = (x_max - x_min) / img_width
                            height = (y_max - y_min) / img_height
                            
                            # Final validation of normalized coordinates
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                                app.logger.warning(f"Skipping out-of-range coordinates in {image['filename']}: "
                                                  f"normalized ({x_center:.4f},{y_center:.4f},{width:.4f},{height:.4f})")
                                continue
                            
                            # Get class index
                            class_idx = class_indices.get(ann['class_name'], 0)
                            
                            # Write to file: class_idx x_center y_center width height
                            f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            valid_annotations += 1
                        except Exception as e:
                            app.logger.error(f"Error processing annotation in {image['filename']}: {str(e)}")
                            continue
                            
                # If no valid annotations were written, create an empty label file
                if valid_annotations == 0:
                    app.logger.warning(f"No valid annotations for {image['filename']}")
    
    # Log the number of images in each split
    app.logger.info(f"Dataset created at {dataset_path} with:")
    for split, images in splits.items():
        app.logger.info(f"  - {split}: {len(images)} images")
    
    return str(dataset_path / "dataset.yaml")

# Clear CUDA cache
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

# Create a custom callbacks class for YOLO training to track progress and update database
def register_training_callbacks(model, model_id, project_id):
    """
    Register custom callbacks to track training progress and update the database
    
    Args:
        model: YOLO model instance
        model_id: ID of the model in the database
        project_id: ID of the project
        
    Returns:
        None
    """
    from ultralytics.engine.trainer import BaseTrainer
    
    class TrainingProgressCallback:
        """Custom callback for tracking training progress"""
        
        def __init__(self, model_id, project_id):
            self.model_id = model_id
            self.project_id = project_id
            self.start_time = datetime.now()
        
        def on_pretrain_routine_end(self, trainer):
            """Called when pretrain routine ends"""
            global _training_status_by_project
            
            status = _training_status_by_project.get(self.project_id, {})
            status.update({
                'status': 'starting',
                'message': 'Starting training...',
                'progress': 30,
                'model_id': self.model_id,
                'project_id': self.project_id,
                'total_epochs': trainer.epochs,
                'current_epoch': 0
            })
            _training_status_by_project[self.project_id] = status
            
            # Update model in database
            try:
                with app.app_context():
                    db_model = Model.query.get(self.model_id)
                    if db_model:
                        db_model.status = 'training'
                        db.session.commit()
            except Exception as e:
                app.logger.error(f"Failed to update model status in database: {str(e)}")
                
            app.logger.info(f"Training started for model {self.model_id} in project {self.project_id}")
            
        def on_train_epoch_start(self, trainer):
            """Called when a training epoch starts"""
            current_epoch = trainer.epoch + 1
            
            global _training_status_by_project
            status = _training_status_by_project.get(self.project_id, {})
            status.update({
                'status': 'training',
                'message': f'Training epoch {current_epoch}/{trainer.epochs}',
                'progress': int(30 + (current_epoch - 1) / trainer.epochs * 60),
                'current_epoch': current_epoch
            })
            _training_status_by_project[self.project_id] = status
            
            # Update epoch in database
            try:
                with app.app_context():
                    db_model = Model.query.get(self.model_id)
                    if db_model:
                        db_model.current_epoch = current_epoch
                        db.session.commit()
            except Exception as e:
                app.logger.error(f"Failed to update model epoch in database: {str(e)}")
                
            app.logger.info(f"Starting epoch {current_epoch}/{trainer.epochs} for model {self.model_id}")
            
        def on_train_epoch_end(self, trainer):
            """Called when a training epoch ends"""
            current_epoch = trainer.epoch + 1
            metrics = trainer.metrics
            
            # Calculate progress based on current epoch
            progress = int(30 + (current_epoch / trainer.epochs) * 60)
            
            # Extract and format metrics
            metric_dict = {}
            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        metric_dict[k] = float(v)
                    
            elapsed_time = datetime.now() - self.start_time
            elapsed_minutes = elapsed_time.total_seconds() / 60
            
            # Estimate remaining time
            if current_epoch > 0:
                minutes_per_epoch = elapsed_minutes / current_epoch
                remaining_epochs = trainer.epochs - current_epoch
                estimated_remaining_minutes = minutes_per_epoch * remaining_epochs
                
                # Format estimated time
                if estimated_remaining_minutes > 60:
                    estimated_time = f"{estimated_remaining_minutes/60:.1f} hours"
                else:
                    estimated_time = f"{estimated_remaining_minutes:.1f} minutes"
            else:
                estimated_time = "calculating..."
            
            # Update status
            global _training_status_by_project
            status = _training_status_by_project.get(self.project_id, {})
            status.update({
                'status': 'training',
                'message': f'Training epoch {current_epoch}/{trainer.epochs} completed',
                'progress': progress,
                'current_epoch': current_epoch,
                'metrics': metric_dict,
                'elapsed_time': f"{elapsed_minutes:.1f} minutes",
                'estimated_remaining_time': estimated_time
            })
            _training_status_by_project[self.project_id] = status
            
            # Update epoch and metrics in database
            try:
                with app.app_context():
                    db_model = Model.query.get(self.model_id)
                    if db_model:
                        db_model.current_epoch = current_epoch
                        # Store metrics as JSON
                        if hasattr(db_model, 'set_metrics') and metric_dict:
                            db_model.set_metrics(metric_dict)
                        db.session.commit()
            except Exception as e:
                app.logger.error(f"Failed to update model metrics in database: {str(e)}")
                
            app.logger.info(f"Completed epoch {current_epoch}/{trainer.epochs} for model {self.model_id}")
            
            # Log metrics
            for k, v in metric_dict.items():
                app.logger.info(f"Epoch {current_epoch} {k}: {v:.4f}")
            
            # Clear GPU memory after each epoch
            clear_gpu_memory()
            
        def on_train_end(self, trainer):
            """Called when training ends"""
            # Update status to completed
            global _training_status_by_project
            status = _training_status_by_project.get(self.project_id, {})
            status.update({
                'status': 'complete',
                'message': 'Training completed successfully',
                'progress': 100
            })
            _training_status_by_project[self.project_id] = status
            
            app.logger.info(f"Training completed for model {self.model_id}")
            
        def on_fit_epoch_end(self, trainer):
            """Called after fit but before validation at the end of an epoch"""
            # This ensures we show progress even during validation
            current_epoch = trainer.epoch + 1
            
            # Update status
            global _training_status_by_project
            status = _training_status_by_project.get(self.project_id, {})
            status.update({
                'message': f'Validating epoch {current_epoch}/{trainer.epochs}',
            })
            _training_status_by_project[self.project_id] = status
            
    # Create and register callbacks
    callback = TrainingProgressCallback(model_id, project_id)
    
    # Register all callbacks with the model
    model.add_callback("on_pretrain_routine_end", callback.on_pretrain_routine_end)
    model.add_callback("on_train_epoch_start", callback.on_train_epoch_start) 
    model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
    model.add_callback("on_train_end", callback.on_train_end)
    model.add_callback("on_fit_epoch_end", callback.on_fit_epoch_end)
    
    return model

def train_yolo_thread(project_data, model_id, model_params):
    global _training_status
    global _training_status_by_project
    
    # Store project-specific training status
    project_id = project_data['id']
    _training_status_by_project[project_id] = {
        'status': 'preparing',
        'message': 'Preparing dataset...',
        'progress': 10,
        'current_epoch': 0,
        'total_epochs': model_params['epochs'],
        'model_id': model_id,
        'project_id': project_id
    }
    
    try:
        app.logger.info(f"Starting training thread for model {model_id} in project {project_id}")
        
        # Reference to status object for this project
        status = _training_status_by_project[project_id]
        
        # Get parameters
        model_type = model_params.get('model_type', 'yolov8n.pt')
        epochs = model_params.get('epochs', 100)
        batch_size = model_params.get('batch_size', 16)
        # Lower the batch size to prevent memory issues
        batch_size = min(batch_size, 8)  # Limit batch size to 8 or lower
        img_size = model_params.get('img_size', 640)
        device = model_params.get('device', 'auto')
        
        # Check CUDA availability and handle device parameter
        if device == 'auto':
            if torch.cuda.is_available():
                device = 0  # Use the first GPU if available
            else:
                device = 'cpu'  # Fall back to CPU if no CUDA is available
                app.logger.warning("CUDA not available, falling back to CPU for training")
        
        # Convert device to the right type - YOLO expects int for GPU index
        if device != 'cpu' and device != 'mps':
            try:
                device = int(device)  # Convert to integer for GPU index
            except ValueError:
                # If conversion fails, fall back to CPU
                app.logger.warning(f"Invalid device value '{device}', falling back to CPU")
                device = 'cpu'
        
        app.logger.info(f"Using device '{device}' for training")
        
        # For backward compatibility, also update global status
        _training_status['status'] = 'preparing'
        _training_status['message'] = 'Preparing dataset...'
        _training_status['progress'] = 10
        _training_status['project_id'] = project_id
        _training_status['model_id'] = model_id

        # Create dataset directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(app.config['YOLO_MODELS_DIR'], f"project_{project_id}_dataset_{timestamp}")
        output_dir = os.path.abspath(output_dir)  # Convert to absolute path
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the dataset
        try:
            dataset_yaml = create_yolo_dataset_from_data(project_data, output_dir)
            status['status'] = 'training'
            status['message'] = 'Training model...'
            status['progress'] = 25
            app.logger.info(f"Dataset created successfully at {dataset_yaml}")
        except Exception as e:
            status['status'] = 'error'
            status['message'] = f'Dataset preparation failed: {str(e)}'
            app.logger.error(f"Error preparing dataset: {str(e)}")
            app.logger.error(traceback.format_exc())
            
            # Update model status in database
            try:
                with app.app_context():
                    model = Model.query.get(model_id)
                    if model:
                        model.status = 'failed'
                        db.session.commit()
            except Exception as db_err:
                app.logger.error(f"Failed to update model status: {str(db_err)}")
            
            return
        
        # Load YOLO model
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(model_type)
            
            # Register callbacks for progress tracking
            yolo_model = register_training_callbacks(yolo_model, model_id, project_id)
            
            status['progress'] = 30
        except Exception as e:
            status['status'] = 'error'
            status['message'] = f'Model initialization failed: {str(e)}'
            app.logger.error(f"Error initializing model: {str(e)}")
            app.logger.error(traceback.format_exc())
            
            # Update model status in database
            try:
                with app.app_context():
                    db_model = Model.query.get(model_id)
                    if db_model:
                        db_model.status = 'failed'
                        db.session.commit()
            except Exception as db_err:
                app.logger.error(f"Failed to update model status: {str(db_err)}")
                
            return
        
        # Set up save directory
        save_dir = os.path.join(app.config['YOLO_MODELS_DIR'], f"project_{project_data['id']}_model_{model_id}")
        save_dir = os.path.abspath(save_dir)  # Convert to absolute path
        os.makedirs(save_dir, exist_ok=True)
        app.logger.info(f"Model will be saved to {save_dir}")
            
        # Start training
        try:
            # Clear GPU memory before training
            clear_gpu_memory()
            
            # Train the model using the Ultralytics API with optimized parameters
            results = yolo_model.train(
                data=dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=os.path.dirname(save_dir),  # Use parent directory
                name=os.path.basename(save_dir),    # Use just the directory name
                exist_ok=True,
                patience=50,  # Early stopping
                verbose=True,
                device=device,
                workers=0,  # Reduce workers to prevent memory issues
                cache=False,  # Disable caching to reduce memory usage
                close_mosaic=10,  # Close mosaic augmentation earlier to reduce memory
                amp=False,  # Disable mixed precision to prevent memory issues
                optimizer="Adam",  # Use Adam optimizer for better memory efficiency
                cos_lr=True,  # Use cosine learning rate scheduler for stability
            )
            
            # Training completed successfully
            status['status'] = 'saving'
            status['message'] = 'Saving trained model...'
            status['progress'] = 90
            
            # Get best weights path
            weights_path = os.path.join(save_dir, "weights", "best.pt")
            
            # Update model record in database
            try:
                with app.app_context():
                    db_model = Model.query.get(model_id)
                    if db_model:
                        db_model.status = 'complete'
                        db_model.weights_path = weights_path
                        db_model.completed_at = datetime.now()
                        
                        # Set metrics if available
                        if hasattr(results, 'metrics') and hasattr(db_model, 'set_metrics'):
                            metrics = {}
                            if hasattr(results.metrics, 'box'):
                                metrics['map50'] = float(results.metrics.box.map50)
                                metrics['map'] = float(results.metrics.box.map)
                                metrics['precision'] = float(results.metrics.box.precision)
                                metrics['recall'] = float(results.metrics.box.recall)
                            
                            if hasattr(results.metrics, 'fitness'):
                                metrics['fitness'] = float(results.metrics.fitness)
                                
                            db_model.set_metrics(metrics)
                            
                        db.session.commit()
            except Exception as db_err:
                app.logger.error(f"Error updating model in database: {str(db_err)}")
                app.logger.error(traceback.format_exc())
            
            # Final status update
            status['status'] = 'complete'
            status['message'] = 'Training completed successfully!'
            status['progress'] = 100
            status['model_path'] = weights_path
            
            app.logger.info(f"Model {model_id} training completed successfully")
            
        except Exception as e:
            # Update status for error
            status['status'] = 'error'
            status['message'] = f'Training failed: {str(e)}'
            status['progress'] = 0
            
            # Update model status in database
            try:
                with app.app_context():
                    db_model = Model.query.get(model_id)
                    if db_model:
                        db_model.status = 'failed'
                        db.session.commit()
            except Exception as db_err:
                app.logger.error(f"Failed to update model status: {str(db_err)}")
                
            app.logger.error(f"Error training model {model_id}: {str(e)}")
            app.logger.error(traceback.format_exc())
    
    except Exception as e:
        app.logger.error(f"Error in training thread: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        # Update status for error
        if project_id in _training_status_by_project:
            _training_status_by_project[project_id]['status'] = 'error'
            _training_status_by_project[project_id]['message'] = f'Training failed: {str(e)}'
            _training_status_by_project[project_id]['progress'] = 0
        
        _training_status['status'] = 'error'
        _training_status['message'] = f'Training failed: {str(e)}'
        _training_status['progress'] = 0

@app.route('/api/projects/<int:project_id>/cleanup_datasets', methods=['POST'])
@login_required
def cleanup_project_datasets(project_id):
    """Clean up all dataset folders associated with a project"""
    try:
        project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
        
        # Check if the models directory exists
        models_dir = app.config['YOLO_MODELS_DIR']
        if not os.path.exists(models_dir):
            return jsonify({
                'success': True,
                'message': 'No datasets found to clean up',
                'deleted_count': 0
            })
        
        # Look for dataset folders with pattern "project_{project_id}_dataset*"
        dataset_pattern = f"project_{project_id}_dataset*"
        
        # Get all matching folders
        deleted_count = 0
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and item.startswith(f"project_{project_id}_dataset"):
                try:
                    shutil.rmtree(item_path)
                    app.logger.info(f"Deleted dataset directory: {item_path}")
                    deleted_count += 1
                except Exception as e:
                    app.logger.error(f"Error deleting dataset directory {item_path}: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully cleaned up {deleted_count} dataset directories',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        app.logger.error(f"Failed to clean up datasets for project {project_id}: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/clear_status', methods=['POST'])
@login_required
def clear_training_status():
    """Clear all training status information from memory"""
    try:
        project_id = request.json.get('project_id') if request.json else None
        
        global _training_status
        global _training_status_by_project
        
        # If project_id is provided, only clear status for that project
        if project_id:
            project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
            
            if project_id in _training_status_by_project:
                _training_status_by_project.pop(project_id, None)
                app.logger.info(f"Cleared training status for project {project_id}")
                
            # Also clear global status if it's for this project
            if _training_status.get('project_id') == project_id:
                _training_status = {
                    'status': 'idle', 
                    'message': 'No training in progress', 
                    'progress': 0
                }
                
            return jsonify({
                'success': True,
                'message': f'Training status cleared for project {project_id}'
            })
        
        # Clear all status information
        _training_status = {
            'status': 'idle', 
            'message': 'No training in progress', 
            'progress': 0
        }
        _training_status_by_project = {}
        
        app.logger.info("Cleared all training status information")
        
        return jsonify({
            'success': True,
            'message': 'All training status information cleared'
        })
        
    except Exception as e:
        app.logger.error(f"Failed to clear training status: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/projects/<int:project_id>/clean_failed_models', methods=['POST'])
@login_required
def clean_failed_models(project_id):
    """Delete all failed models for a project"""
    try:
        project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
        
        # Get all failed models for this project
        failed_models = Model.query.filter_by(project_id=project_id, status='failed').all()
        deleted_count = 0
        
        # Process each failed model
        for model in failed_models:
            try:
                # Clean up model weights if they exist
                if model.weights_path and os.path.exists(model.weights_path):
                    try:
                        os.remove(model.weights_path)
                        app.logger.info(f"Deleted model weights file: {model.weights_path}")
                    except Exception as e:
                        app.logger.error(f"Error deleting model weights file {model.weights_path}: {str(e)}")
                
                # Clean up model directory if it exists
                model_dir = os.path.dirname(model.weights_path) if model.weights_path else None
                if model_dir and os.path.exists(model_dir):
                    try:
                        shutil.rmtree(model_dir)
                        app.logger.info(f"Deleted model directory: {model_dir}")
                    except Exception as e:
                        app.logger.error(f"Error deleting model directory {model_dir}: {str(e)}")
                
                # Delete the model record from the database
                db.session.delete(model)
                deleted_count += 1
                app.logger.info(f"Deleted failed model ID {model.id}")
                
            except Exception as e:
                app.logger.error(f"Error deleting failed model {model.id}: {str(e)}")
        
        # Commit all deletions at once
        if deleted_count > 0:
            db.session.commit()
            
        return jsonify({
            'success': True,
            'message': f'Successfully deleted {deleted_count} failed models',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to clean up failed models for project {project_id}: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Inference function for deployments
@app.route('/api/deployments/inference', methods=['POST'])
@login_required
def deployment_inference():
    try:
        # Get parameters from the request
        if 'model_id' not in request.form:
            return jsonify({'error': 'No model ID provided'}), 400
            
        model_id = request.form.get('model_id')
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
        
        # Verify user has access to this model
        model = Model.query.get_or_404(model_id)
        project = Project.query.get_or_404(model.project_id)
        
        if project.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Check if model is completed and has weights
        if model.status != 'complete' or not model.weights_path:
            return jsonify({'error': 'Model is not trained or weights are missing'}), 400
            
        # Check for input source (image or video)
        input_source_type = request.form.get('input_source_type', 'Image')
        
        # Load the model
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(model.weights_path)
            app.logger.info(f"Loaded model from {model.weights_path}")
        except Exception as e:
            app.logger.error(f"Error loading model: {str(e)}")
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500
            
        # Process based on input type
        if input_source_type == "Webcam":
            return Response(
                stream_webcam(yolo_model, confidence_threshold, classes_to_detect),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
                    
        elif input_source_type == "Video":
            if 'video_file' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
                
            video_file = request.files['video_file']
            if video_file.filename == '':
                return jsonify({'error': 'No selected video'}), 400
                
            # Create a unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(video_file.filename)}"
            
            # Save the video temporarily with proper directory creation
            temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_video_path = os.path.join(temp_dir, unique_filename)
            
            try:
                app.logger.info(f"Saving uploaded video to {temp_video_path}")
                video_file.save(temp_video_path)
                
                # Verify the video file can be opened
                test_cap = cv2.VideoCapture(temp_video_path)
                if not test_cap.isOpened():
                    app.logger.error(f"Failed to open video file at {temp_video_path}")
                    test_cap.release()
                    return jsonify({'error': 'Failed to open uploaded video file'}), 500
                test_cap.release()
                
                app.logger.info(f"Successfully opened video file, streaming now...")
                return Response(
                    stream_video(yolo_model, temp_video_path, confidence_threshold, classes_to_detect),
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )
            except Exception as e:
                app.logger.error(f"Error processing uploaded video: {str(e)}")
                app.logger.error(traceback.format_exc())
                return jsonify({'error': f'Error processing video: {str(e)}'}), 500
                    
        elif input_source_type == "Image":
            if 'image_file' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
                
            image_file = request.files['image_file']
            if image_file.filename == '':
                return jsonify({'error': 'No selected image'}), 400
                
            # Read and process the image
            image_bytes = image_file.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Convert to RGB for YOLO
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run prediction with class filtering
            results = yolo_model.predict(
                source=image_rgb,
                conf=float(confidence_threshold),
                classes=classes_to_detect,
                verbose=True
            )
            
            # Store detection results for API access
            app.latest_detections = results
            
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
                    'class_name': yolo_model.names[cls],
                    'confidence': conf
                })
            
            # Add detection count text to image
            cv2.putText(
                annotated_image, 
                f"Detections: {len(detections)}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Save detection results to be displayed in the UI
            detection_results = {
                'count': len(detections),
                'items': detections,
                'model_name': model.name
            }
            
            # Save the result temporarily
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', 'result.jpg')
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            cv2.imwrite(result_path, annotated_image)
            
            # Save detection results to a JSON file for the frontend to access
            result_json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', 'result_data.json')
            with open(result_json_path, 'w') as f:
                json.dump(detection_results, f)
            
            return send_file(result_path, mimetype='image/jpeg')
            
        return jsonify({'error': 'Invalid input type'}), 400
    
    except Exception as e:
        app.logger.error(f"Error in deployment inference: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Helper functions for streaming video and webcam with detections
def stream_video(model, video_path, confidence=0.25, classes=None):
    """Stream video with YOLO detection"""
    cap = None
    try:
        app.logger.info(f"Opening video file: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            app.logger.error(f"Video file not found: {video_path}")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Error: Video file not found", (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
        
        # Check if supported file format
        _, ext = os.path.splitext(video_path)
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        if ext.lower() not in supported_formats:
            app.logger.warning(f"Potentially unsupported video format: {ext}")
        
        # Try to open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            app.logger.error(f"Could not open video file: {video_path}")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Error: Could not open video", (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        app.logger.info(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
        
        # Store the model for API access
        app.current_model = model
        
        # Counter for processed frames
        frame_counter = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                app.logger.info(f"End of video or reading error, frame {frame_counter}/{frame_count}")
                # Loop back to start of video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    app.logger.error("Could not restart video")
                    break
                frame_counter = 0
            
            frame_counter += 1
            
            try:
                # Resize large frames to avoid memory issues
                max_dimension = 1280
                if width > max_dimension or height > max_dimension:
                    scale = max_dimension / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
            
                # Convert BGR to RGB for YOLO processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with YOLO
                results = model.predict(
                    source=frame_rgb,
                    conf=confidence,
                    classes=classes,
                    verbose=False
                )
                
                # Store results for API access
                app.latest_detections = results
                
                # Draw results on frame (results[0].plot() returns an RGB image)
                annotated_frame = results[0].plot()
                
                # Convert back to BGR for OpenCV
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
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
                
                # Add frame counter
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_counter}/{frame_count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    app.logger.error("Failed to encode frame")
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Add a small delay to control streaming speed
                # Adjust this value if stream is too fast/slow
                time.sleep(1/min(fps, 30))  # Cap at 30 fps for browser performance
                
            except Exception as e:
                app.logger.error(f"Error processing frame {frame_counter}: {str(e)}")
                app.logger.error(traceback.format_exc())
                # Skip problematic frame and continue
                continue
    
    except Exception as e:
        app.logger.error(f"Error in stream_video: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        error_msg = str(e)
        # Split long error messages
        if len(error_msg) > 50:
            chunks = [error_msg[i:i+50] for i in range(0, len(error_msg), 50)]
            for i, chunk in enumerate(chunks[:3]):  # Show only first 3 lines
                cv2.putText(error_frame, chunk, (10, 240 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(error_frame, error_msg, (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    finally:
        # Always release the video capture object
        if cap is not None:
            cap.release()
            app.logger.info("Video capture released")

def stream_webcam(model, confidence=0.25, classes=None):
    """Stream webcam with YOLO detection"""
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
        
        # Store the model for API access
        app.current_model = model
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB for YOLO processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with YOLO
            results = model.predict(
                source=rgb_frame,
                conf=confidence,
                classes=classes,
                verbose=False
            )
            
            # Store results for API access
            app.latest_detections = results
            
            # Draw results on frame (results[0].plot() returns an RGB image)
            annotated_frame = results[0].plot()
            
            # Convert back to BGR for OpenCV
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
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
        app.logger.error(f"Error in stream_webcam: {str(e)}")
        app.logger.error(traceback.format_exc())
        
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

# Initialize attribute to store inference results
app.latest_detections = None

# API endpoint to get detection results
@app.route('/api/deployments/detection_results')
@login_required
def get_deployment_detection_results():
    """Get the detection results from the last inference"""
    try:
        # Check if we have detection results
        if not hasattr(app, 'latest_detections') or app.latest_detections is None:
            return jsonify({
                'success': True,
                'count': 0,
                'items': []
            })
            
        # Format the results
        formatted_results = []
        results = app.latest_detections[0]  # Get the first result (for single image)
        
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
                    app.logger.error(f"Error processing detection {i}: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'count': len(formatted_results),
            'items': formatted_results
        })
    except Exception as e:
        app.logger.error(f"Error getting detection results: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/deployments/video_feed')
@login_required
def deployment_video_feed():
    """Stream video feed with detection"""
    model_id = request.args.get('model_id')
    confidence_threshold = float(request.args.get('confidence_threshold', 0.25))
    source_type = request.args.get('source_type', 'webcam')
    video_path = request.args.get('video_path')
    selected_classes = request.args.get('selected_classes', '').split(',')
    selected_classes = [int(c) for c in selected_classes if c and c.isdigit()]  # Convert to integers and filter empty strings
    
    if not model_id:
        return "No model ID provided", 400
        
    try:
        # Verify user has access to this model
        model = Model.query.get_or_404(model_id)
        project = Project.query.get_or_404(model.project_id)
        
        if project.user_id != current_user.id:
            return "Unauthorized", 403
        
        # Load the model
        from ultralytics import YOLO
        yolo_model = YOLO(model.weights_path)
        app.current_model = yolo_model
            
        # Get video source (webcam or file)
        if source_type == 'video' and video_path:
            return Response(
                stream_video(yolo_model, video_path, confidence_threshold, selected_classes if selected_classes else None),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        else:
            return Response(
                stream_webcam(yolo_model, confidence_threshold, selected_classes if selected_classes else None),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
                        
    except Exception as e:
        app.logger.error(f"Error in video_feed: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        # Generate error frame with message
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: {str(e)}", (10, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        
        return Response(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

@app.route('/api/models/<int:model_id>/classes')
@login_required
def api_get_model_classes(model_id):
    """Get classes for a specific model"""
    try:
        # Get the model
        model = Model.query.get_or_404(model_id)
        project = Project.query.get_or_404(model.project_id)
        
        # Check if user has access to this project
        if project.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # If the model is not complete, we can't get classes
        if model.status != 'complete' or not model.weights_path:
            return jsonify({'error': 'Model is not trained or weights are missing'}), 400
            
        # Try to load the model to get class names
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(model.weights_path)
            
            # Get class names from the model
            classes = []
            for idx, name in yolo_model.names.items():
                classes.append({
                    'id': idx,
                    'name': name
                })
            
            return jsonify({
                'success': True,
                'classes': classes
            })
        except Exception as e:
            app.logger.error(f"Error loading model classes: {str(e)}")
            app.logger.error(traceback.format_exc())
            
            # Fallback to project classes if model loading fails
            project_classes = Class.query.filter_by(project_id=project.id).all()
            classes = []
            for i, cls in enumerate(project_classes):
                classes.append({
                    'id': i,
                    'name': cls.name
                })
            
            return jsonify({
                'success': True,
                'classes': classes,
                'note': 'Using project classes as fallback'
            })
        
    except Exception as e:
        app.logger.error(f"Error getting model classes: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# Add custom model upload functionality
@app.route('/api/models/upload', methods=['POST'])
@login_required
def upload_custom_model():
    """Upload a custom model file"""
    try:
        # Check if project_id is provided
        if 'project_id' not in request.form:
            return jsonify({'error': 'Project ID is required'}), 400
            
        project_id = request.form.get('project_id')
        model_name = request.form.get('model_name', 'Custom Model')
        
        # Verify user has access to this project
        project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
        
        # Check if model file was uploaded
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
            
        model_file = request.files['model_file']
        if model_file.filename == '':
            return jsonify({'error': 'No model file selected'}), 400
        
        # Check file extension
        if not model_file.filename.lower().endswith(('.pt', '.pth', '.weights', '.onnx')):
            return jsonify({'error': 'Unsupported file format. Please upload .pt, .pth, .weights, or .onnx files'}), 400
        
        # Create directory for the new model
        model_version = db.session.query(db.func.max(Model.version)).filter(Model.project_id == project_id).scalar() or 0
        model_version += 1
        model_dir = os.path.join(app.config['YOLO_MODELS_DIR'], f"project_{project_id}_model_{model_version}")
        weights_dir = os.path.join(model_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Save the model file
        filename = secure_filename(model_file.filename)
        weights_path = os.path.join(weights_dir, filename)
        model_file.save(weights_path)
        
        # Create model record in database
        model = Model(
            name=model_name,
            model_type='custom',
            status='complete',
            project_id=project.id,
            version=model_version,
            weights_path=weights_path,
            epochs=1,  # Placeholder
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Try to load the model to verify it
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(weights_path)
            
            # If successfully loaded, add metrics
            metrics = {
                'map50': 0.0,
                'map': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'note': 'Custom uploaded model (no metrics available)'
            }
            model.set_metrics(metrics)
            
        except Exception as e:
            # If loading fails, mark as 'failed'
            app.logger.error(f"Error loading custom model: {str(e)}")
            model.status = 'failed'
            
        db.session.add(model)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'model_id': model.id,
            'model_name': model.name
        })
        
    except Exception as e:
        app.logger.error(f"Error uploading custom model: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Add pretrained model selection route
@app.route('/api/models/pretrained', methods=['POST'])
@login_required
def create_pretrained_model():
    """Create a model using pretrained weights"""
    try:
        # Check if project_id is provided
        if 'project_id' not in request.form:
            return jsonify({'error': 'Project ID is required'}), 400
            
        project_id = request.form.get('project_id')
        model_name = request.form.get('model_name', 'Pretrained Model')
        model_type = request.form.get('model_type', 'yolov8n.pt')
        
        # Available pretrained models
        pretrained_models = {
            'yolov8n.pt': 'YOLOv8 Nano',
            'yolov8s.pt': 'YOLOv8 Small',
            'yolov8m.pt': 'YOLOv8 Medium',
            'yolov8l.pt': 'YOLOv8 Large',
            'yolov8x.pt': 'YOLOv8 XLarge'
        }
        
        if model_type not in pretrained_models:
            return jsonify({'error': f'Invalid model type. Available models: {", ".join(pretrained_models.keys())}'}), 400
        
        # Verify user has access to this project
        project = Project.query.filter_by(id=project_id, user_id=current_user.id).first_or_404()
        
        # Create new model version
        model_version = db.session.query(db.func.max(Model.version)).filter(Model.project_id == project_id).scalar() or 0
        model_version += 1
        
        # Create directory for the new model
        model_dir = os.path.join(app.config['YOLO_MODELS_DIR'], f"project_{project_id}_model_{model_version}")
        weights_dir = os.path.join(model_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Path to save the weights
        weights_path = os.path.join(weights_dir, model_type)
        
        # Create model record
        model = Model(
            name=model_name or f"Pretrained {pretrained_models[model_type]}",
            model_type=model_type,
            status='complete',
            project_id=project.id,
            version=model_version,
            weights_path=weights_path,
            epochs=0,
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        db.session.add(model)
        db.session.commit()
        
        # Load the pretrained model (will download if needed)
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(model_type)
            
            # Save the model to the specified path
            yolo_model.save(weights_path)
            
            # Add metrics
            metrics = {
                'map50': 0.0,
                'map': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'note': f'Pretrained {pretrained_models[model_type]} model'
            }
            model.set_metrics(metrics)
            db.session.commit()
            
        except Exception as e:
            # If loading fails, mark as 'failed'
            app.logger.error(f"Error loading pretrained model: {str(e)}")
            model.status = 'failed'
            db.session.commit()
            return jsonify({'error': f'Failed to load pretrained model: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'model_id': model.id,
            'model_name': model.name
        })
        
    except Exception as e:
        app.logger.error(f"Error creating pretrained model: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<int:project_id>/semi-supervised-annotation', methods=['POST'])
@login_required
def api_semi_supervised_annotation(project_id):
    """API endpoint to trigger semi-supervised annotation for a project.
    
    This will use the semi-supervised learning module to:
    1. Train a model on existing manually labeled images
    2. Use pseudo-labeling to annotate unlabeled images
    3. Retrain the model for multiple iterations
    """
    project = Project.query.get_or_404(project_id)
    
    # Check if user has permission to access this project
    if project.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'You do not have permission to access this project'}), 403
    
    # Get parameters from request
    data = request.get_json() or {}
    confidence_threshold = data.get('confidence_threshold', 0.5)
    pseudo_label_threshold = data.get('pseudo_label_threshold', 0.8)
    
    try:
        # Start the semi-supervised workflow in a background task
        @copy_current_request_context
        def run_annotation_task():
            try:
                results = auto_annotate_project(
                    project_id=project_id,
                    confidence_threshold=confidence_threshold,
                    pseudo_label_threshold=pseudo_label_threshold
                )
                app.logger.info(f"Semi-supervised annotation complete for project {project_id}: {results}")
            except Exception as e:
                app.logger.error(f"Error in semi-supervised annotation for project {project_id}: {str(e)}")
        
        thread = threading.Thread(target=run_annotation_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success', 
            'message': 'Semi-supervised annotation process started'
        })
    except Exception as e:
        app.logger.error(f"Error starting semi-supervised annotation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add a dedicated route for video upload and streaming
@app.route('/api/video/upload', methods=['POST'])
@login_required
def video_upload():
    """Upload a video file and prepare it for streaming"""
    try:
        # Check if video file was provided
        if 'video_file' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        video_file = request.files['video_file']
        if video_file.filename == '':
            return jsonify({'error': 'No selected video'}), 400
            
        # Create a unique filename
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(video_file.filename)}"
        
        # Save the video
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, unique_filename)
        
        app.logger.info(f"Saving uploaded video to {temp_video_path}")
        video_file.save(temp_video_path)
        
        # Verify the video file can be opened
        test_cap = cv2.VideoCapture(temp_video_path)
        if not test_cap.isOpened():
            test_cap.release()
            return jsonify({'error': 'Unable to open the uploaded video file'}), 400
            
        # Get basic video info
        width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = test_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        test_cap.release()
        
        # Create a video session ID for streaming
        video_id = unique_filename
        
        # Store video info in session for easy retrieval
        if not hasattr(app, 'video_sessions'):
            app.video_sessions = {}
            
        app.video_sessions[video_id] = {
            'path': temp_video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'uploaded_at': datetime.now().isoformat(),
            'user_id': current_user.id
        }
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'info': {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': frame_count / fps if fps > 0 else 0
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error handling video upload: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/stream/<video_id>')
@login_required
def video_stream(video_id):
    """Stream a previously uploaded video"""
    # Check if video exists in sessions
    if not hasattr(app, 'video_sessions') or video_id not in app.video_sessions:
        return jsonify({'error': 'Video not found or expired'}), 404
        
    # Check if user has access to this video
    video_session = app.video_sessions[video_id]
    if video_session['user_id'] != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get video path
    video_path = video_session['path']
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
        
    # Create a basic video streamer without detection
    def generate_frames():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
            
        # Get FPS for timing
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS if not available
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                      
                # Control frame rate
                time.sleep(1/min(fps, 30))
                
        finally:
            cap.release()
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=True)