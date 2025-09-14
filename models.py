from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy.sql import func
import json
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120))
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=func.now())
    
    # Relationships
    projects = db.relationship('Project', backref='owner', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # object-detection, classification, segmentation, etc.
    visibility = db.Column(db.String(20), default='Private')  # Private or Public
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=func.now())
    updated_at = db.Column(db.DateTime, default=func.now(), onupdate=func.now())
    license = db.Column(db.String(50))
    annotation_group = db.Column(db.String(100))
    
    # Foreign Keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    images = db.relationship('Image', backref='project', lazy=True, cascade="all, delete-orphan")
    models = db.relationship('Model', backref='project', lazy=True, cascade="all, delete-orphan")
    classes = db.relationship('Class', backref='project', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Project {self.name}>'
    
    def image_count(self):
        return len(self.images)
    
    def model_count(self):
        return len(self.models)
    
    def annotation_count(self):
        count = 0
        for image in self.images:
            count += len(image.annotations)
        return count
    
    def avg_annotations_per_image(self):
        if not self.images:
            return 0
        return self.annotation_count() / len(self.images)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    format = db.Column(db.String(10))  # jpg, png, etc.
    filesize = db.Column(db.Integer)  # in bytes
    uploaded_at = db.Column(db.DateTime, default=func.now())
    batch_name = db.Column(db.String(100))
    split = db.Column(db.String(20), default='train')  # train, valid, test
    is_annotated = db.Column(db.Boolean, default=False)  # Flag to indicate if the image has annotations
    auto_annotated = db.Column(db.Boolean, default=False)  # Flag to indicate if annotated automatically
    
    # Foreign Keys
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    
    # Relationships
    annotations = db.relationship('Annotation', backref='image', lazy=True, cascade="all, delete-orphan")
    tags = db.relationship('ImageTag', backref='image', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Image {self.filename}>'

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    x_min = db.Column(db.Float)  # For bounding boxes
    y_min = db.Column(db.Float)
    x_max = db.Column(db.Float)
    y_max = db.Column(db.Float)
    polygon_points = db.Column(db.Text)  # JSON string for segmentation masks
    keypoints = db.Column(db.Text)  # JSON string for keypoint annotations
    label = db.Column(db.Text)  # For multimodal annotations (text)
    confidence = db.Column(db.Float, nullable=True)  # Confidence score for model predictions/pseudo-labels
    created_at = db.Column(db.DateTime, default=func.now())
    updated_at = db.Column(db.DateTime, default=func.now(), onupdate=func.now())
    
    # Foreign Keys
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    
    def __repr__(self):
        return f'<Annotation {self.id}>'
    
    def get_polygon_points(self):
        if self.polygon_points:
            return json.loads(self.polygon_points)
        return []
    
    def set_polygon_points(self, points):
        self.polygon_points = json.dumps(points)
    
    def get_keypoints(self):
        if self.keypoints:
            return json.loads(self.keypoints)
        return []
    
    def set_keypoints(self, points):
        self.keypoints = json.dumps(points)

class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    color = db.Column(db.String(20))  # RGB or HEX code
    created_at = db.Column(db.DateTime, default=func.now())
    
    # Foreign Keys
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    
    # Relationships
    annotations = db.relationship('Annotation', backref='class', lazy=True)
    
    def __repr__(self):
        return f'<Class {self.name}>'

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.Integer, default=1)
    model_type = db.Column(db.String(50))  # yolov8n, yolov8s, etc.
    status = db.Column(db.String(20), default='created')  # created, training, completed, failed
    metrics = db.Column(db.Text)  # JSON string with metrics
    weights_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=func.now())
    completed_at = db.Column(db.DateTime)
    
    # Training progress
    current_epoch = db.Column(db.Integer, default=0)
    
    # Settings
    epochs = db.Column(db.Integer, default=100)
    batch_size = db.Column(db.Integer, default=16)
    image_size = db.Column(db.Integer, default=640)
    
    # Foreign Keys
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    
    def __repr__(self):
        return f'<Model {self.name} v{self.version}>'
    
    def get_metrics(self):
        if self.metrics:
            return json.loads(self.metrics)
        return {}
    
    def set_metrics(self, metrics_dict):
        self.metrics = json.dumps(metrics_dict)

class ImageTag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    
    # Foreign Keys
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    
    def __repr__(self):
        return f'<Tag {self.name}>' 