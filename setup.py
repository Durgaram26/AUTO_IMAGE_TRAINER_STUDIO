#!/usr/bin/env python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash

# Load environment variables
load_dotenv()

# Import models
from models import db, User, Project, Class

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///griffonder.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

def setup_database():
    """Set up the database, create tables, and add sample data"""
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check if sample data already exists
        if User.query.filter_by(username='admin').first():
            print("Sample data already exists")
            return
        
        # Create admin user
        admin = User(
            username='admin',
            email='admin@example.com',
            name='Admin User',
            password_hash=generate_password_hash('password')
        )
        db.session.add(admin)
        db.session.commit()
        
        # Create a sample project
        project = Project(
            name='Hard Hat Detection',
            type='Object Detection',
            visibility='Private',
            annotation_group='safety',
            user_id=admin.id
        )
        db.session.add(project)
        db.session.commit()
        
        # Add classes
        classes = [
            Class(name='helmet', color='#6c5ce7', project_id=project.id),
            Class(name='head', color='#2ecc71', project_id=project.id),
            Class(name='person', color='#e74c3c', project_id=project.id)
        ]
        db.session.add_all(classes)
        db.session.commit()
        
        print("Database initialized with sample data")
        print(f"Username: admin")
        print(f"Password: password")

def create_directories():
    """Create necessary directories for the application"""
    upload_folder = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    models_dir = os.getenv('YOLO_MODELS_DIR', 'static/models')
    
    for directory in [upload_folder, models_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == '__main__':
    create_directories()
    setup_database()
    print("Setup complete") 