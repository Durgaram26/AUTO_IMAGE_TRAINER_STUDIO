# Griffonder - Computer Vision Platform

A Roboflow-like computer vision platform built with Flask, SQLAlchemy, and Ultralytics YOLOv8.

## Features

- User authentication and account management
- Project creation with various model types (object detection, classification, segmentation)
- Image upload with batch processing and tagging
- Annotation tools for different annotation types (bounding boxes, polygons, keypoints)
- YOLOv8 model training integration
- Dataset analytics and visualization
- Export annotations in various formats (COCO, YOLO)
- Model inference API

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/griffonder.git
cd griffonder
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create an environment file:
```bash
# Create a .env file with the following content
SECRET_KEY=your-secret-key-here
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
DATABASE_URI=sqlite:///griffonder.db
UPLOAD_FOLDER=static/uploads
YOLO_MODELS_DIR=static/models
```

5. Initialize the database and create sample data:
```bash
python setup.py
```

6. Run the application:
```bash
flask run
# or
python app.py
```

7. Open your browser and go to `http://localhost:5000`

8. Login with the created admin account:
   - Username: `admin`
   - Password: `password`

## Using the Platform

1. **Create a Project**: After logging in, click on "Create a Project" to start a new computer vision project.

2. **Upload Images**: Navigate to your project and use the Upload page to add images to your dataset.

3. **Annotate Images**: Use the Annotation interface to label your images with bounding boxes or other annotation types.

4. **Train Models**: Go to the Train tab to set up and start training a YOLOv8 model on your annotated data.

5. **Analyze Dataset**: Use the Analytics page to understand your dataset composition and class distribution.

## Database Migrations

If you need to modify the database schema:

```bash
# Initialize migrations (first time only)
flask db init

# Create a migration after model changes
flask db migrate -m "Description of changes"

# Apply migrations
flask db upgrade
```

## Project Structure

```
griffonder/
├── app.py                 # Main Flask application
├── models.py              # SQLAlchemy database models
├── utils.py               # Utility functions for YOLOv8 integration
├── setup.py               # Database initialization script
├── requirements.txt       # Python dependencies
├── env.example            # Example environment variables
├── static/                # Static files
│   ├── css/               # CSS stylesheets
│   ├── js/                # JavaScript files
│   ├── images/            # Images
│   ├── uploads/           # User uploaded images
│   └── models/            # Trained model files
└── templates/             # HTML templates
    ├── base.html          # Base template
    ├── login.html         # Login page
    ├── register.html      # Registration page
    ├── projects.html      # Projects listing page
    ├── annotate.html      # Annotation interface
    ├── analytics.html     # Dataset analytics
    ├── train.html         # Model training interface
    └── ...                # Other templates
```

## License

This project is open source and available under the [MIT License](LICENSE). 
``` 