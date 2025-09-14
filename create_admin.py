from app import app, db
from models import User

with app.app_context():
    # Check if admin user exists
    admin = User.query.filter_by(username='admin').first()
    
    if admin:
        # Update admin password if user exists
        admin.set_password('password')
        print("Admin password has been reset to 'password'")
    else:
        # Create new admin user
        admin = User(username='admin', email='admin@example.com', name='Admin User')
        admin.set_password('password')
        db.session.add(admin)
        print("Admin user created with username 'admin' and password 'password'")
    
    # Commit changes
    db.session.commit()
    print("Changes committed to database") 