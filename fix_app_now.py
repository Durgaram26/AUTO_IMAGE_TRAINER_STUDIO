with open('app.py', 'r') as f:
    content = f.read()

# Replace the broken part with the fixed code
broken_part = '''    print('Database seeding completed.')


(
        file_path,
        mimetype=content_type,
        as_attachment=True,
        download_name=filename
    )'''

fixed_part = '''    print('Database seeding completed.')


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
    )'''

new_content = content.replace(broken_part, fixed_part)

with open('app.py', 'w') as f:
    f.write(new_content)

print("Fixed app.py file.") 