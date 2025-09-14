import os
import re

def fix_app_py():
    # Read the file
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Find duplicate routes and remove them
    routes_to_fix = [
        {
            'pattern': r'@app\.route\(\'/projects/\<int:project_id\>/delete\', methods=\[\'POST\'\]\)\n@login_required\ndef delete_project_v2\(project_id\):.*?return redirect\(url_for\(\'projects_page\'\)\)',
            'name': 'delete_project'
        },
        {
            'pattern': r'@app\.route\(\'/models/\<int:model_id\>/download/\<format\>\'\)\n@login_required\ndef download_model\(model_id, format\):.*?return send_file',
            'name': 'download_model'
        },
        {
            'pattern': r'@app\.route\(\'/api/models/\<int:model_id\>\', methods=\[\'DELETE\'\]\)\n@login_required\ndef delete_model\(model_id\):.*?Exception as e\)',
            'name': 'delete_model'
        }
    ]
    
    changes_made = False
    
    for route in routes_to_fix:
        matches = list(re.finditer(route['pattern'], content, re.DOTALL))
        if len(matches) >= 2:
            # Keep the first occurrence, remove all others
            for match in reversed(matches[1:]):
                start_idx = match.start()
                end_idx = match.end()
                content = content[:start_idx] + content[end_idx:]
                changes_made = True
            print(f"Removed {len(matches) - 1} duplicate(s) of {route['name']}")
    
    if not changes_made:
        print("No duplicate routes found to fix")
        return False
    
    # Write back to the file
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("Successfully fixed app.py by removing duplicate routes")
    return True

if __name__ == "__main__":
    fix_app_py() 