import os
import sys
import logging
from semisupervised import auto_annotate_project

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Demo script for running semi-supervised learning workflow manually.
    
    Usage:
        python demo_semisupervised.py <project_id>
        
    This will:
    1. Train a model on existing manually labeled images
    2. Use pseudo-labeling to auto-annotate unlabeled images
    3. Retrain the model with the combined data
    4. Repeat for a few iterations
    
    Requirements:
    - Some manually labeled images in the project
    - Classes defined in the project
    """
    if len(sys.argv) < 2:
        print("Usage: python demo_semisupervised.py <project_id>")
        return
    
    try:
        project_id = int(sys.argv[1])
        
        print(f"\n======= Semi-Supervised Learning Demo =======")
        print(f"Starting semi-supervised learning for project {project_id}")
        print(f"This will use manually labeled images to train a model,")
        print(f"then auto-annotate the remaining unlabeled images.")
        print(f"==============================================\n")
        
        # Set confidence thresholds
        confidence_threshold = 0.5  # Standard detection threshold
        pseudo_label_threshold = 0.8  # Higher threshold for pseudo-labeling
        
        # Run the semi-supervised workflow
        results = auto_annotate_project(
            project_id=project_id,
            confidence_threshold=confidence_threshold,
            pseudo_label_threshold=pseudo_label_threshold
        )
        
        # Display results
        print("\n======= Results =======")
        print(f"Total images pseudo-labeled: {results['total_pseudo_labeled']}")
        print(f"Final model path: {results['final_model_path']}")
        print("\nIteration details:")
        
        for iteration in results['iterations']:
            print(f"  Iteration {iteration['iteration']}: " 
                  f"Labeled {iteration['images_pseudo_labeled']} images. "
                  f"Total labeled: {iteration['total_labeled_images']}")
        
        print("\nDone! The model has been trained and unlabeled images have been annotated.")
        print("You can now review the auto-annotations in the UI.")
        
    except Exception as e:
        print(f"Error running semi-supervised learning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 