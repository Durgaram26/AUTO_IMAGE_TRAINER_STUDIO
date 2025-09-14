/**
 * Annotation Canvas Fix
 * This script ensures the annotation canvas is properly initialized and visible
 */
(function() {
    console.log('Annotation Canvas Fix script loaded');

    // Function to fix the annotation canvas
    function fixAnnotationCanvas() {
        console.log('Attempting to fix annotation canvas');
        
        // Find the annotation modal
        const annotationModal = document.getElementById('annotation-modal');
        if (!annotationModal) {
            console.error('Annotation modal not found in the DOM');
            return;
        }
        
        // Check if we should force the modal to be visible based on localStorage
        const shouldFixModal = localStorage.getItem('fix_annotation_modal') === 'true';
        if (shouldFixModal) {
            console.log('Forcing annotation modal to be visible based on localStorage setting');
        }
        
        // Ensure the modal is visible
        annotationModal.style.display = 'flex';
        
        // Find and fix the canvas element
        const canvas = document.querySelector('#annotation-canvas');
        const shouldFixCanvas = localStorage.getItem('fix_annotation_canvas') === 'true';
        
        if (canvas) {
            console.log('Canvas found, ensuring proper initialization');
            // Make sure the canvas is visible
            canvas.style.display = 'block';
            canvas.style.visibility = 'visible';
            
            // Try to force canvas redraw
            const context = canvas.getContext('2d');
            if (context) {
                // Store current transform
                context.save();
                
                // Clear canvas
                context.setTransform(1, 0, 0, 1, 0, 0);
                context.clearRect(0, 0, canvas.width, canvas.height);
                
                // Restore transform
                context.restore();
                
                console.log('Canvas context reset and cleared');
            }
        } else {
            console.error('Annotation canvas not found in the DOM');
            
            if (shouldFixCanvas) {
                console.log('Will attempt to create canvas element based on localStorage setting');
            }
            
            // Try to find a container where we could create the canvas
            const canvasContainer = document.querySelector('.annotation-content') || 
                                  document.querySelector('.canvas-container') || 
                                  document.querySelector('.annotation-canvas-container');
                                  
            if (canvasContainer) {
                console.log('Canvas container found, creating canvas element');
                const newCanvas = document.createElement('canvas');
                newCanvas.id = 'annotation-canvas';
                newCanvas.width = 800;
                newCanvas.height = 600;
                newCanvas.style.display = 'block';
                canvasContainer.appendChild(newCanvas);
                console.log('Canvas element created');
            }
        }
        
        // Look for canvas container elements that might be hidden
        const canvasContainers = [
            document.querySelector('.canvas-container'),
            document.querySelector('.annotation-canvas-container'),
            document.querySelector('.annotation-content'),
            document.querySelector('.annotation-editor')
        ];
        
        canvasContainers.forEach(container => {
            if (container) {
                container.style.display = 'block';
                container.style.visibility = 'visible';
                console.log('Fixed container visibility:', container);
            }
        });
        
        // Fix AI labeling panel
        const aiLabelingPanel = document.getElementById('ai-labeling-panel');
        const shouldFixAiPanel = localStorage.getItem('fix_ai_panel') === 'true';
        
        if (aiLabelingPanel) {
            console.log('Found AI labeling panel, ensuring visibility');
            
            if (shouldFixAiPanel) {
                console.log('Forcing AI panel visibility based on localStorage setting');
            }
            
            aiLabelingPanel.style.display = 'block';
            aiLabelingPanel.style.visibility = 'visible';
            aiLabelingPanel.style.position = 'fixed';
            aiLabelingPanel.style.top = '80px';
            aiLabelingPanel.style.right = '20px';
            aiLabelingPanel.style.zIndex = '9999';
        }
        
        // Check for fabric.js initialization issues
        if (window.fabric && canvas && !canvas.__fabric__) {
            try {
                console.log('Attempting to initialize fabric.js canvas');
                new fabric.Canvas('annotation-canvas');
            } catch (err) {
                console.error('Error initializing fabric.js canvas:', err);
            }
        }
    }
    
    // Add a helper button for users to manually trigger the fix
    function addFixButton() {
        // Remove existing button if it exists
        const existingButton = document.getElementById('canvas-fix-btn');
        if (existingButton) {
            existingButton.remove();
        }
        
        const fixButton = document.createElement('button');
        fixButton.id = 'canvas-fix-btn';
        fixButton.innerHTML = '<i class="fas fa-brush"></i> Fix Canvas';
        fixButton.style.cssText = `
            position: fixed;
            top: 15px;
            right: 70px;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 14px;
            cursor: pointer;
            z-index: 10001;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        `;
        
        fixButton.addEventListener('click', function() {
            console.log('Fix Canvas button clicked');
            fixAnnotationCanvas();
        });
        
        document.body.appendChild(fixButton);
        console.log('Added canvas fix button to the page');
        
        // Also add a link to the fix helper page
        const fixHelperLink = document.createElement('a');
        fixHelperLink.id = 'fix-helper-link';
        fixHelperLink.innerHTML = '<i class="fas fa-tools"></i> Annotation Fixer';
        fixHelperLink.href = '/annotation-fix';
        fixHelperLink.target = '_blank';
        fixHelperLink.style.cssText = `
            position: fixed;
            top: 15px;
            right: 180px;
            background-color: #6c5ce7;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 14px;
            cursor: pointer;
            z-index: 10001;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            text-decoration: none;
        `;
        
        // Add current project ID to the link if we can find it
        const projectIdMatch = window.location.pathname.match(/\/projects\/(\d+)\/annotate/);
        if (projectIdMatch && projectIdMatch[1]) {
            fixHelperLink.href += '?project_id=' + projectIdMatch[1];
        }
        
        document.body.appendChild(fixHelperLink);
        console.log('Added fix helper link to the page');
    }
    
    // Add alternate fallback buttons if needed
    function addFallbackButtons() {
        // Add AI panel button if it doesn't exist
        if (!document.getElementById('fallback-ai-button')) {
            const aiButton = document.createElement('div');
            aiButton.id = 'fallback-ai-button';
            aiButton.innerHTML = '<i class="fas fa-robot"></i>';
            aiButton.style.cssText = `
                position: fixed; 
                top: 10px; 
                right: 120px; 
                background-color: #3a86ff; 
                color: white; 
                width: 40px; 
                height: 40px; 
                border-radius: 50%; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                cursor: pointer; 
                z-index: 10000; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            `;
            
            document.body.appendChild(aiButton);
            
            // Add click handler to toggle AI panel
            aiButton.addEventListener('click', function() {
                const aiLabelingPanel = document.getElementById('ai-labeling-panel');
                if (aiLabelingPanel) {
                    if (aiLabelingPanel.style.display === 'none' || aiLabelingPanel.style.display === '') {
                        aiLabelingPanel.style.display = 'block';
                        console.log('Showing AI panel via fallback button');
                    } else {
                        aiLabelingPanel.style.display = 'none';
                        console.log('Hiding AI panel via fallback button');
                    }
                }
            });
            
            console.log('Added fallback AI button to the page');
        }
        
        // Add canvas fix circle button if it doesn't exist
        if (!document.getElementById('canvas-fix-circle')) {
            const canvasButton = document.createElement('div');
            canvasButton.id = 'canvas-fix-circle';
            canvasButton.innerHTML = '<i class="fas fa-brush"></i>';
            canvasButton.style.cssText = `
                position: fixed; 
                top: 10px; 
                right: 170px; 
                background-color: #e74c3c; 
                color: white; 
                width: 40px; 
                height: 40px; 
                border-radius: 50%; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                cursor: pointer; 
                z-index: 10000; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            `;
            
            document.body.appendChild(canvasButton);
            
            // Add click handler to fix canvas
            canvasButton.addEventListener('click', fixAnnotationCanvas);
            
            console.log('Added canvas fix circle button to the page');
        }
    }
    
    // Fix image card click handlers
    function fixImageClickHandlers() {
        const imageCards = document.querySelectorAll('.image-card');
        console.log(`Found ${imageCards.length} image cards`);
        
        // Fix image card styling to ensure clickability
        imageCards.forEach(card => {
            card.style.cursor = 'pointer';
            card.style.pointerEvents = 'auto';
            
            // Remove existing click listeners to avoid duplicates
            const newCard = card.cloneNode(true);
            card.parentNode.replaceChild(newCard, card);
            
            // Add click handler to show annotation modal
            newCard.addEventListener('click', function(e) {
                console.log('Image card clicked');
                const annotationModal = document.getElementById('annotation-modal');
                if (annotationModal) {
                    annotationModal.style.display = 'flex';
                    
                    // Also try to fix the canvas after a short delay
                    setTimeout(fixAnnotationCanvas, 300);
                }
            });
        });
    }
    
    // Listen for localStorage changes
    window.addEventListener('storage', function(e) {
        if (e.key && (e.key.startsWith('fix_annotation') || e.key.startsWith('fix_ai'))) {
            console.log(`LocalStorage change detected: ${e.key}=${e.newValue}`);
            setTimeout(fixAnnotationCanvas, 300);
        }
    });
    
    // Initialize the fix when DOM is loaded
    function initialize() {
        console.log('Initializing annotation canvas fix');
        
        // Wait a bit for the page to fully load
        setTimeout(() => {
            // Add the fix buttons
            addFixButton();
            addFallbackButtons();
            
            // Fix image click handlers
            fixImageClickHandlers();
            
            // Try to fix the canvas automatically
            fixAnnotationCanvas();
            
            // Set additional timeouts to try again in case the page is still loading
            setTimeout(fixAnnotationCanvas, 1000);
            setTimeout(fixAnnotationCanvas, 2000);
            setTimeout(fixAnnotationCanvas, 3000);
        }, 500);
    }
    
    // Run when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
    
    // Also listen for page visibility changes
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            console.log('Page became visible, fixing canvas');
            setTimeout(fixAnnotationCanvas, 300);
        }
    });
})(); 