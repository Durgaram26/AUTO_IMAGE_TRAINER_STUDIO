/**
 * AI Panel Visibility Fix
 * This script ensures the AI labeling panel is visible and functional
 */
(function() {
    // Don't run immediately, wait for annotation modal to be visible first
    const checkAndFixAnnotationModal = function() {
        // Check if we're in the annotation mode
        const annotationModal = document.getElementById('annotation-modal');
        if (!annotationModal || annotationModal.style.display !== 'flex') {
            // Not in annotation mode yet, try again later
            setTimeout(checkAndFixAnnotationModal, 1000);
            return;
        }
        
        console.log('Annotation modal is open, initializing AI panel fix');
        
        // Find AI panel elements
        const aiLabelingPanel = document.getElementById('ai-labeling-panel');
        const aiLabelToggle = document.getElementById('ai-label-toggle');
        const detectObjectsBtn = document.getElementById('run-canvas-ai');
        
        // Create a helper button
        const fixButton = document.createElement('button');
        fixButton.id = 'ai-panel-fix-btn';
        fixButton.innerHTML = '<i class="fas fa-magic"></i> AI';
        fixButton.style.cssText = `
            position: fixed;
            top: 15px;
            right: 15px;
            background-color: #3a86ff;
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 10000;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        `;
        
        // Add the button to the document
        document.body.appendChild(fixButton);
        
        // Add click event listener
        fixButton.addEventListener('click', function() {
            console.log('AI Fix button clicked');
            
            // Force the AI panel to be visible with correct styling
            if (aiLabelingPanel) {
                // Reset styling
                aiLabelingPanel.style.position = 'fixed';
                aiLabelingPanel.style.top = '80px';
                aiLabelingPanel.style.right = '20px';
                aiLabelingPanel.style.zIndex = '9999';
                aiLabelingPanel.style.display = 'block';
                aiLabelingPanel.style.visibility = 'visible';
                aiLabelingPanel.style.opacity = '1';
                
                // Add important class for highlighting
                aiLabelingPanel.classList.add('ai-panel-visible');
                
                console.log('AI panel forced visible');
            } else {
                console.error('AI panel element not found');
            }
            
            // Also activate the original toggle if it exists
            if (aiLabelToggle) {
                aiLabelToggle.classList.add('active');
            }
        });
        
        // Find all buttons that might be related to object detection and hook them up
        const hookupButtons = () => {
            // All buttons with text containing 'detect' or having specific IDs
            const buttons = [
                ...Array.from(document.querySelectorAll('button')),
                ...Array.from(document.querySelectorAll('a.btn')), 
                ...Array.from(document.querySelectorAll('.control-btn'))
            ];
            
            const detectButtons = buttons.filter(btn => {
                const text = (btn.textContent || '').toLowerCase();
                const id = (btn.id || '').toLowerCase();
                return text.includes('detect') || 
                       text.includes('ai') || 
                       id.includes('detect') || 
                       id.includes('ai') ||
                       btn === detectObjectsBtn;
            });
            
            // Add listeners to all detection-related buttons
            detectButtons.forEach(btn => {
                console.log('Adding AI panel show listener to button:', btn.textContent || btn.id);
                btn.addEventListener('click', function(e) {
                    // Don't prevent default or stop propagation - let the normal action happen too
                    console.log('Detect-related button clicked');
                    
                    // Show the panel with delay to ensure it happens after other handlers
                    setTimeout(() => {
                        if (aiLabelingPanel) {
                            aiLabelingPanel.style.display = 'block';
                            aiLabelingPanel.style.visibility = 'visible';
                            aiLabelingPanel.style.opacity = '1';
                            console.log('AI panel shown via detect button');
                        }
                    }, 100);
                });
            });
        };
        
        // Run button hookup initially
        hookupButtons();
        
        // And again after a delay to catch any dynamically added elements
        setTimeout(hookupButtons, 2000);
        
        // Force panel visibility after delays
        [1000, 2000, 3000].forEach(delay => {
            setTimeout(function() {
                if (aiLabelingPanel) {
                    aiLabelingPanel.style.display = 'block';
                    aiLabelingPanel.style.visibility = 'visible';
                    console.log(`Panel visibility forced after ${delay}ms`);
                }
            }, delay);
        });
    };
    
    // Also listen for image cards being clicked to ensure annotation modal opens
    const setupImageClickHandlers = function() {
        // Find all image cards
        const imageCards = document.querySelectorAll('.image-card');
        if (imageCards && imageCards.length > 0) {
            console.log(`Found ${imageCards.length} image cards, ensuring click handlers work`);
            
            // Ensure image cards are clickable
            imageCards.forEach(card => {
                card.style.pointerEvents = 'auto';
                card.style.cursor = 'pointer';
            });
        }
        
        // Also check for annotate buttons
        const annotateButtons = document.querySelectorAll('.annotate-btn, .image-action-btn');
        if (annotateButtons && annotateButtons.length > 0) {
            console.log(`Found ${annotateButtons.length} annotate buttons`);
            
            // Ensure buttons are clickable
            annotateButtons.forEach(btn => {
                btn.style.pointerEvents = 'auto';
                btn.style.cursor = 'pointer';
            });
        }
    };
    
    // Start by ensuring image click handlers work
    setTimeout(setupImageClickHandlers, 1000);
    
    // Set up a mutation observer to detect when the annotation modal appears
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && 
                mutation.attributeName === 'style' && 
                mutation.target.id === 'annotation-modal') {
                
                const modal = mutation.target;
                if (modal.style.display === 'flex') {
                    console.log('Annotation modal opened, initializing AI panel fix');
                    checkAndFixAnnotationModal();
                }
            }
        });
    });
    
    // Start observing the document for the annotation modal
    const annotationModal = document.getElementById('annotation-modal');
    if (annotationModal) {
        observer.observe(annotationModal, { attributes: true });
        console.log('Observing annotation modal for changes');
    } else {
        // If modal doesn't exist yet, check periodically
        setTimeout(function checkForModal() {
            const modal = document.getElementById('annotation-modal');
            if (modal) {
                observer.observe(modal, { attributes: true });
                console.log('Found and observing annotation modal');
            } else {
                setTimeout(checkForModal, 1000);
            }
        }, 1000);
    }
    
    // If the modal is already open, initialize right away
    if (annotationModal && annotationModal.style.display === 'flex') {
        checkAndFixAnnotationModal();
    }
})(); 