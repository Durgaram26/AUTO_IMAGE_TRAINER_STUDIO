/**
 * Model Training JS - Handles real-time updates for YOLOv8 model training
 */

document.addEventListener('DOMContentLoaded', function() {
    // Training status polling
    const projectId = document.getElementById('project-id').value;
    const refreshStatusBtn = document.getElementById('refresh-status');
    const autoRefreshCheckbox = document.getElementById('auto-refresh');
    let refreshInterval = null;
    
    // Function to update the UI with training status
    function updateTrainingStatus(data) {
        const noActiveTraining = document.getElementById('no-active-training');
        const activeTraining = document.getElementById('active-training');
        
        if (data.status === 'none' || data.status === 'idle') {
            // No active training
            if (noActiveTraining) {
                noActiveTraining.style.display = 'flex';
            } else {
                const container = document.getElementById('training-status-container');
                container.innerHTML = `
                    <div class="no-active-training" id="no-active-training">
                        <i class="fas fa-info-circle"></i>
                        <p>No models are currently training. Start a new training job below.</p>
                    </div>
                `;
            }
            
            if (activeTraining) {
                activeTraining.style.display = 'none';
            }
            
            return;
        }
        
        // Active training exists
        if (noActiveTraining) {
            noActiveTraining.style.display = 'none';
        }
        
        if (!activeTraining) {
            // Create active training card if it doesn't exist
            const container = document.getElementById('training-status-container');
            container.innerHTML = `
                <div class="active-model-card" id="active-training">
                    <div class="model-info">
                        <div class="model-name" id="training-model-name">Training Job</div>
                        <div class="model-type" id="training-model-type">-</div>
                    </div>
                    <div class="training-details">
                        <div class="progress-info">
                            <div class="progress-header">
                                <span>Status: <span id="training-status-text">${data.status}</span></span>
                                <span class="epoch-count" id="epoch-count">0/0</span>
                            </div>
                            <div class="progress">
                                <div class="progress-bar" id="progress-bar" role="progressbar" 
                                     style="width: 0%" 
                                     aria-valuenow="0" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">0%</div>
                            </div>
                            <div class="status-message" id="status-message"></div>
                        </div>
                        <div class="metrics-container" id="metrics-container">
                            <!-- Metrics will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Update training card details
        document.getElementById('training-status-text').textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
        
        if (data.current_epoch !== undefined && data.total_epochs !== undefined) {
            document.getElementById('epoch-count').textContent = `${data.current_epoch}/${data.total_epochs}`;
        }
        
        const progressBar = document.getElementById('progress-bar');
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
            progressBar.setAttribute('aria-valuenow', data.progress);
            progressBar.textContent = `${data.progress}%`;
        }
        
        if (data.message) {
            document.getElementById('status-message').textContent = data.message;
        }
        
        // Update metrics if available
        if (data.metrics) {
            const metricsContainer = document.getElementById('metrics-container');
            metricsContainer.innerHTML = '';
            
            for (const [key, value] of Object.entries(data.metrics)) {
                const metricEl = document.createElement('div');
                metricEl.className = 'metric-item';
                metricEl.innerHTML = `<strong>${key}:</strong> ${value.toFixed(4)}`;
                metricsContainer.appendChild(metricEl);
            }
        }
        
        // If training completed, refresh the page to show updated models
        if (data.status === 'complete') {
            setTimeout(() => {
                window.location.reload();
            }, 3000);
        }
    }
    
    // Function to fetch training status
    function fetchTrainingStatus() {
        fetch(`/api/projects/${projectId}/training_status`)
            .then(response => response.json())
            .then(data => {
                updateTrainingStatus(data);
            })
            .catch(error => {
                console.error('Error fetching training status:', error);
            });
    }
    
    // Set up auto-refresh
    function setupAutoRefresh() {
        if (autoRefreshCheckbox.checked) {
            refreshInterval = setInterval(fetchTrainingStatus, 3000);
        } else {
            clearInterval(refreshInterval);
        }
    }
    
    // Initialize with first fetch
    fetchTrainingStatus();
    setupAutoRefresh();
    
    // Event listeners
    refreshStatusBtn.addEventListener('click', fetchTrainingStatus);
    autoRefreshCheckbox.addEventListener('change', setupAutoRefresh);

    // Find all models in training
    const trainingModels = document.querySelectorAll('.model-card[data-status="training"]');
    
    // If there are models in training, start updating their status
    if (trainingModels.length > 0) {
        // Start polling for updates
        trainingModels.forEach(modelCard => {
            const modelId = modelCard.dataset.modelId;
            if (modelId) {
                updateModelStatus(modelId);
            }
        });
    }
    
    // Also check for models with 'creating' status
    const creatingModels = document.querySelectorAll('.model-card[data-status="creating"]');
    if (creatingModels.length > 0) {
        creatingModels.forEach(modelCard => {
            const modelId = modelCard.dataset.modelId;
            if (modelId) {
                // Start checking status for models that are being created
                // with a slight delay to allow server processing
                setTimeout(() => updateModelStatus(modelId), 3000);
            }
        });
    }
    
    // Add event listener for training form submission
    const trainingForm = document.getElementById('training-form');
    if (trainingForm) {
        trainingForm.addEventListener('submit', function() {
            const submitButton = trainingForm.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting training...';
            }
        });
    }
    
    // Add event listeners for the cancel training buttons
    document.querySelectorAll('.cancel-training-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const modelId = this.dataset.modelId;
            if (modelId && confirm('Are you sure you want to cancel this training? Progress will be lost.')) {
                cancelTraining(modelId);
            }
        });
    });
    
    // Add event listeners for retry buttons
    document.querySelectorAll('.retry-training-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const modelId = this.dataset.modelId;
            if (modelId && confirm('Do you want to retry training this model?')) {
                retryTraining(modelId);
            }
        });
    });
    
    // Add event listeners for delete model buttons
    document.querySelectorAll('.delete-model-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const modelId = this.dataset.modelId;
            if (modelId) {
                deleteModel(modelId);
            }
        });
    });
});

/**
 * Update the status of a model in training
 * @param {string} modelId - The ID of the model to update
 */
function updateModelStatus(modelId) {
    fetch(`/api/models/${modelId}/status`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Get the model card element
            const modelCard = document.querySelector(`.model-card[data-model-id="${modelId}"]`);
            if (!modelCard) return;
            
            // Update model status
            modelCard.dataset.status = data.status;
            
            // Get status badge and update it
            const statusBadge = modelCard.querySelector('.status-badge');
            if (statusBadge) {
                statusBadge.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                
                // Update badge class based on status
                statusBadge.classList.remove('bg-primary', 'bg-success', 'bg-danger', 'bg-secondary');
                statusBadge.classList.remove('status-training', 'status-complete', 'status-failed', 'status-creating');
                
                statusBadge.classList.add(`status-${data.status}`);
                
                if (data.status === 'training') {
                    statusBadge.classList.add('bg-primary');
                } else if (data.status === 'complete') {
                    statusBadge.classList.add('bg-success');
                } else if (data.status === 'failed') {
                    statusBadge.classList.add('bg-danger');
                } else {
                    statusBadge.classList.add('bg-secondary');
                }
            }
            
            // Update progress if model is in training
            const progressBar = modelCard.querySelector('.progress-bar');
            if (progressBar && data.status === 'training') {
                // Calculate progress percentage
                const progress = data.current_epoch && data.epochs ? 
                    Math.round((data.current_epoch / data.epochs) * 100) : 0;
                
                progressBar.style.width = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);
                progressBar.textContent = `${progress}%`;
                
                // Update epoch count if available
                const epochCount = modelCard.querySelector('.epoch-count');
                if (epochCount && data.current_epoch !== undefined) {
                    epochCount.textContent = `${data.current_epoch}/${data.epochs}`;
                }
            }
            
            // Update metrics if model is complete
            if (data.status === 'complete' && data.metrics) {
                const metricsContainer = modelCard.querySelector('.metrics-container');
                if (metricsContainer) {
                    metricsContainer.innerHTML = '';
                    
                    // Add metrics
                    if (data.metrics.map50 !== undefined) {
                        const map50El = document.createElement('div');
                        map50El.classList.add('metric-item');
                        map50El.innerHTML = `<strong>mAP50:</strong> ${(data.metrics.map50 * 100).toFixed(2)}%`;
                        metricsContainer.appendChild(map50El);
                    }
                    
                    if (data.metrics.map !== undefined) {
                        const mapEl = document.createElement('div');
                        mapEl.classList.add('metric-item');
                        mapEl.innerHTML = `<strong>mAP50-95:</strong> ${(data.metrics.map * 100).toFixed(2)}%`;
                        metricsContainer.appendChild(mapEl);
                    }
                    
                    if (data.metrics.fitness !== undefined) {
                        const fitnessEl = document.createElement('div');
                        fitnessEl.classList.add('metric-item');
                        fitnessEl.innerHTML = `<strong>Fitness:</strong> ${data.metrics.fitness.toFixed(4)}`;
                        metricsContainer.appendChild(fitnessEl);
                    }
                }
            }
            
            // If model is still training, continue polling
            if (data.status === 'training' || data.status === 'creating') {
                // Use longer polling interval for creating state
                const interval = data.status === 'creating' ? 10000 : 5000;
                setTimeout(() => updateModelStatus(modelId), interval);
            } else if (data.status === 'complete' || data.status === 'failed') {
                // If model training has finished, update the page without reloading
                const actionButtons = modelCard.querySelector('.action-buttons');
                if (actionButtons) {
                    // Clear the action buttons
                    actionButtons.innerHTML = '';
                    
                    // Add appropriate buttons based on status
                    if (data.status === 'complete') {
                        // Add download button
                        const downloadBtn = document.createElement('a');
                        downloadBtn.href = `/models/${modelId}/download/pt`;
                        downloadBtn.classList.add('btn', 'btn-sm', 'btn-success', 'mr-2');
                        downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download';
                        actionButtons.appendChild(downloadBtn);
                        
                        // Add view results button
                        const viewBtn = document.createElement('button');
                        viewBtn.classList.add('btn', 'btn-sm', 'btn-primary', 'view-model-btn');
                        viewBtn.dataset.modelId = modelId;
                        viewBtn.innerHTML = '<i class="fas fa-eye"></i> View Results';
                        actionButtons.appendChild(viewBtn);
                    } else if (data.status === 'failed') {
                        // Add retry button
                        const retryBtn = document.createElement('button');
                        retryBtn.classList.add('btn', 'btn-sm', 'btn-primary', 'retry-training-btn');
                        retryBtn.dataset.modelId = modelId;
                        retryBtn.innerHTML = '<i class="fas fa-redo"></i> Retry';
                        actionButtons.appendChild(retryBtn);
                        
                        // Add event listener to retry button
                        retryBtn.addEventListener('click', function() {
                            if (confirm('Do you want to retry training this model?')) {
                                retryTraining(modelId);
                            }
                        });
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error updating model status:', error);
            // Retry after a longer delay on error
            setTimeout(() => updateModelStatus(modelId), 15000);
        });
}

/**
 * Cancel training for a model
 * @param {string} modelId - The ID of the model to cancel
 */
function cancelTraining(modelId) {
    // Call the cancel endpoint
    fetch(`/api/models/${modelId}/cancel`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Refresh the page to show updated status
            window.location.reload();
        } else {
            // Show error message
            alert('Error cancelling training: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error cancelling training:', error);
        alert('Error cancelling training: ' + error.message);
    });
}

/**
 * Retry training for a failed model
 * @param {string} modelId - The ID of the model to retry
 */
function retryTraining(modelId) {
    // Call the retry endpoint
    fetch(`/api/models/${modelId}/retry`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // If there's a redirect URL, go there
            if (data.redirect) {
                window.location.href = data.redirect;
            } else {
                // Otherwise just reload the page
                window.location.reload();
            }
        } else {
            // Show error message
            alert('Error retrying training: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error retrying training:', error);
        alert('Error retrying training: ' + error.message);
    });
}

/**
 * Delete a model
 * @param {string} modelId - The ID of the model to delete
 */
function deleteModel(modelId) {
    if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
        return;
    }
    
    // Call the delete endpoint
    fetch(`/api/models/${modelId}`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Remove the model card from the UI
            const modelCard = document.querySelector(`.model-card[data-model-id="${modelId}"]`);
            if (modelCard) {
                modelCard.remove();
            }
            // Show a success message
            alert(data.message || 'Model deleted successfully');
            
            // If there are no more models, refresh the page to show the "no models" message
            const remainingCards = document.querySelectorAll('.model-card');
            if (remainingCards.length === 0) {
                window.location.reload();
            }
        } else {
            // Show error message
            alert('Error deleting model: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error deleting model:', error);
        alert('Error deleting model: ' + error.message);
    });
}

// Dataset Preview Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Get total annotation count
    const projectId = document.getElementById('project-id').value;
    if (projectId) {
        fetchAnnotationCount(projectId);
    }
    
    // Modal functionality
    const previewBtn = document.getElementById('preview-datasets-btn');
    const modal = document.getElementById('dataset-preview-modal');
    const closeModal = document.querySelector('.close-modal');
    
    if (previewBtn && modal) {
        previewBtn.addEventListener('click', function() {
            modal.classList.add('active');
            // Load images for active tab
            const activeTab = document.querySelector('.modal-tabs .tab.active');
            if (activeTab) {
                const splitType = activeTab.dataset.tab;
                loadPreviewImages(splitType);
            }
        });
    }
    
    if (closeModal && modal) {
        closeModal.addEventListener('click', function() {
            modal.classList.remove('active');
        });
        
        // Close modal when clicking outside the content
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }
    
    // Tab switching
    const tabs = document.querySelectorAll('.modal-tabs .tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            // Add active class to clicked tab
            tab.classList.add('active');
            
            // Hide all preview containers
            const containers = document.querySelectorAll('.preview-container');
            containers.forEach(c => c.classList.remove('active'));
            
            // Show the selected preview container
            const splitType = tab.dataset.tab;
            const selectedContainer = document.getElementById(`${splitType}-preview`);
            if (selectedContainer) {
                selectedContainer.classList.add('active');
                loadPreviewImages(splitType);
            }
        });
    });
});

/**
 * Fetch total annotation count for the project
 */
function fetchAnnotationCount(projectId) {
    fetch(`/api/projects/${projectId}/annotation_count`)
        .then(response => response.json())
        .then(data => {
            const countElement = document.getElementById('annotation-count');
            if (countElement && data.count !== undefined) {
                countElement.textContent = data.count;
            }
        })
        .catch(error => {
            console.error('Error fetching annotation count:', error);
            const countElement = document.getElementById('annotation-count');
            if (countElement) {
                countElement.textContent = 'N/A';
            }
        });
}

/**
 * Load preview images for a specific split type
 */
function loadPreviewImages(splitType) {
    const projectId = document.getElementById('project-id').value;
    const previewGrid = document.querySelector(`#${splitType}-preview .preview-grid`);
    const loadingIndicator = document.querySelector(`#${splitType}-preview .preview-loading`);
    
    if (!projectId || !previewGrid || !loadingIndicator) return;
    
    // Show loading indicator
    loadingIndicator.style.display = 'flex';
    previewGrid.innerHTML = '';
    
    // Fetch images for this split
    fetch(`/api/projects/${projectId}/images_by_split?split=${splitType}&limit=12`)
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            
            if (data.images && data.images.length > 0) {
                data.images.forEach(image => {
                    const imageContainer = document.createElement('div');
                    imageContainer.className = 'preview-image-container';
                    
                    const img = document.createElement('img');
                    img.className = 'preview-image';
                    img.src = image.url;
                    img.alt = image.original_filename;
                    img.loading = 'lazy';
                    
                    const annotationCount = document.createElement('div');
                    annotationCount.className = 'preview-annotation-count';
                    annotationCount.textContent = `${image.annotations_count || 0} annotations`;
                    
                    imageContainer.appendChild(img);
                    imageContainer.appendChild(annotationCount);
                    previewGrid.appendChild(imageContainer);
                });
            } else {
                previewGrid.innerHTML = `
                    <div style="text-align: center; padding: 40px; color: #718096;">
                        <i class="fas fa-image" style="font-size: 32px; margin-bottom: 16px;"></i>
                        <p>No ${splitType} images available</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error(`Error loading ${splitType} images:`, error);
            loadingIndicator.style.display = 'none';
            previewGrid.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #718096;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 32px; margin-bottom: 16px;"></i>
                    <p>Error loading images</p>
                </div>
            `;
        });
} 