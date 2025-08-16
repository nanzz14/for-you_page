const API_BASE = 'http://localhost:8000';

// Global variables for current user and item
let currentUsername = '';
let currentItemId = null;

async function getRecommendations() {
    const username = document.getElementById('usernameInput').value.trim();
    
    if (!username) {
        showError('Please enter a username');
        return;
    }

    currentUsername = username;

    // Show loading state
    document.getElementById('loadingDiv').style.display = 'block';
    document.getElementById('errorDiv').style.display = 'none';
    document.getElementById('recommendationsContainer').style.display = 'none';
    document.getElementById('searchBtn').disabled = true;

    try {
        const response = await fetch(`${API_BASE}/v1/reco/homefeed/${encodeURIComponent(username)}`);
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status} - ${await response.text()}`);
        }

        const data = await response.json();
        displayRecommendations(data);

    } catch (error) {
        console.error('Error fetching recommendations:', error);
        showError(`Failed to load recommendations: ${error.message}`);
    } finally {
        document.getElementById('loadingDiv').style.display = 'none';
        document.getElementById('searchBtn').disabled = false;
    }
}

function displayRecommendations(data) {
    document.getElementById('userTitle').textContent = `Recommendations for ${data.user_name}`;
    document.getElementById('userCommunity').textContent = `Community: ${data.user_community}`;

    const listContainer = document.getElementById('recommendationsList');
    
    if (!data.recommendations || data.recommendations.length === 0) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <h3>No recommendations found</h3>
                <p>Try with a different username or check back later!</p>
            </div>
        `;
    } else {
        listContainer.innerHTML = data.recommendations.map((rec, index) => `
            <div class="recommendation-item">
                <div class="item-type">${rec.item_type}</div>
                <div class="item-title">${rec.title}</div>
                <div class="item-description">${rec.description}</div>
                <div class="item-meta">
                    <span>${rec.price === 0 ? 'Free' : `Rs ${rec.price}`}</span>
                    <span>⭐ ${rec.rating > 0 ? rec.rating.toFixed(1) + '/5' : 'No rating'}</span>
                    <span><strong>Score:</strong> ${rec.score.toFixed(2)}</span>
                </div>
                <div class="item-reason">
                    💡 ${rec.reason}
                </div>
                <div class="item-actions">
                    <button class="action-btn feedback-trigger" onclick="openFeedbackModal(${rec.item_id})">
                        💬 Give Feedback
                    </button>
                </div>
            </div>
        `).join('');
    }

    document.getElementById('recommendationsContainer').style.display = 'block';
}

// Feedback Modal Functions
function openFeedbackModal(itemId) {
    currentItemId = itemId;
    document.getElementById('feedbackModal').style.display = 'block';
    document.getElementById('feedbackMessage').style.display = 'none';
}

async function submitFeedback(feedbackType) {
    if (!currentUsername || !currentItemId) {
        showFeedbackMessage('Error: Missing user or item information', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/v1/reco/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: currentItemId, // Using item_id as user_id for demo
                item_id: currentItemId,
                feedback_type: feedbackType
            })
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }

        const result = await response.json();
        showFeedbackMessage(`Feedback submitted successfully! Thank you for your ${feedbackType} response.`, 'success');
        
        // Close modal after 2 seconds
        setTimeout(() => {
            closeFeedbackModal();
        }, 2000);

    } catch (error) {
        console.error('Error submitting feedback:', error);
        showFeedbackMessage(`Failed to submit feedback: ${error.message}`, 'error');
    }
}

function showFeedbackMessage(message, type) {
    const messageDiv = document.getElementById('feedbackMessage');
    messageDiv.textContent = message;
    messageDiv.className = `feedback-message ${type}`;
    messageDiv.style.display = 'block';
}

function closeFeedbackModal() {
    document.getElementById('feedbackModal').style.display = 'none';
    currentItemId = null;
}

function showError(message) {
    const errorDiv = document.getElementById('errorDiv');
    errorDiv.innerHTML = `<div class="error-message">${message}</div>`;
    errorDiv.style.display = 'block';
}

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Allow Enter key to trigger search
    document.getElementById('usernameInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getRecommendations();
        }
    });

    // Close modals when clicking outside or on close button
    window.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
        }
    });

    // Close button event listeners
    document.querySelectorAll('.close').forEach(closeBtn => {
        closeBtn.addEventListener('click', function() {
            this.closest('.modal').style.display = 'none';
        });
    });

    // Escape key to close modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal').forEach(modal => {
                modal.style.display = 'none';
            });
        }
    });
});
