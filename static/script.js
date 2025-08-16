const API_BASE = 'http://localhost:8000';

async function getRecommendations() {
    const username = document.getElementById('usernameInput').value.trim();
    
    if (!username) {
        showError('Please enter a username');
        return;
    }

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
                    <span><strong>Price:</strong> ${rec.price === 0 ? 'Free' : `Rs ${rec.price}`}</span>
                    <span><strong>Rating:</strong> ${rec.rating > 0 ? rec.rating.toFixed(1) + '/5' : 'No rating'}</span>
                    <span><strong>Score:</strong> ${rec.score.toFixed(2)}</span>
                </div>
                <div class="item-reason">
                    💡 ${rec.reason}
                </div>
                
            </div>
        `).join('');
    }

    document.getElementById('recommendationsContainer').style.display = 'block';
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
});
