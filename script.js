const API_URL = 'https://nba-stats-xcyv.onrender.com';

console.log('🚀 Script chargé! API:', API_URL);

async function scanOpportunities(statType) {
    console.log('▶️ Scan demandé:', statType);
    
    const minEdge = document.getElementById('minEdge').value;
    const minR2 = parseFloat(document.getElementById('minR2').value);

    document.querySelectorAll('.btn-scan').forEach(btn => btn.disabled = true);

    document.getElementById('loadingDiv').classList.remove('hidden');
    document.getElementById('statsBar').classList.add('hidden');
    document.getElementById('resultsDiv').classList.add('hidden');
    document.getElementById('errorDiv').classList.add('hidden');

    const endpoints = {
        'points': 'daily-opportunities-points',
        'assists': 'daily-opportunities-assists',
        'rebounds': 'daily-opportunities-rebounds',
        '3pt': 'daily-opportunities-3pt'
    };

    const url = `${API_URL}/api/${endpoints[statType]}?min_edge=${minEdge}&min_r2=${minR2}`;
    console.log('📡 URL:', url);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000);

    try {
        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeoutId);

        console.log('✅ Réponse reçue:', response.status);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('📊 Données:', data);

        if (data.status === 'SUCCESS') {
            displayResults(data, statType);
        } else {
            displayError(data.message || 'Erreur inconnue');
        }
    } catch (error) {
        clearTimeout(timeoutId);
        console.error('❌ Erreur scan:', error);
        
        if (error.name === 'AbortError') {
            displayError('⏱️ Timeout après 3 min');
        } else {
            displayError(`❌ Erreur: ${error.message}`);
        }
    } finally {
        document.getElementById('loadingDiv').classList.add('hidden');
        document.querySelectorAll('.btn-scan').forEach(btn => btn.disabled = false);
    }
}

function displayResults(data, statType) {
    console.log('📊 Affichage résultats...');
    
    const opportunities = data.opportunities || [];
    
    document.getElementById('statsBar').classList.remove('hidden');
    
    const statLabels = {
        'points': 'POINTS',
        'assists': 'ASSISTS',
        'rebounds': 'REBOUNDS',
        '3pt': '3-POINTS'
    };
    
    document.getElementById('statType').textContent = statLabels[statType];
    document.getElementById('analyzedProps').textContent = data.total_analyzed;
    document.getElementById('foundOpps').textContent = opportunities.length;
    
    if (opportunities.length > 0) {
        const avgR2 = opportunities.reduce((sum, o) => 
            sum + (o.regression_stats?.r_squared || 0), 0) / opportunities.length;
        document.getElementById('avgR2').textContent = avgR2.toFixed(3);
    }

    if (opportunities.length === 0) {
        displayError(`Aucun pari trouvé. Analysé: ${data.total_analyzed}`);
        return;
    }
const resultsDiv = document.getElementById('resultsDiv');
    resultsDiv.innerHTML = `
        <div style="padding: 30px; background: white; border-radius: 15px; text-align: center;">
            <h2 style="color: #10b981; margin-bottom: 15px;">✅ ${opportunities.length} Opportunités Trouvées!</h2>
            <p style="font-size: 1.1em; color: #666;">
                📊 Total analysé: ${data.total_analyzed}<br>
                🎯 R² moyen: ${document.getElementById('avgR2').textContent}
            </p>
            <div style="margin-top: 20px;">
                ${opportunities.slice(0, 5).map((opp, i) => `
                    <div style="margin: 10px 0; padding: 15px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #10b981;">
                        <strong style="color: #667eea; font-size: 1.2em;">${i+1}. ${opp.player}</strong><br>
                        <span style="color: #666;">
                            ${opp.line_analysis.recommendation} ${opp.line_analysis.bookmaker_line} • 
                            R² ${(opp.regression_stats.r_squared * 100).toFixed(0)}% • 
                            Edge +${opp.line_analysis.edge.toFixed(1)}%
                        </span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    resultsDiv.classList.remove('hidden');
}

function displayError(message) {
    console.log('❌ Erreur:', message);
    const errorDiv = document.getElementById('errorDiv');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function closePlayerModal() {
    document.getElementById('playerModal').classList.remove('open');
}

function openPlayerModal(player) {
    alert('Historique joueur: ' + player);
}

console.log('✅ Script complet chargé!');