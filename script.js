const API_URL = 'https://nba-stats-xcyv.onrender.com';

async function scanOpportunities(statType) {
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

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000);

    try {
        const response = await fetch(
            `${API_URL}/api/${endpoints[statType]}?min_edge=${minEdge}&min_r2=${minR2}`,
            { signal: controller.signal }
        );

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (data.status === 'SUCCESS') {
            displayResults(data, statType);
        } else {
            displayError(data.message || 'Erreur inconnue');
        }
    } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
            displayError('⏱️ Timeout après 5 min');
        } else {
            displayError(`❌ Erreur: ${error.message}`);
        }
    } finally {
        document.getElementById('loadingDiv').classList.add('hidden');
        document.querySelectorAll('.btn-scan').forEach(btn => btn.disabled = false);
    }
}

function displayResults(data, statType) {
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
        displayError(`Aucun pari trouvé avec R² ≥ ${document.getElementById('minR2').value} et Edge ≥ ${document.getElementById('minEdge').value}%. Baisse R² à 0.10 et Edge à 0% pour tester.`);
        return;
    }
    
    const resultsDiv = document.getElementById('resultsDiv');
    resultsDiv.innerHTML = opportunities.map((opp, index) => 
        createOpportunityCard(opp, index + 1)
    ).join('');
    resultsDiv.classList.remove('hidden');
    
    document.querySelectorAll('.technical-header').forEach(header => {
        header.addEventListener('click', function() {
            const content = this.nextElementSibling;
            const icon = this.querySelector('.toggle-icon');
            content.classList.toggle('open');
            icon.classList.toggle('open');
        });
    });
}

function displayError(message) {
    const errorDiv = document.getElementById('errorDiv');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function getTeamName(abbr) {
    const teams = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'LA Clippers', 'LAL': 'LA Lakers', 'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    };
    return teams[abbr] || abbr;
}

function calculateProbabilities(prediction, ci, line) {
    const std = (ci.upper - ci.lower) / 3.92;
    const results = [];
    
    for (let offset = -2; offset <= 2; offset += 0.5) {
        const testLine = line + offset;
        const z = (testLine - prediction) / std;
        const overProb = (1 - normalCDF(z)) * 100;
        const underProb = 100 - overProb;
        
        results.push({
            line: testLine,
            overProb: overProb.toFixed(1),
            underProb: underProb.toFixed(1)
        });
    }
    
    return results;
}

function normalCDF(z) {
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989423 * Math.exp(-z * z / 2);
    const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z > 0 ? 1 - prob : prob;
}

function closePlayerModal() {
    document.getElementById('playerModal').classList.remove('open');
}

window.onclick = function(event) {
    const modal = document.getElementById('playerModal');
    if (event.target === modal) {
        closePlayerModal();
    }
}
function createOpportunityCard(opp, rank) {
    const rec = opp.line_analysis.recommendation;
    const recClass = rec.toLowerCase();
    const edge = opp.line_analysis.edge;
    const kelly = opp.line_analysis.kelly_criterion;
    const r2 = opp.regression_stats?.r_squared || 0;
    const rmse = opp.regression_stats?.rmse || 0;
    const ci = opp.confidence_interval;
    const line = opp.line_analysis.bookmaker_line;

    const statLabel = {
        'points': 'Points',
        'assists': 'Assists',
        'rebounds': 'Rebounds',
        '3pt': '3-Points'
    }[opp.stat_type];

    const teamName = getTeamName(opp.opponent);
    const probabilities = calculateProbabilities(opp.prediction, ci, line);
    const betRangeLower = rec === 'OVER' ? line : ci.lower;
    const betRangeUpper = rec === 'OVER' ? ci.upper : line;

    return `
        <div class="opp-card ${recClass}" data-rank="${rank}">
            <div class="opp-header">
                <div class="opp-player">${opp.player}</div>
                <div class="opp-matchup">
                    ${opp.is_home ? '🏠 vs' : '✈️ @'} ${teamName} • ${statLabel}
                </div>
                <div class="opp-badge ${recClass}">
                    ${rec} ${line}
                </div>
            </div>

            <div class="prediction-box">
                <div class="pred-row">
                    <div>
                        <div class="pred-label">Prédiction modèle</div>
                        <div class="pred-value">${opp.prediction.toFixed(1)}</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="pred-label">vs Ligne</div>
                        <div class="pred-value">${line}</div>
                    </div>
                </div>
                
                <div class="confidence-interval">
                    <div class="confidence-interval-label">Intervalle de confiance 95%</div>
                    <div class="confidence-interval-range">${ci.lower.toFixed(1)} - ${ci.upper.toFixed(1)}</div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">🎯 R² TEST</div>
                        <div class="metric-value">${(r2 * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">📊 RMSE</div>
                        <div class="metric-value">${rmse.toFixed(1)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">💰 Edge</div>
                        <div class="metric-value">+${edge.toFixed(1)}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">📈 Kelly</div>
                        <div class="metric-value">${kelly.toFixed(1)}%</div>
                    </div>
                </div>
            </div>

            <div class="action-box ${recClass}">
                <div class="action-title">🎯 ACTION À PRENDRE</div>
                <div class="action-detail">
                    ➤ Parier <strong>${rec} ${line}</strong><br>
                    ➤ Mise: <strong>${kelly.toFixed(1)}%</strong> de ta bankroll<br>
                    ➤ R² TEST: <strong>${(r2 * 100).toFixed(0)}%</strong> (performance réelle!)
                </div>
            </div>
            
            <div class="technical-details">
                <div class="technical-header">
                    <div class="technical-title">📊 Détails techniques</div>
                    <div class="toggle-icon">▼</div>
                </div>
                <div class="technical-content">
                    <div class="bet-range" style="margin-top: 15px;">
                        <div class="bet-range-title">💡 Range de pari recommandé</div>
                        <div class="bet-range-value">
                            ${rec} entre ${betRangeLower.toFixed(1)} et ${betRangeUpper.toFixed(1)}
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9em; color: #374151;">
                            Si la ligne sur ton bookmaker est dans cet intervalle, le pari reste +EV.
                        </div>
                    </div>
                    
                    <table class="prob-table" style="margin-top: 15px;">
                        <thead>
                            <tr>
                                <th>Ligne</th>
                                <th>Prob OVER</th>
                                <th>Prob UNDER</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${probabilities.map(p => `
                                <tr ${p.line === line ? 'style="background: #fef3c7; font-weight: bold;"' : ''}>
                                    <td>${p.line.toFixed(1)}</td>
                                    <td>${p.overProb}%</td>
                                    <td>${p.underProb}%</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}