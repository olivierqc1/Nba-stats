// ========================================
// BET TRACKER - NBA ANALYZER
// ========================================

// Initialize
let bets = [];
let currentFilter = 'all';

// Load bets from localStorage
function loadBets() {
    const stored = localStorage.getItem('nbaBets');
    if (stored) {
        bets = JSON.parse(stored);
    }
    updateAll();
}

// Save bets to localStorage
function saveBets() {
    localStorage.setItem('nbaBets', JSON.stringify(bets));
}

// Set today's date as default
document.addEventListener('DOMContentLoaded', () => {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('betDate').value = today;
    loadBets();
});

// Form submission
document.getElementById('addBetForm').addEventListener('submit', (e) => {
    e.preventDefault();
    
    const bet = {
        id: Date.now(),
        date: document.getElementById('betDate').value,
        playerName: document.getElementById('playerName').value,
        propType: document.getElementById('propType').value,
        betSide: document.getElementById('betSide').value,
        line: parseFloat(document.getElementById('line').value),
        prediction: parseFloat(document.getElementById('prediction').value),
        r2: parseFloat(document.getElementById('r2').value),
        edge: parseFloat(document.getElementById('edge').value),
        stake: parseFloat(document.getElementById('stake').value),
        odds: parseFloat(document.getElementById('odds').value),
        result: document.getElementById('result').value,
        actualResult: document.getElementById('actualResult').value ? parseFloat(document.getElementById('actualResult').value) : null
    };
    
    // Calculate profit/loss
    bet.profitLoss = calculateProfitLoss(bet);
    
    bets.push(bet);
    saveBets();
    updateAll();
    
    // Reset form
    e.target.reset();
    document.getElementById('betDate').value = new Date().toISOString().split('T')[0];
    
    // Show success message
    alert('✅ Pari ajouté avec succès!');
});

// Calculate profit/loss
function calculateProfitLoss(bet) {
    if (bet.result === 'pending') return 0;
    
    if (bet.result === 'won') {
        // American odds to decimal
        if (bet.odds < 0) {
            return bet.stake * (100 / Math.abs(bet.odds));
        } else {
            return bet.stake * (bet.odds / 100);
        }
    } else {
        return -bet.stake;
    }
}

// Delete bet
function deleteBet(id) {
    if (confirm('⚠️ Êtes-vous sûr de vouloir supprimer ce pari?')) {
        bets = bets.filter(b => b.id !== id);
        saveBets();
        updateAll();
    }
}

// Edit bet result
function editBetResult(id) {
    const bet = bets.find(b => b.id === id);
    if (!bet) return;
    
    const newResult = prompt('Résultat (won/lost/pending):', bet.result);
    if (newResult && ['won', 'lost', 'pending'].includes(newResult)) {
        bet.result = newResult;
        
        const actualResult = prompt('Résultat réel (optionnel):', bet.actualResult || '');
        if (actualResult) {
            bet.actualResult = parseFloat(actualResult);
        }
        
        bet.profitLoss = calculateProfitLoss(bet);
        saveBets();
        updateAll();
    }
}

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        updateTable();
    });
});

// Update all sections
function updateAll() {
    updateStats();
    updateTable();
    updateCharts();
    updateAnalysis();
}

// Update overview stats
function updateStats() {
    const settledBets = bets.filter(b => b.result !== 'pending');
    const wonBets = bets.filter(b => b.result === 'won');
    const lostBets = bets.filter(b => b.result === 'lost');
    
    // Win rate
    const winRate = settledBets.length > 0 ? (wonBets.length / settledBets.length * 100) : 0;
    document.getElementById('totalWinRate').textContent = winRate.toFixed(1) + '%';
    document.getElementById('totalWinRate').className = 'stat-value ' + (winRate >= 55 ? 'positive' : winRate < 50 ? 'negative' : '');
    document.getElementById('totalRecord').textContent = `${wonBets.length}-${lostBets.length}-${bets.length - settledBets.length}`;
    
    // Total profit
    const totalProfit = bets.reduce((sum, bet) => sum + bet.profitLoss, 0);
    document.getElementById('totalProfit').textContent = (totalProfit > 0 ? '+' : '') + totalProfit.toFixed(2) + '€';
    document.getElementById('totalProfit').className = 'stat-value ' + (totalProfit > 0 ? 'positive' : totalProfit < 0 ? 'negative' : '');
    document.getElementById('totalBets').textContent = `${bets.length} paris`;
    
    // ROI
    const totalStaked = bets.reduce((sum, bet) => sum + bet.stake, 0);
    const roi = totalStaked > 0 ? (totalProfit / totalStaked * 100) : 0;
    document.getElementById('totalROI').textContent = (roi > 0 ? '+' : '') + roi.toFixed(1) + '%';
    document.getElementById('totalROI').className = 'stat-value ' + (roi > 0 ? 'positive' : roi < 0 ? 'negative' : '');
    const avgStake = bets.length > 0 ? (totalStaked / bets.length) : 0;
    document.getElementById('avgStake').textContent = avgStake.toFixed(0) + '€ moy.';
    
    // Current streak
    let currentStreak = 0;
    let bestStreak = 0;
    let tempStreak = 0;
    
    const sortedBets = [...settledBets].sort((a, b) => new Date(b.date) - new Date(a.date));
    
    for (let i = 0; i < sortedBets.length; i++) {
        if (sortedBets[i].result === 'won') {
            tempStreak++;
            if (i === 0) currentStreak = tempStreak;
        } else {
            if (i === 0) currentStreak = -1;
            tempStreak = 0;
        }
        bestStreak = Math.max(bestStreak, tempStreak);
    }
    
    document.getElementById('currentStreak').textContent = currentStreak > 0 ? `${currentStreak}W 🔥` : currentStreak < 0 ? '1L' : '--';
    document.getElementById('bestStreak').textContent = `Record: ${bestStreak}W`;
}

// Update bets table
function updateTable() {
    const tbody = document.getElementById('betsTableBody');
    
    let filteredBets = bets;
    if (currentFilter !== 'all') {
        filteredBets = bets.filter(b => b.result === currentFilter);
    }
    
    // Sort by date (most recent first)
    filteredBets.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    if (filteredBets.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="12" class="empty-state">
                    <div class="empty-state-icon">📭</div>
                    <div>Aucun pari ${currentFilter !== 'all' ? currentFilter : 'enregistré'}</div>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = filteredBets.map(bet => {
        const statusClass = bet.result === 'won' ? 'status-won' : bet.result === 'lost' ? 'status-lost' : 'status-pending';
        const statusText = bet.result === 'won' ? 'Gagné ✅' : bet.result === 'lost' ? 'Perdu ❌' : 'En cours ⏳';
        const profitClass = bet.profitLoss > 0 ? 'profit-positive' : bet.profitLoss < 0 ? 'profit-negative' : '';
        
        return `
            <tr>
                <td>${new Date(bet.date).toLocaleDateString('fr-FR')}</td>
                <td><strong>${bet.playerName}</strong></td>
                <td>${bet.betSide} ${bet.propType}</td>
                <td>${bet.line}</td>
                <td>${bet.prediction.toFixed(1)}</td>
                <td>${(bet.r2 * 100).toFixed(1)}%</td>
                <td>+${bet.edge.toFixed(1)}%</td>
                <td>${bet.stake}€</td>
                <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                <td>${bet.actualResult !== null ? bet.actualResult.toFixed(1) : '--'}</td>
                <td class="${profitClass}">${bet.profitLoss > 0 ? '+' : ''}${bet.profitLoss.toFixed(2)}€</td>
                <td>
                    <button class="btn-edit" onclick="editBetResult(${bet.id})">✏️</button>
                    <button class="btn-delete" onclick="deleteBet(${bet.id})">🗑️</button>
                </td>
            </tr>
        `;
    }).join('');
}

// Update charts
function updateCharts() {
    updateWinRateByR2Chart();
    updateCumulativeProfitChart();
}

// Win Rate by R² Chart
function updateWinRateByR2Chart() {
    const ctx = document.getElementById('winRateByR2Chart');
    
    // Group by R² ranges
    const ranges = [
        { label: '< 50%', min: 0, max: 0.5 },
        { label: '50-60%', min: 0.5, max: 0.6 },
        { label: '60-70%', min: 0.6, max: 0.7 },
        { label: '70-80%', min: 0.7, max: 0.8 },
        { label: '80%+', min: 0.8, max: 1 }
    ];
    
    const data = ranges.map(range => {
        const inRange = bets.filter(b => b.r2 >= range.min && b.r2 < range.max && b.result !== 'pending');
        const won = inRange.filter(b => b.result === 'won').length;
        return inRange.length > 0 ? (won / inRange.length * 100) : 0;
    });
    
    const counts = ranges.map(range => {
        return bets.filter(b => b.r2 >= range.min && b.r2 < range.max && b.result !== 'pending').length;
    });
    
    if (window.winRateChart) {
        window.winRateChart.destroy();
    }
    
    window.winRateChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ranges.map((r, i) => `${r.label}\n(n=${counts[i]})`),
            datasets: [{
                label: 'Win Rate (%)',
                data: data,
                backgroundColor: data.map(d => d >= 55 ? 'rgba(16, 185, 129, 0.7)' : d < 50 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(251, 191, 36, 0.7)'),
                borderColor: data.map(d => d >= 55 ? 'rgb(16, 185, 129)' : d < 50 ? 'rgb(239, 68, 68)' : 'rgb(251, 191, 36)'),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `Win Rate: ${context.parsed.y.toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: (value) => value + '%'
                    }
                }
            }
        }
    });
}

// Cumulative Profit Chart
function updateCumulativeProfitChart() {
    const ctx = document.getElementById('cumulativeProfitChart');
    
    const sortedBets = [...bets].filter(b => b.result !== 'pending').sort((a, b) => new Date(a.date) - new Date(b.date));
    
    let cumulative = 0;
    const data = sortedBets.map(bet => {
        cumulative += bet.profitLoss;
        return cumulative;
    });
    
    const labels = sortedBets.map((bet, i) => `${i + 1}`);
    
    if (window.profitChart) {
        window.profitChart.destroy();
    }
    
    window.profitChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Profit Cumulé (€)',
                data: data,
                borderColor: 'rgb(102, 126, 234)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        label: (context) => `Profit: ${context.parsed.y >= 0 ? '+' : ''}${context.parsed.y.toFixed(2)}€`
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Nombre de Paris'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Profit (€)'
                    },
                    ticks: {
                        callback: (value) => value + '€'
                    }
                }
            }
        }
    });
}

// Update analysis section
function updateAnalysis() {
    updatePropTypeStats();
    updateBestBets();
    updateWorstBets();
}

// Prop type stats
function updatePropTypeStats() {
    const container = document.getElementById('propTypeStats');
    
    const propTypes = ['PTS', 'REB', 'AST', '3PT', 'STL', 'BLK'];
    
    const html = propTypes.map(type => {
        const typeBets = bets.filter(b => b.propType === type && b.result !== 'pending');
        const won = typeBets.filter(b => b.result === 'won').length;
        const lost = typeBets.filter(b => b.result === 'lost').length;
        const winRate = typeBets.length > 0 ? (won / typeBets.length * 100) : 0;
        
        if (typeBets.length === 0) return '';
        
        return `
            <div class="prop-stat-item">
                <div>
                    <div class="prop-type">${type}</div>
                    <div class="prop-record">${won}-${lost} (${winRate.toFixed(1)}%)</div>
                </div>
                <div style="font-size: 24px;">
                    ${winRate >= 60 ? '🔥' : winRate >= 50 ? '✅' : '❌'}
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html || '<div class="empty-state">Aucune donnée</div>';
}

// Best bets
function updateBestBets() {
    const container = document.getElementById('bestBets');
    
    const settledBets = bets.filter(b => b.result !== 'pending');
    const bestBets = [...settledBets]
        .sort((a, b) => b.profitLoss - a.profitLoss)
        .slice(0, 5);
    
    const html = bestBets.map(bet => `
        <div class="bet-item">
            <div class="bet-item-header">
                ${bet.playerName} - ${bet.betSide} ${bet.propType} ${bet.line}
            </div>
            <div class="bet-item-details">
                ${new Date(bet.date).toLocaleDateString('fr-FR')} • R² ${(bet.r2 * 100).toFixed(0)}% • 
                <span class="profit-positive">+${bet.profitLoss.toFixed(2)}€</span>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html || '<div class="empty-state">Aucun pari complété</div>';
}

// Worst bets
function updateWorstBets() {
    const container = document.getElementById('worstBets');
    
    const settledBets = bets.filter(b => b.result !== 'pending');
    const worstBets = [...settledBets]
        .sort((a, b) => a.profitLoss - b.profitLoss)
        .slice(0, 5);
    
    const html = worstBets.map(bet => `
        <div class="bet-item">
            <div class="bet-item-header">
                ${bet.playerName} - ${bet.betSide} ${bet.propType} ${bet.line}
            </div>
            <div class="bet-item-details">
                ${new Date(bet.date).toLocaleDateString('fr-FR')} • R² ${(bet.r2 * 100).toFixed(0)}% • 
                <span class="profit-negative">${bet.profitLoss.toFixed(2)}€</span>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html || '<div class="empty-state">Aucun pari complété</div>';
}

// Export bets to CSV
function exportToCSV() {
    const headers = ['Date', 'Joueur', 'Type', 'Pari', 'Ligne', 'Prédiction', 'R²', 'Edge', 'Mise', 'Cote', 'Résultat', 'Réel', 'P/L'];
    const rows = bets.map(bet => [
        bet.date,
        bet.playerName,
        bet.propType,
        bet.betSide,
        bet.line,
        bet.prediction,
        bet.r2,
        bet.edge,
        bet.stake,
        bet.odds,
        bet.result,
        bet.actualResult || '',
        bet.profitLoss
    ]);
    
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nba-bets-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
}