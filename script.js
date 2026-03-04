// ============================================================
// NBA Analyzer — script.js
// Badge cache backtest intégré sur chaque carte joueur
// ============================================================

const BACKEND = 'https://nba-stats-xcyv.onrender.com';

// Cache backtest chargé au démarrage
let backtestCache = {};

// Charge le cache dès que la page s'ouvre
async function loadBacktestCache() {
  try {
    const res  = await fetch(`${BACKEND}/api/backtest-cache`);
    backtestCache = await res.json();
    console.log(`Cache backtest: ${Object.keys(backtestCache).length} joueurs`);
  } catch (e) {
    console.warn('Cache backtest non disponible');
  }
}
loadBacktestCache();

// ── Badge fiabilité ──────────────────────────────────────────
function getBadgeHtml(playerName, statType, inProgress = false) {
  if (inProgress) {
    return `<span class="bt-badge bt-grey" title="Backtest en cours...">⏳ En cours...</span>`;
  }
  const key  = playerName.toLowerCase().replace(/ /g, '_');
  const data = backtestCache[key]?.[statType];
  if (!data || data.total_bets < 10) {
    return `<span class="bt-badge bt-grey" title="Backtest automatique en attente">⬜ Non testé</span>`;
  }
  const wr = data.win_rate;
  if (wr >= 60) return `<span class="bt-badge bt-green" title="${data.total_bets} paris · ${data.season}">🟢 ${wr}% (${data.total_bets} paris)</span>`;
  if (wr >= 55) return `<span class="bt-badge bt-yellow" title="${data.total_bets} paris · ${data.season}">🟡 ${wr}% (${data.total_bets} paris)</span>`;
  return `<span class="bt-badge bt-red" title="${data.total_bets} paris · ${data.season}">🔴 ${wr}% (${data.total_bets} paris)</span>`;
}

// ── CSS badges (injecté une seule fois) ─────────────────────
(function injectBadgeStyles() {
  const s = document.createElement('style');
  s.textContent = `
    .bt-badge { display:inline-block; padding:4px 12px; border-radius:20px;
      font-size:0.78em; font-weight:700; margin-left:8px; vertical-align:middle; cursor:default; }
    .bt-green  { background:#d1fae5; color:#065f46; }
    .bt-yellow { background:#fef3c7; color:#92400e; }
    .bt-red    { background:#fee2e2; color:#991b1b; }
    .bt-grey   { background:#f3f4f6; color:#6b7280; }
  `;
  document.head.appendChild(s);
})();

// ── Scan ────────────────────────────────────────────────────
async function scanOpportunities(statType) {
  const minR2   = parseFloat(document.getElementById('minR2').value);
  const minEdge = parseFloat(document.getElementById('minEdge').value);

  document.querySelectorAll('.btn-scan').forEach(b => b.disabled = true);
  document.getElementById('loadingDiv').classList.remove('hidden');
  document.getElementById('resultsDiv').classList.add('hidden');
  document.getElementById('errorDiv').classList.add('hidden');
  document.getElementById('statsBar').classList.add('hidden');

  try {
    const url = `${BACKEND}/api/daily-opportunities-${statType}?min_r2=${minR2}&min_edge=${minEdge}`;
    const res  = await fetch(url);
    const data = await res.json();

    document.getElementById('loadingDiv').classList.add('hidden');
    document.querySelectorAll('.btn-scan').forEach(b => b.disabled = false);

    if (data.opportunities && data.opportunities.length > 0) {
      displayOpportunities(data.opportunities, statType);
      document.getElementById('statsBar').classList.remove('hidden');
      document.getElementById('statType').textContent   = statType.toUpperCase();
      document.getElementById('analyzedProps').textContent = data.total_analyzed || '?';
      document.getElementById('foundOpps').textContent  = data.opportunities.length;
      const avgR2 = data.opportunities.reduce((s, o) => s + o.regression_stats.r_squared, 0) / data.opportunities.length;
      document.getElementById('avgR2').textContent = (avgR2 * 100).toFixed(1) + '%';
    } else {
      document.getElementById('errorDiv').classList.remove('hidden');
      document.getElementById('errorDiv').textContent = '❌ Aucune opportunité trouvée. Essaie de baisser les filtres.';
    }
  } catch (err) {
    document.getElementById('loadingDiv').classList.add('hidden');
    document.querySelectorAll('.btn-scan').forEach(b => b.disabled = false);
    document.getElementById('errorDiv').classList.remove('hidden');
    document.getElementById('errorDiv').textContent = `❌ Erreur: ${err.message}`;
  }
}

// ── Auto-backtest pour joueurs non testés ────────────────────
async function autoBacktestIfNeeded(playerName, statType) {
  const key  = playerName.toLowerCase().replace(/ /g, '_');
  const data = backtestCache[key]?.[statType];
  if (data && data.total_bets >= 10) return; // déjà en cache

  console.log(`Auto-backtest: ${playerName} / ${statType}`);

  try {
    const res  = await fetch(`${BACKEND}/api/backtest`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        player:    playerName,
        stat_type: statType,
        season:    '2023-24',   // saison complète = plus fiable
        min_edge:  5,
        stake:     50
      })
    });
    const result = await res.json();

    if (result.status === 'SUCCESS' && result.total_bets >= 10) {
      // Sauvegarde dans le cache backend
      await fetch(`${BACKEND}/api/backtest-cache`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          player:       result.player,
          stat_type:    result.stat_type,
          season:       result.season,
          win_rate:     result.win_rate,
          total_bets:   result.total_bets,
          roi:          result.roi,
          verdict:      result.verdict_label || '',
          verdict_color: result.verdict_color || 'grey',
          avg_error:    result.avg_error
        })
      });

      // Met à jour le cache local et rafraîchit le badge sur la carte
      if (!backtestCache[key]) backtestCache[key] = {};
      backtestCache[key][statType] = {
        player:    playerName,
        stat_type: statType,
        season:    result.season,
        win_rate:  result.win_rate,
        total_bets: result.total_bets,
        roi:       result.roi
      };

      // Rafraîchit le badge dans le DOM sans recharger la page
      document.querySelectorAll('.opp-player').forEach(el => {
        if (el.textContent.includes(playerName)) {
          const badgeEl = el.querySelector('.bt-badge');
          if (badgeEl) badgeEl.outerHTML = getBadgeHtml(playerName, statType);
        }
      });

      console.log(`✅ Auto-backtest OK: ${playerName} → ${result.win_rate}% (${result.total_bets} paris)`);
    }
  } catch (e) {
    console.warn(`Auto-backtest échoué pour ${playerName}:`, e.message);
  }
}

// ── Rendu des cartes ────────────────────────────────────────
function displayOpportunities(opportunities, statType) {
  const container = document.getElementById('resultsDiv');
  container.classList.remove('hidden');
  container.innerHTML = opportunities.map((opp, i) => buildCard(opp, i + 1, statType)).join('');

  // Lance auto-backtest en arrière-plan pour les joueurs non testés
  // Délai échelonné pour ne pas surcharger le serveur (un à la fois)
  opportunities.forEach((opp, i) => {
    const key  = opp.player.toLowerCase().replace(/ /g, '_');
    const data = backtestCache[key]?.[statType];
    if (!data || data.total_bets < 10) {
      setTimeout(() => {
        // Affiche "⏳ En cours" avant de lancer
        document.querySelectorAll('.opp-player').forEach(el => {
          if (el.textContent.includes(opp.player)) {
            const badgeEl = el.querySelector('.bt-badge');
            if (badgeEl) badgeEl.outerHTML = getBadgeHtml(opp.player, statType, true);
          }
        });
        autoBacktestIfNeeded(opp.player, statType);
      }, i * 30000);
    }
  });
}

function buildCard(opp, rank, statType) {
  const rec    = opp.line_analysis.recommendation;
  const isOver = rec === 'OVER';
  const line   = opp.line_analysis.bookmaker_line;
  const pred   = opp.prediction;
  const ci     = opp.confidence_interval;
  const stats  = opp.regression_stats;
  const lv     = opp.line_analysis.line_value_score || 0;
  const edge   = opp.line_analysis.edge;
  const player = opp.player;

  const badge = getBadgeHtml(player, statType);

  const kelly = opp.line_analysis.kelly_fraction
    ? `<div class="metric"><div class="metric-label">📈 Kelly</div><div class="metric-value">${(opp.line_analysis.kelly_fraction * 100).toFixed(1)}%</div></div>`
    : '';

  return `
    <div class="opp-card ${isOver ? '' : 'under'}" data-rank="${rank}">
      <div class="opp-header">
        <div class="opp-player" onclick="openPlayerModal('${player.replace(/'/g, "\\'")}')">
          ${player}${badge}
        </div>
        <div class="opp-matchup">
          ${opp.is_home ? '🏠' : '✈️'} vs ${opp.opponent} • ${statType.charAt(0).toUpperCase() + statType.slice(1)}
        </div>
        <span class="opp-badge ${isOver ? 'over' : 'under'}">${rec} ${line}</span>
      </div>

      <div class="prediction-box">
        <div class="pred-row">
          <div><div class="pred-label">Prédiction modèle</div><div class="pred-value">${pred}</div></div>
          <div><div class="pred-label">vs Ligne</div><div class="pred-value">${line}</div></div>
        </div>
        <div class="confidence-interval">
          <div class="confidence-interval-label">Intervalle de confiance 95%</div>
          <div class="confidence-interval-range">${ci.lower} - ${ci.upper}</div>
        </div>
        <div class="metrics-grid" style="margin-top:15px;">
          <div class="metric"><div class="metric-label">🎯 R² TEST</div><div class="metric-value">${(stats.r_squared * 100).toFixed(1)}%</div></div>
          <div class="metric"><div class="metric-label">📊 RMSE</div><div class="metric-value">${stats.rmse}</div></div>
          <div class="metric"><div class="metric-label">💰 Edge</div><div class="metric-value">+${edge.toFixed(1)}%</div></div>
          ${kelly}
          <div class="metric"><div class="metric-label">📉 Line Value</div><div class="metric-value">${lv > 0 ? '+' : ''}${lv.toFixed(2)}</div></div>
        </div>
      </div>

      <div class="action-box ${isOver ? '' : 'under'}">
        <div class="action-title">${isOver ? '✅ PARIE OVER' : '✅ PARIE UNDER'} ${line}</div>
        <div class="action-detail">Prédiction: <strong>${pred}</strong> pts | Edge: <strong>+${edge.toFixed(1)}%</strong></div>
      </div>
    </div>`;
}

// ── Modal joueur ─────────────────────────────────────────────
async function openPlayerModal(playerName) {
  document.getElementById('modalPlayerName').textContent = playerName;
  document.getElementById('playerModal').classList.add('open');
  document.getElementById('modalContent').innerHTML = '<div class="spinner" style="margin:40px auto;width:50px;height:50px;border:5px solid #f3f3f3;border-top:5px solid #667eea;border-radius:50%;animation:spin 1s linear infinite;"></div>';

  try {
    const res  = await fetch(`${BACKEND}/api/player-history/${encodeURIComponent(playerName)}`);
    const data = await res.json();
    renderPlayerModal(data, playerName);
  } catch (e) {
    document.getElementById('modalContent').innerHTML = '<p style="color:#ef4444;text-align:center;">Erreur de chargement</p>';
  }
}

function renderPlayerModal(data, playerName) {
  if (!data.games || data.games.length === 0) {
    document.getElementById('modalContent').innerHTML = '<p style="text-align:center;">Aucun match trouvé</p>';
    return;
  }

  const rows = data.games.slice(0, 10).map(g => `
    <tr>
      <td>${g.date || ''}</td>
      <td>${g.opponent || ''}</td>
      <td>${g.home ? '🏠' : '✈️'}</td>
      <td><strong>${g.pts ?? '-'}</strong></td>
      <td>${g.ast ?? '-'}</td>
      <td>${g.reb ?? '-'}</td>
      <td>${g.min ?? '-'}</td>
    </tr>`).join('');

  document.getElementById('modalContent').innerHTML = `
    <table class="prob-table">
      <thead><tr><th>Date</th><th>Adversaire</th><th>Lieu</th><th>PTS</th><th>AST</th><th>REB</th><th>MIN</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function closePlayerModal() {
  document.getElementById('playerModal').classList.remove('open');
}

// Ferme modal en cliquant hors
window.addEventListener('click', e => {
  if (e.target === document.getElementById('playerModal')) closePlayerModal();
});
