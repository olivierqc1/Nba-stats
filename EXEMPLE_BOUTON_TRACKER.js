// ========================================
// EXEMPLE D'INTÉGRATION - BOUTON "TRACKER CE PARI"
// À ajouter dans script.js (fonction createOpportunityCard ou similaire)
// ========================================

// FONCTION À AJOUTER À LA FIN DE script.js
function addToTracker(player, propType, side, line, prediction, r2, edge, game) {
    // Prepare bet data
    const betData = {
        playerName: player,
        propType: propType,
        betSide: side,
        line: line,
        prediction: prediction,
        r2: r2,
        edge: edge,
        game: game,
        stake: 0,  // L'utilisateur entrera ceci dans le tracker
        odds: -110  // Valeur par défaut
    };
    
    // Store in localStorage temporarily
    localStorage.setItem('pendingBet', JSON.stringify(betData));
    
    // Redirect to tracker with auto-fill
    window.location.href = 'bet-tracker.html?autofill=true';
}

// ========================================
// EXEMPLE D'AJOUT DU BOUTON DANS UNE CARTE
// ========================================

function createOpportunityCard(opportunity) {
    const { player, propType, side, line, prediction, r2, edge, game } = opportunity;
    
    // ... ton code existant pour créer la carte ...
    
    // AJOUTE CE HTML DANS TA CARTE (probablement avant </div> final)
    const trackingButtonHTML = `
        <div class="tracking-action" style="margin-top: 15px; text-align: center;">
            <button 
                onclick="addToTracker('${player}', '${propType}', '${side}', ${line}, ${prediction}, ${r2}, ${edge}, '${game}')"
                style="
                    width: 100%;
                    padding: 12px;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.3s;
                "
                onmouseover="this.style.transform='scale(1.02)'"
                onmouseout="this.style.transform='scale(1)'"
            >
                ➕ Tracker ce pari
            </button>
        </div>
    `;
    
    // ... retourne ta carte avec le bouton ...
}

// ========================================
// ALTERNATIVE: SI TU UTILISES JQUERY
// ========================================

$(document).on('click', '.btn-track', function() {
    const $card = $(this).closest('.opportunity-card');
    
    const betData = {
        playerName: $card.data('player'),
        propType: $card.data('prop-type'),
        betSide: $card.data('side'),
        line: parseFloat($card.data('line')),
        prediction: parseFloat($card.data('prediction')),
        r2: parseFloat($card.data('r2')),
        edge: parseFloat($card.data('edge')),
        game: $card.data('game'),
        stake: 0,
        odds: -110
    };
    
    localStorage.setItem('pendingBet', JSON.stringify(betData));
    window.location.href = 'bet-tracker.html?autofill=true';
});

// ========================================
// ALTERNATIVE: HTML AVEC DATA-ATTRIBUTES
// ========================================

// Dans ton HTML de carte:
<div class="opportunity-card" 
     data-player="LeBron James"
     data-prop-type="PTS"
     data-side="OVER"
     data-line="25.5"
     data-prediction="28.3"
     data-r2="0.82"
     data-edge="35.8"
     data-game="LAL @ GSW">
    
    <!-- Contenu de la carte -->
    
    <button class="btn-track">➕ Tracker ce pari</button>
</div>

// Dans script.js:
document.querySelectorAll('.btn-track').forEach(btn => {
    btn.addEventListener('click', function() {
        const card = this.closest('.opportunity-card');
        const betData = {
            playerName: card.dataset.player,
            propType: card.dataset.propType,
            betSide: card.dataset.side,
            line: parseFloat(card.dataset.line),
            prediction: parseFloat(card.dataset.prediction),
            r2: parseFloat(card.dataset.r2),
            edge: parseFloat(card.dataset.edge),
            game: card.dataset.game,
            stake: 0,
            odds: -110
        };
        
        localStorage.setItem('pendingBet', JSON.stringify(betData));
        window.location.href = 'bet-tracker.html?autofill=true';
    });
});