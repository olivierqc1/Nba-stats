#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKTEST WALK-FORWARD - NBA Props
===================================
Méthode honnête:
- Walk-forward: on entraîne sur les N premiers matchs, on prédit le N+1ème
- Ligne bookmaker simulée = moyenne mobile pondérée (proxy réaliste)
- Pas de data leakage possible
- Rapport complet avec ROI, win rate, Sharpe ratio

Usage:
    python backtest_walkforward.py
    python backtest_walkforward.py --player "Luka Doncic" --stat points --season 2023-24
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports du projet
from advanced_data_collector import AdvancedDataCollector
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PLAYERS = [
    "LeBron James",
    "Stephen Curry",
    "Luka Doncic",
    "Jayson Tatum",
    "Nikola Jokic",
    "Giannis Antetokounmpo",
    "Kevin Durant",
    "Anthony Edwards"
]

STAT_TYPES = ['points', 'assists', 'rebounds']

STAT_COL_MAP = {
    'points': 'PTS',
    'assists': 'AST',
    'rebounds': 'REB',
    '3pt': 'FG3M'
}

# Paramètres de pari
ODDS = -110          # Cote américaine standard
VIG_BREAK_EVEN = 52.38  # % de victoires nécessaire pour break-even à -110
STAKE = 50           # Mise fixe en euros/dollars

# Paramètres du modèle
MIN_TRAIN_GAMES = 20     # Matchs minimum avant de commencer à prédire
MIN_EDGE = 5.0           # Edge minimum pour parier (%)
MIN_R2_THRESHOLD = 0.10  # R² minimum sur le train set pour parier


# ============================================================================
# SIMULATION DE LIGNE BOOKMAKER
# ============================================================================

def simulate_bookmaker_line(historical_values, window=10):
    """
    Simule une ligne de bookmaker réaliste.
    
    Les bookmakers utilisent une moyenne pondérée des performances récentes,
    pas l'actuel. C'est notre meilleure approximation sans données d'odds historiques.
    
    Méthode: moyenne pondérée des N derniers matchs + petit bruit (~0.5 pts)
    """
    if len(historical_values) < 3:
        return None
    
    recent = historical_values[-min(window, len(historical_values)):]
    
    # Poids décroissants (plus récent = plus de poids)
    weights = np.exp(np.linspace(0, 1, len(recent)))
    weights = weights / weights.sum()
    
    weighted_avg = np.average(recent, weights=weights)
    
    # Arrondi au 0.5 le plus proche (comme les vrais bookmakers)
    rounded = round(weighted_avg * 2) / 2
    
    # Petit bruit aléatoire (±0.5) pour simuler le juice/ajustement du book
    noise = np.random.choice([-0.5, 0, 0.5], p=[0.25, 0.5, 0.25])
    
    return rounded + noise


# ============================================================================
# MODÈLE XGBoost RAPIDE POUR BACKTEST
# ============================================================================

def train_quick_model(X_train, y_train, feature_cols):
    """Entraîne un modèle XGBoost rapide"""
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbosity': 0
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def get_feature_cols(df, target_col):
    """Retourne les colonnes de features disponibles"""
    if target_col == 'PTS':
        trend_col = 'recent_trend_pts'
        avg_cols = ['avg_pts_last_5', 'avg_pts_last_10']
    elif target_col == 'AST':
        trend_col = 'recent_trend_ast'
        avg_cols = ['avg_ast_last_5', 'avg_ast_last_10']
    elif target_col == 'REB':
        trend_col = 'recent_trend_reb'
        avg_cols = ['avg_reb_last_5', 'avg_reb_last_10']
    else:
        trend_col = 'recent_trend_fg3m'
        avg_cols = ['avg_fg3m_last_5', 'avg_fg3m_last_10']
    
    candidates = avg_cols + [
        'home', 'rest_days', 'minutes_avg',
        'opponent_def_rating', 'pace',
        'usage_rate', 'back_to_back', trend_col
    ]
    
    return [c for c in candidates if c in df.columns]


# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def backtest_player(player_name, stat_type='points', season='2023-24',
                    min_edge=MIN_EDGE, stake=STAKE, verbose=True):
    """
    Walk-forward backtest pour un joueur/stat.
    
    Pour chaque match i (à partir du MIN_TRAIN_GAMES-ième):
    1. Entraîne sur les matchs 0..i-1
    2. Simule une ligne bookmaker basée sur l'historique 0..i-1
    3. Prédit le match i
    4. Compare avec le résultat réel
    5. Enregistre si on aurait parié (edge >= min_edge) et si on aurait gagné
    """
    
    if verbose:
        print(f"\n{'='*65}")
        print(f"🎯 BACKTEST: {player_name} — {stat_type.upper()} ({season})")
        print(f"{'='*65}")
    
    # 1. Collecte données
    collector = AdvancedDataCollector()
    df = collector.get_complete_player_data(player_name, season)
    
    if df is None or len(df) < MIN_TRAIN_GAMES + 10:
        return {
            'status': 'ERROR',
            'message': f'Pas assez de données: {len(df) if df is not None else 0} matchs',
            'player': player_name,
            'stat_type': stat_type
        }
    
    # 2. Remet en ordre chronologique
    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    
    target_col = STAT_COL_MAP[stat_type]
    feature_cols = get_feature_cols(df, target_col)
    
    if len(feature_cols) < 5:
        return {
            'status': 'ERROR',
            'message': f'Pas assez de features: {feature_cols}',
            'player': player_name,
            'stat_type': stat_type
        }
    
    if verbose:
        print(f"   📊 {len(df)} matchs | {len(feature_cols)} features")
        print(f"   🎲 Walk-forward à partir du match #{MIN_TRAIN_GAMES}")
    
    # 3. Walk-forward
    bets = []
    predictions_log = []
    
    np.random.seed(42)  # Reproductibilité de la simulation de ligne
    
    for i in range(MIN_TRAIN_GAMES, len(df)):
        
        # Données disponibles AVANT ce match
        df_train = df.iloc[:i].copy()
        
        # Features et target pour le train
        X_train = df_train[feature_cols].dropna()
        y_train = df_train.loc[X_train.index, target_col]
        
        if len(X_train) < 10:
            continue
        
        # Résultat réel du match qu'on prédit
        actual = float(df.iloc[i][target_col])
        game_date = df.iloc[i]['GAME_DATE']
        is_home = int(df.iloc[i].get('home', 0))
        
        # Ligne simulée du bookmaker (basée sur historique AVANT ce match)
        historical_vals = df_train[target_col].values
        line = simulate_bookmaker_line(historical_vals)
        
        if line is None:
            continue
        
        # Entraîne le modèle
        try:
            model = train_quick_model(X_train, y_train, feature_cols)
        except Exception:
            continue
        
        # Évalue la qualité du modèle sur train (proxy pour décider de parier)
        y_pred_train = model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred_train)
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        
        # Prépare features pour la prédiction
        X_pred = df.iloc[i:i+1][feature_cols].copy()
        X_pred = X_pred.fillna(X_train.mean())
        
        if X_pred.isna().any().any():
            continue
        
        # Prédiction
        prediction = float(model.predict(X_pred)[0])
        
        # Calcule edge
        std = train_rmse if train_rmse > 0 else 3.0
        
        from scipy import stats as scipy_stats
        z_score = (line - prediction) / std
        over_prob = (1 - scipy_stats.norm.cdf(z_score)) * 100
        under_prob = 100 - over_prob
        
        if over_prob > VIG_BREAK_EVEN:
            edge = over_prob - VIG_BREAK_EVEN
            recommendation = 'OVER'
            win_condition = actual > line
        elif under_prob > VIG_BREAK_EVEN:
            edge = under_prob - VIG_BREAK_EVEN
            recommendation = 'UNDER'
            win_condition = actual < line
        else:
            edge = 0
            recommendation = 'SKIP'
            win_condition = False
        
        # Log toutes les prédictions (même SKIP)
        predictions_log.append({
            'date': game_date,
            'actual': actual,
            'prediction': prediction,
            'line': line,
            'edge': edge,
            'recommendation': recommendation,
            'train_r2': train_r2,
            'error': abs(actual - prediction)
        })
        
        # On ne parie que si:
        # 1. Edge >= min_edge
        # 2. R² du modèle sur train >= seuil minimum
        # 3. Recommandation != SKIP
        if recommendation == 'SKIP' or edge < min_edge:
            continue
        
        if train_r2 < MIN_R2_THRESHOLD:
            continue
        
        # Calcule profit/loss
        if win_condition:
            profit = stake * (100 / abs(ODDS))  # ODDS = -110 → profit = stake * 0.909
        else:
            profit = -stake
        
        bets.append({
            'date': game_date,
            'actual': actual,
            'prediction': round(prediction, 1),
            'line': line,
            'edge': round(edge, 1),
            'recommendation': recommendation,
            'won': win_condition,
            'profit': round(profit, 2),
            'train_r2': round(train_r2, 3),
            'train_rmse': round(train_rmse, 2)
        })
    
    # =========================================================================
    # RÉSULTATS
    # =========================================================================
    
    if len(bets) == 0:
        return {
            'status': 'NO_BETS',
            'message': f'0 pari avec edge >= {min_edge}% et R² >= {MIN_R2_THRESHOLD}',
            'player': player_name,
            'stat_type': stat_type,
            'predictions_log': predictions_log
        }
    
    bets_df = pd.DataFrame(bets)
    
    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets * 100
    
    total_profit = bets_df['profit'].sum()
    total_staked = total_bets * stake
    roi = total_profit / total_staked * 100
    
    # Série actuelle
    current_streak = 0
    for won in reversed(bets_df['won'].tolist()):
        if won == bets_df['won'].iloc[-1]:
            current_streak += 1
        else:
            break
    
    # Sharpe ratio (mesure de qualité de la série)
    returns = bets_df['profit'] / stake
    sharpe = returns.mean() / returns.std() * np.sqrt(total_bets) if returns.std() > 0 else 0
    
    # Accuracy du modèle (prédiction dans ±3 pts de l'actuel)
    pred_log_df = pd.DataFrame(predictions_log)
    accuracy_3 = (pred_log_df['error'] <= 3).mean() * 100 if len(pred_log_df) > 0 else 0
    avg_error = pred_log_df['error'].mean() if len(pred_log_df) > 0 else 0
    
    result = {
        'status': 'SUCCESS',
        'player': player_name,
        'stat_type': stat_type,
        'season': season,
        # Performance paris
        'total_bets': total_bets,
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': round(win_rate, 1),
        'total_profit': round(total_profit, 2),
        'roi': round(roi, 2),
        'sharpe': round(sharpe, 2),
        # Qualité des prédictions
        'avg_error': round(avg_error, 2),
        'accuracy_within_3': round(accuracy_3, 1),
        'avg_edge': round(bets_df['edge'].mean(), 1),
        'avg_train_r2': round(bets_df['train_r2'].mean(), 3),
        # Détails
        'bets': bets_df.to_dict('records'),
        'predictions_count': len(predictions_log)
    }
    
    if verbose:
        _print_results(result)
    
    return result


def _print_results(r):
    """Affiche les résultats formatés"""
    
    print(f"\n{'─'*65}")
    print(f"📊 RÉSULTATS BACKTEST WALK-FORWARD")
    print(f"{'─'*65}")
    
    profit_sign = '+' if r['total_profit'] >= 0 else ''
    roi_sign = '+' if r['roi'] >= 0 else ''
    
    print(f"\n💰 PERFORMANCE PARIS:")
    print(f"   Paris effectués : {r['total_bets']}")
    print(f"   Gagnés / Perdus : {r['wins']} W — {r['losses']} L")
    print(f"   Win rate        : {r['win_rate']:.1f}%  (break-even: {VIG_BREAK_EVEN:.1f}%)")
    print(f"   Profit total    : {profit_sign}{r['total_profit']:.2f}€  (à {STAKE}€/pari)")
    print(f"   ROI             : {roi_sign}{r['roi']:.2f}%")
    print(f"   Sharpe ratio    : {r['sharpe']:.2f}")
    
    print(f"\n🎯 QUALITÉ DES PRÉDICTIONS:")
    print(f"   Erreur moyenne  : {r['avg_error']:.2f} pts")
    print(f"   Précision ±3pts : {r['accuracy_within_3']:.1f}%")
    print(f"   Edge moyen      : {r['avg_edge']:.1f}%")
    print(f"   R² moyen (train): {r['avg_train_r2']:.3f}")
    
    # Verdict
    print(f"\n{'─'*65}")
    if r['win_rate'] > VIG_BREAK_EVEN + 3:
        verdict = "✅ EDGE POSITIF DÉTECTÉ"
        detail = "Ce joueur/stat combo montre un vrai signal."
    elif r['win_rate'] > VIG_BREAK_EVEN:
        verdict = "⚡ EDGE MARGINAL"
        detail = "Légèrement au-dessus du break-even. Plus de données nécessaires."
    elif r['total_bets'] < 30:
        verdict = "❓ ÉCHANTILLON TROP PETIT"
        detail = f"Seulement {r['total_bets']} paris — impossible de conclure."
    else:
        verdict = "❌ PAS D'EDGE DÉTECTÉ"
        detail = "Le modèle ne bat pas le vig sur ce combo."
    
    print(f"   VERDICT: {verdict}")
    print(f"   {detail}")
    print(f"{'─'*65}")


# ============================================================================
# RAPPORT MULTI-JOUEURS
# ============================================================================

def run_full_backtest(players=None, stats=None, season='2023-24',
                      min_edge=MIN_EDGE, stake=STAKE):
    """
    Lance le backtest sur plusieurs joueurs/stats et produit un rapport complet.
    """
    
    if players is None:
        players = DEFAULT_PLAYERS
    if stats is None:
        stats = ['points']
    
    print("\n" + "="*65)
    print("🚀 BACKTEST COMPLET — NBA PROPS ANALYZER")
    print(f"   Saison: {season} | Edge min: {min_edge}% | Mise: {stake}€")
    print("="*65)
    
    all_results = []
    
    for player in players:
        for stat in stats:
            result = backtest_player(player, stat, season, min_edge, stake)
            if result['status'] in ['SUCCESS', 'NO_BETS']:
                all_results.append(result)
    
    # =========================================================================
    # RAPPORT GLOBAL
    # =========================================================================
    
    success_results = [r for r in all_results if r['status'] == 'SUCCESS']
    
    if not success_results:
        print("\n❌ Aucun résultat exploitable.")
        return all_results
    
    print(f"\n\n{'='*65}")
    print("📊 RAPPORT GLOBAL")
    print("="*65)
    
    total_bets = sum(r['total_bets'] for r in success_results)
    total_wins = sum(r['wins'] for r in success_results)
    total_profit = sum(r['total_profit'] for r in success_results)
    total_staked = total_bets * stake
    global_roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    global_wr = total_wins / total_bets * 100 if total_bets > 0 else 0
    avg_error = np.mean([r['avg_error'] for r in success_results])
    
    print(f"\n   Combos testés      : {len(success_results)}")
    print(f"   Total paris        : {total_bets}")
    print(f"   Win rate global    : {global_wr:.1f}%")
    print(f"   Profit total       : {'+' if total_profit >= 0 else ''}{total_profit:.2f}€")
    print(f"   ROI global         : {'+' if global_roi >= 0 else ''}{global_roi:.2f}%")
    print(f"   Erreur moy. préd.  : {avg_error:.2f} pts")
    
    # Tableau récap par joueur/stat
    print(f"\n{'─'*65}")
    print(f"{'Joueur':<22} {'Stat':<10} {'Paris':>5} {'WR%':>6} {'ROI%':>7} {'Verdict'}")
    print(f"{'─'*65}")
    
    for r in sorted(success_results, key=lambda x: x['roi'], reverse=True):
        wr = r['win_rate']
        roi = r['roi']
        n = r['total_bets']
        
        if n < 10:
            verdict = "❓ trop peu"
        elif wr > VIG_BREAK_EVEN + 3:
            verdict = "✅ edge+"
        elif wr > VIG_BREAK_EVEN:
            verdict = "⚡ marginal"
        else:
            verdict = "❌ no edge"
        
        print(f"{r['player']:<22} {r['stat_type']:<10} {n:>5} "
              f"{wr:>5.1f}% {roi:>+6.1f}%  {verdict}")
    
    print(f"{'─'*65}")
    
    # Conclusion
    print(f"\n📌 CONCLUSION:")
    
    edge_combos = [r for r in success_results 
                   if r['win_rate'] > VIG_BREAK_EVEN + 3 and r['total_bets'] >= 15]
    
    if edge_combos:
        print(f"   {len(edge_combos)} combo(s) avec edge détecté:")
        for r in edge_combos:
            print(f"   → {r['player']} ({r['stat_type']}): "
                  f"WR={r['win_rate']:.1f}%, ROI={r['roi']:+.1f}%")
    else:
        print("   Aucun combo ne montre un edge statistiquement solide.")
        print("   Options:")
        print("   1. Baisser le min_edge (essaie 3%)")
        print("   2. Tester d'autres joueurs/stats")
        print("   3. Vérifier que les vraies stats adversaires sont chargées")
    
    print(f"\n{'='*65}\n")
    
    # Sauvegarde CSV
    if success_results:
        all_bets = []
        for r in success_results:
            for bet in r['bets']:
                bet['player'] = r['player']
                bet['stat_type'] = r['stat_type']
                all_bets.append(bet)
        
        if all_bets:
            bets_df = pd.DataFrame(all_bets)
            filename = f"backtest_{season.replace('-','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            bets_df.to_csv(filename, index=False)
            print(f"   💾 Résultats sauvegardés: {filename}")
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Walk-forward backtest NBA props')
    parser.add_argument('--player', type=str, default=None,
                        help='Nom du joueur (défaut: 8 joueurs par défaut)')
    parser.add_argument('--stat', type=str, default='points',
                        choices=['points', 'assists', 'rebounds', '3pt'],
                        help='Type de stat')
    parser.add_argument('--season', type=str, default='2023-24',
                        help='Saison (ex: 2023-24)')
    parser.add_argument('--edge', type=float, default=MIN_EDGE,
                        help=f'Edge minimum % (défaut: {MIN_EDGE})')
    parser.add_argument('--stake', type=float, default=STAKE,
                        help=f'Mise par pari (défaut: {STAKE})')
    parser.add_argument('--all', action='store_true',
                        help='Teste tous les joueurs par défaut')
    
    args = parser.parse_args()
    
    if args.player:
        # Test un seul joueur
        result = backtest_player(
            args.player, args.stat, args.season, args.edge, args.stake
        )
    else:
        # Rapport complet
        run_full_backtest(
            players=DEFAULT_PLAYERS if args.all else DEFAULT_PLAYERS[:3],
            stats=[args.stat],
            season=args.season,
            min_edge=args.edge,
            stake=args.stake)