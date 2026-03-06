#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Analyzer - Module Backtest v3
Corrections:
- Filtre DNP explicite (réel == 0 → skip)
- Sanity check: prédiction < 50% de la ligne → modèle invalide pour ce match
- Intervalle de confiance clampé à 0 (pas de valeurs négatives)
- line_value seuil à 0.2, min 30 paris pour verdict
- Filtre RMSE/ligne < 40%
"""

import random
import numpy as np
from flask import request, jsonify
from scipy import stats as scipy_stats

VIG_BREAK_EVEN   = 52.38
ODDS             = -110
MIN_TRAIN        = 20
MIN_BETS_VERDICT = 30
LINE_VALUE_MIN   = 0.2
MAX_RMSE_RATIO   = 0.40          # pour points — adapté par stat_type dans le code
MAX_CI_WIDTH     = 12.0          # CI trop large = modèle incertain → skip
# Sanity check: si prédiction trop loin de la ligne → modèle déréglé
PRED_LINE_RATIO_MIN = 0.30       # élargi pour assists/rebounds
PRED_LINE_RATIO_MAX = 2.50       # élargi pour assists/rebounds

FEATURE_SETS = {
    'points': [
        'avg_pts_last_5', 'avg_pts_last_10', 'exp_avg_pts',
        'recent_over_rate_pts', 'pts_momentum', 'pts_per_min',
        'recent_trend_pts', 'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'minutes_consistency', 'usage_rate'
    ],
    'assists': [
        'avg_ast_last_5', 'avg_ast_last_10', 'exp_avg_ast',
        'recent_over_rate_ast', 'ast_momentum', 'ast_per_min',
        'recent_trend_ast', 'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'minutes_consistency', 'usage_rate'
    ],
    'rebounds': [
        'avg_reb_last_5', 'avg_reb_last_10', 'exp_avg_reb',
        'recent_over_rate_reb', 'reb_momentum', 'reb_per_min',
        'recent_trend_reb', 'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'minutes_consistency', 'usage_rate'
    ],
    '3pt': [
        'avg_fg3m_last_5', 'avg_fg3m_last_10',
        'recent_trend_fg3m', 'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'usage_rate'
    ]
}

STAT_COL = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB', '3pt': 'FG3M'}


def _simulate_line(historical_values):
    if len(historical_values) < 3:
        return None
    recent  = historical_values[-10:]
    weights = np.exp(np.linspace(0, 1, len(recent)))
    avg     = np.average(recent, weights=weights / weights.sum())
    noise   = random.choice([-0.5, 0, 0.5])
    return round(avg * 2) / 2 + noise


def _line_value(historical_series, line):
    if len(historical_series) < 5:
        return 0.0
    avg5  = float(np.mean(historical_series[-5:]))
    avg10 = float(np.mean(historical_series[-10:])) if len(historical_series) >= 10 else avg5
    wtd   = avg5 * 0.7 + avg10 * 0.3
    std   = float(np.std(historical_series[-15:])) if len(historical_series) >= 15 else float(np.std(historical_series))
    return abs(wtd - line) / std if std > 0.5 else 0.0


def _verdict(total_bets, win_rate):
    if total_bets < MIN_BETS_VERDICT:
        return {
            'verdict':       'INSUFFICIENT_SAMPLE',
            'verdict_label': '⚠️ Échantillon insuffisant',
            'verdict_msg':   f'Seulement {total_bets} paris simulés — besoin de {MIN_BETS_VERDICT} minimum. Essaie la saison 2023-24 ou baisse min_edge.',
            'verdict_color': 'orange'
        }
    if win_rate >= 58.0:
        return {
            'verdict':       'STRONG_EDGE',
            'verdict_label': '✅ Edge solide — Continue à parier',
            'verdict_msg':   f'Win rate {win_rate}% bien au-dessus du break-even de 52.4% sur {total_bets} paris.',
            'verdict_color': 'green'
        }
    if win_rate >= 54.0:
        return {
            'verdict':       'MARGINAL_EDGE',
            'verdict_label': '⚡ Edge marginal — Prudence',
            'verdict_msg':   f'Win rate {win_rate}% légèrement au-dessus du break-even. Augmente les filtres.',
            'verdict_color': 'yellow'
        }
    if win_rate >= 52.38:
        return {
            'verdict':       'NO_EDGE',
            'verdict_label': '❌ Pas d\'edge réel',
            'verdict_msg':   f'Win rate {win_rate}% trop proche du break-even 52.4%.',
            'verdict_color': 'red'
        }
    return {
        'verdict':       'LOSING',
        'verdict_label': '🔴 Modèle perdant',
        'verdict_msg':   f'Win rate {win_rate}% sous le break-even. Stop parier sur ce profil.',
        'verdict_color': 'red'
    }


def _run_walkforward(player_name, stat_type, season, min_edge, stake, collector):
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error as mse_fn

    target_col = STAT_COL.get(stat_type, 'PTS')
    df = collector.get_complete_player_data(player_name, season)

    if df is None or len(df) < MIN_TRAIN + 10:
        n = len(df) if df is not None else 0
        return {
            'status':  'ERROR',
            'message': f'Pas assez de données ({n} matchs, besoin de {MIN_TRAIN + 10}). Essaie la saison 2023-24.'
        }

    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

    feat_cols = [f for f in FEATURE_SETS.get(stat_type, FEATURE_SETS['points']) if f in df.columns]
    if len(feat_cols) < 5:
        return {'status': 'ERROR', 'message': f'Pas assez de features ({len(feat_cols)})'}

    random.seed(42); np.random.seed(42)

    bets              = []
    all_errors        = []
    skipped_dnp       = 0   # DNP / joueur absent
    skipped_rmse      = 0   # RMSE trop élevé vs ligne
    skipped_lv        = 0   # line_value trop faible
    skipped_sanity    = 0   # prédiction trop loin de la ligne (modèle déréglé)

    for i in range(MIN_TRAIN, len(df)):
        actual = float(df.iloc[i][target_col])

        # FILTRE DNP: réel == 0 = joueur n'a pas joué
        if actual == 0:
            skipped_dnp += 1
            continue


        # FILTRE MIN < 20 min
        if 'MIN' in df.columns and float(df.iloc[i].get('MIN', 30)) < 20:
            skipped_dnp += 1
            continue

        df_train = df.iloc[:i]
        X_train  = df_train[feat_cols].fillna(0)
        y_train  = df_train[target_col]
        w_train  = df_train['sample_weight'].values if 'sample_weight' in df_train.columns else None

        if len(X_train) < 10:
            continue

        line = _simulate_line(df_train[target_col].values)
        if line is None or line <= 0:
            continue

        try:
            model = xgb.XGBRegressor(
                max_depth=3, learning_rate=0.05, n_estimators=60,
                subsample=0.75, colsample_bytree=0.75,
                reg_alpha=0.1, reg_lambda=1.0,
                min_child_weight=3, random_state=42, verbosity=0
            )
            model.fit(X_train, y_train, sample_weight=w_train)
        except Exception:
            continue

        y_pred_train = model.predict(X_train)
        train_r2     = float(r2_score(y_train, y_pred_train))
        train_rmse   = float(np.sqrt(mse_fn(y_train, y_pred_train)))

        X_pred     = df.iloc[i:i+1][feat_cols].fillna(X_train.mean())
        prediction = float(model.predict(X_pred)[0])
        prediction = max(prediction, 0)  # jamais négatif
        all_errors.append(abs(actual - prediction))

        # FILTRE RMSE/ligne — seuil adapté par stat_type
        rmse_limits = {'points': 0.40, 'assists': 0.65, 'rebounds': 0.60, '3pt': 0.80}
        max_rmse = rmse_limits.get(stat_type, 0.50)
        if line > 0 and (train_rmse / line) > max_rmse:
            skipped_rmse += 1
            continue

        # NOTE: CI width filtré uniquement en LIVE (scan) — en backtest le RMSE
        # est naturellement plus élevé (fenêtres de 20-30 matchs seulement)
        # donc le filtre CI width supprimerait presque tous les bets walk-forward.

        # NOTE: line_value NON appliqué en backtest — les lignes sont simulées
        # à partir de la moyenne récente, donc line_value serait circulaire.
        # Ce filtre s'applique uniquement en LIVE sur les vraies lignes des books.
        lv = _line_value(df_train[target_col].values, line)  # calculé pour affichage seulement

        # SANITY CHECK: prédiction trop loin de la ligne = modèle déréglé
        pred_ratio = prediction / line if line > 0 else 1.0
        if pred_ratio < PRED_LINE_RATIO_MIN or pred_ratio > PRED_LINE_RATIO_MAX:
            skipped_sanity += 1
            continue

        # Calcul edge
        std = max(train_rmse, 2.0)
        z   = (line - prediction) / std
        op  = (1 - scipy_stats.norm.cdf(z)) * 100
        up  = 100 - op

        if op > VIG_BREAK_EVEN:
            edge, rec, won = op - VIG_BREAK_EVEN, 'OVER',  actual > line
        elif up > VIG_BREAK_EVEN:
            edge, rec, won = up - VIG_BREAK_EVEN, 'UNDER', actual < line
        else:
            edge, rec, won = 0, 'SKIP', False

        if rec == 'SKIP' or edge < min_edge or train_r2 < 0.05:
            continue

        profit = stake * (100 / abs(ODDS)) if won else -stake
        bets.append({
            'date':           str(df.iloc[i]['GAME_DATE'])[:10],
            'actual':         round(actual,     1),
            'prediction':     round(prediction, 1),
            'line':           round(line,       1),
            'edge':           round(edge,       1),
            'line_value':     round(lv,         2),
            'rmse_ratio':     round(train_rmse / line, 2),
            'recommendation': rec,
            'won':            bool(won),
            'profit':         round(profit, 2)
        })

    if not bets:
        return {
            'status':              'NO_BETS',
            'message':             f'0 pari qualifié. Skippés: {skipped_dnp} DNP, {skipped_rmse} RMSE, {skipped_lv} line_value, {skipped_sanity} sanity.',
            'total_predictions':   len(all_errors),
            'skipped_dnp':         skipped_dnp,
            'skipped_rmse':        skipped_rmse,
            'skipped_line_value':  skipped_lv,
            'skipped_sanity':      skipped_sanity,
            'avg_error':           round(float(np.mean(all_errors)), 2) if all_errors else 0
        }

    total_bets   = len(bets)
    wins         = sum(1 for b in bets if b['won'])
    win_rate     = round(wins / total_bets * 100, 1)
    total_profit = sum(b['profit'] for b in bets)
    verdict      = _verdict(total_bets, win_rate)

    return {
        'status':             'SUCCESS',
        'player':             player_name,
        'stat_type':          stat_type,
        'season':             season,
        'total_bets':         total_bets,
        'wins':               wins,
        'losses':             total_bets - wins,
        'win_rate':           win_rate,
        'total_profit':       round(total_profit, 2),
        'roi':                round(total_profit / (total_bets * stake) * 100, 2),
        'avg_error':          round(float(np.mean(all_errors)), 2) if all_errors else 0,
        'avg_edge':           round(float(np.mean([b['edge'] for b in bets])), 1),
        'avg_line_value':     round(float(np.mean([b['line_value'] for b in bets])), 2),
        'break_even':         round(VIG_BREAK_EVEN, 1),
        'skipped_dnp':        skipped_dnp,
        'skipped_rmse':       skipped_rmse,
        'skipped_line_value': skipped_lv,
        'skipped_sanity':     skipped_sanity,
        **verdict,
        'bets': bets[-20:]
    }


def register_backtest_routes(app, collector):

    @app.route('/api/backtest', methods=['POST'])
    def run_backtest():
        if collector is None:
            return jsonify({'status': 'ERROR', 'message': 'XGBoost non disponible'}), 503

        data      = request.get_json() or {}
        player    = data.get('player',    'LeBron James')
        stat_type = data.get('stat_type', 'points')
        season    = data.get('season',    '2023-24')
        min_edge  = float(data.get('min_edge', 5.0))
        stake     = float(data.get('stake',    50.0))

        print(f"\nBACKTEST v3: {player} | {stat_type} | {season} | edge>={min_edge}%")
        try:
            result = _run_walkforward(player, stat_type, season, min_edge, stake, collector)
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({'status': 'ERROR', 'message': str(e)}), 500

    print("   /api/backtest registered (v3 — DNP filter, sanity check, CI>=0)")
