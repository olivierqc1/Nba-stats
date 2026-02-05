from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

try:
    from nba_api.stats.endpoints import playergamelog, commonteamroster, teamgamelog
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True
except ImportError as e:
    NBA_API_AVAILABLE = False

class NBAAnalyzerV3:
    def __init__(self):
        self.current_season = '2024-25'
        self.defensive_ratings = {
            'CLE': 108.2, 'OKC': 109.1, 'ORL': 108.6, 'MIN': 108.9, 'MIA': 111.2,
            'BOS': 110.5, 'NYK': 110.7, 'HOU': 110.9, 'DEN': 111.3, 'MIL': 112.4,
            'LAC': 111.8, 'MEM': 112.6, 'PHI': 113.1, 'GSW': 112.1, 'DAL': 112.7,
            'SAC': 114.9, 'NOP': 114.3, 'PHX': 114.2, 'BKN': 114.8, 'LAL': 113.4,
            'CHI': 113.9, 'IND': 116.5, 'TOR': 115.4, 'ATL': 115.2, 'UTA': 115.3,
            'DET': 115.8, 'POR': 115.7, 'SAS': 116.0, 'CHA': 116.1, 'WAS': 116.8
        }
    
    def get_player_id(self, player_name):
        if not NBA_API_AVAILABLE:
            return None
        try:
            all_players = players.get_players()
            player = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
            if player:
                return player[0]['id']
            return None
        except:
            return None
    
    def get_season_games(self, player_id):
        if not NBA_API_AVAILABLE or not player_id:
            return None
        try:
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=self.current_season)
            df = gamelog.get_data_frames()[0]
            if df.empty:
                return None
            df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
            df['OPPONENT'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*([A-Z]{3})')[0]
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            return df
        except:
            return None
    
    def calculate_weighted_average(self, games_df):
        if games_df is None or len(games_df) == 0:
            return 0, 0
        points = games_df['PTS'].values
        n = len(points)
        weights = np.zeros(n)
        if n >= 10:
            weights[:10] = 0.5 / 10
        else:
            weights[:n] = 0.5 / n
        if n > 10:
            end_idx = min(30, n)
            count = end_idx - 10
            weights[10:end_idx] = 0.3 / count
        if n > 30:
            count = n - 30
            weights[30:] = 0.2 / count
        weights = weights / weights.sum()
        weighted_avg = np.average(points, weights=weights)
        weighted_std = np.sqrt(np.average((points - weighted_avg)**2, weights=weights))
        return float(weighted_avg), float(weighted_std)
    
    def calculate_splits(self, games_df, opponent=None):
        if games_df is None or len(games_df) == 0:
            return {}
        splits = {}
        home_games = games_df[games_df['IS_HOME'] == True]
        if len(home_games) > 0:
            splits['home'] = {
                'avg': round(float(home_games['PTS'].mean()), 1),
                'std': round(float(home_games['PTS'].std()), 1),
                'games': int(len(home_games))
            }
        away_games = games_df[games_df['IS_HOME'] == False]
        if len(away_games) > 0:
            splits['away'] = {
                'avg': round(float(away_games['PTS'].mean()), 1),
                'std': round(float(away_games['PTS'].std()), 1),
                'games': int(len(away_games))
            }
        return splits
    
    def calculate_trend_with_r2(self, games_df, num_games=10):
        if games_df is None or len(games_df) < 5:
            return {'slope': 0.0, 'r_squared': 0.0, 'p_value': 1.0, 'interpretation': 'Données insuffisantes', 'reliable': False}
        try:
            recent = games_df.head(num_games)
            n = len(recent)
            X = np.arange(n).reshape(-1, 1)
            y = recent['PTS'].values
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            residuals = y - y_pred
            s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
            x_var = np.sum((X - X.mean())**2)
            se_slope = s_err / np.sqrt(x_var) if x_var > 0 else 1
            t_stat = model.coef_[0] / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            slope = float(model.coef_[0])
            if r2 < 0.3:
                interpretation = "Tendance non fiable"
                reliable = False
            elif r2 < 0.6:
                interpretation = "Tendance modérée"
                reliable = p_value < 0.05
            else:
                interpretation = "Tendance forte"
                reliable = True
            return {'slope': round(slope, 2), 'r_squared': round(float(r2), 3), 'p_value': round(float(p_value), 4), 'interpretation': interpretation, 'reliable': reliable, 'sample_size': n}
        except:
            return {'slope': 0.0, 'r_squared': 0.0, 'p_value': 1.0, 'interpretation': 'Erreur', 'reliable': False}
    
    def adjust_for_matchup(self, base_prediction, opponent, is_home, splits):
        adjusted = base_prediction
        avg_rating = 113.0
        opp_rating = self.defensive_ratings.get(opponent, avg_rating)
        defense_factor = opp_rating / avg_rating
        adjusted *= defense_factor
        if is_home and 'home' in splits:
            home_diff = splits['home']['avg'] - base_prediction
            adjusted += home_diff * 0.5
        elif not is_home and 'away' in splits:
            away_diff = splits['away']['avg'] - base_prediction
            adjusted += away_diff * 0.5
        return float(adjusted)
    
    def predict_points(self, player_name, opponent, is_home=True, line=None):
        if not NBA_API_AVAILABLE:
            return {'error': 'nba_api not available', 'status': 'API_UNAVAILABLE'}
        player_id = self.get_player_id(player_name)
        if not player_id:
            return {'error': f'Player not found: {player_name}', 'status': 'PLAYER_NOT_FOUND'}
        season_games = self.get_season_games(player_id)
        if season_games is None or len(season_games) < 10:
            return {'error': 'Not enough games', 'status': 'INSUFFICIENT_DATA'}
        weighted_avg, weighted_std = self.calculate_weighted_average(season_games)
        splits = self.calculate_splits(season_games, opponent)
        trend_stats = self.calculate_trend_with_r2(season_games)
        base_prediction = weighted_avg
        if trend_stats['reliable']:
            base_prediction += (trend_stats['slope'] * 1.5)
        final_prediction = self.adjust_for_matchup(base_prediction, opponent, is_home, splits)
        n = len(season_games)
        se = weighted_std / np.sqrt(n)
        confidence_interval = stats.t.interval(0.95, n - 1, loc=final_prediction, scale=se)
        recommendation = None
        over_probability = None
        edge = None
        if line is not None:
            z_score = (line - final_prediction) / weighted_std if weighted_std > 0 else 0
            over_probability = 1 - stats.norm.cdf(z_score)
            edge = over_probability - 0.5
            if over_probability >= 0.58 and edge >= 0.08:
                recommendation = 'OVER'
            elif over_probability <= 0.42 and edge <= -0.08:
                recommendation = 'UNDER'
            else:
                recommendation = 'SKIP'
        result = {
            'player': player_name,
            'opponent': opponent,
            'is_home': is_home,
            'prediction': round(final_prediction, 1),
            'confidence_interval': {
                'lower': round(float(confidence_interval[0]), 1),
                'upper': round(float(confidence_interval[1]), 1),
                'confidence_level': '95%',
                'width': round(float(confidence_interval[1] - confidence_interval[0]), 1)
            },
            'season_stats': {
                'weighted_avg': round(weighted_avg, 1),
                'std_dev': round(weighted_std, 1),
                'games_played': len(season_games),
                'games_used': len(season_games),
                'consistency_level': 'Excellent' if weighted_std < 3 else 'Bon' if weighted_std < 5 else 'Moyen'
            },
            'trend_analysis': trend_stats,
            'splits': splits,
            'timestamp': datetime.now().isoformat()
        }
        if line is not None:
            result['line_analysis'] = {
                'bookmaker_line': float(line),
                'over_probability': round(float(over_probability), 3),
                'under_probability': round(float(1 - over_probability), 3),
                'edge': round(float(edge), 3),
                'recommendation': recommendation,
                'bet_confidence': 'HIGH' if abs(edge) >= 0.12 else 'MEDIUM' if abs(edge) >= 0.08 else 'LOW',
                'kelly_criterion': round(float(edge * 2), 3) if abs(edge) >= 0.08 else 0.0
            }
        result['status'] = 'SUCCESS'
        return result

analyzer = NBAAnalyzerV3()


@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'NBA Betting Analyzer API v3.0', 'status': 'online', 'season': '2024-25'})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'nba_api': NBA_API_AVAILABLE, 'season': '2024-25'})

@app.route('/api/analyze', methods=['POST'])
def analyze_player():
    try:
        data = request.json
        player = data.get('player')
        opponent = data.get('opponent')
        is_home = data.get('is_home', True)
        line = data.get('line')
        if not player or not opponent:
            return jsonify({'error': 'Missing required fields'}), 400
        result = analyzer.predict_points(player, opponent, is_home, line)
        if result.get('status') == 'SUCCESS':
            return jsonify(result)
        else:
            return jsonify(result), 404
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'ERROR'}), 500

@app.route('/api/player-stats/<player_name>', methods=['GET'])
def get_player_stats(player_name):
    try:
        player_id = analyzer.get_player_id(player_name)
        if not player_id:
            return jsonify({'error': f'Player not found: {player_name}'}), 404
        season_games = analyzer.get_season_games(player_id)
        if season_games is None:
            return jsonify({'error': 'No games found'}), 404
        all_games_list = []
        for idx, row in season_games.iterrows():
            all_games_list.append({
                'game_number': len(season_games) - idx,
                'date': str(row['GAME_DATE'].date()),
                'matchup': str(row['MATCHUP']),
                'opponent': str(row['OPPONENT']),
                'points': float(row['PTS']),
                'minutes': float(row.get('MIN', 0)) if not pd.isna(row.get('MIN', 0)) else 0,
                'is_home': bool(row['IS_HOME']),
                'location': 'Domicile' if row['IS_HOME'] else 'Extérieur'
            })
        recent_10 = season_games.head(10)['PTS'].mean()
        recent_5 = season_games.head(5)['PTS'].mean()
        home_games = season_games[season_games['IS_HOME'] == True]
        away_games = season_games[season_games['IS_HOME'] == False]
        return jsonify({
            'player': player_name,
            'season': '2024-25',
            'total_games': int(len(season_games)),
            'date_range': {
                'first_game': str(season_games.iloc[-1]['GAME_DATE'].date()),
                'last_game': str(season_games.iloc[0]['GAME_DATE'].date())
            },
            'averages': {
                'season': round(float(season_games['PTS'].mean()), 1),
                'last_10': round(float(recent_10), 1),
                'last_5': round(float(recent_5), 1),
                'home': round(float(home_games['PTS'].mean()), 1) if len(home_games) > 0 else 0,
                'away': round(float(away_games['PTS'].mean()), 1) if len(away_games) > 0 else 0
            },
            'volatility': round(float(season_games['PTS'].std()), 2),
            'all_games': all_games_list,
            'status': 'SUCCESS'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'ERROR'}), 500



@app.route('/api/team-roster/<team_code>', methods=['GET'])
def get_team_roster(team_code):
    if not NBA_API_AVAILABLE:
        return jsonify({'error': 'NBA API not available'}), 503
    try:
        team_dict = {
            'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751, 'CHA': 1610612766,
            'CHI': 1610612741, 'CLE': 1610612739, 'DAL': 1610612742, 'DEN': 1610612743,
            'DET': 1610612765, 'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
            'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763, 'MIA': 1610612748,
            'MIL': 1610612749, 'MIN': 1610612750, 'NOP': 1610612740, 'NYK': 1610612752,
            'OKC': 1610612760, 'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
            'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'TOR': 1610612761,
            'UTA': 1610612762, 'WAS': 1610612764
        }
        team_id = team_dict.get(team_code.upper())
        if not team_id:
            return jsonify({'error': f'Invalid team code: {team_code}'}), 400
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2024-25')
        roster_df = roster.get_data_frames()[0]
        roster_list = []
        for _, player in roster_df.iterrows():
            player_name = str(player['PLAYER'])
            player_id_val = analyzer.get_player_id(player_name)
            volatility = None
            if player_id_val:
                games = analyzer.get_season_games(player_id_val)
                if games is not None and len(games) >= 5:
                    volatility = float(games['PTS'].std())
            roster_list.append({
                'name': player_name,
                'position': str(player.get('POSITION', 'N/A')),
                'number': str(player.get('NUM', '')),
                'volatility': round(volatility, 2) if volatility else None
            })
        roster_list.sort(key=lambda x: x['volatility'] if x['volatility'] else 999)
        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2024-25')
        games_df = gamelog.get_data_frames()[0]
        if len(games_df) == 0:
            return jsonify({'error': 'No games found'}), 404
        last_game = games_df.iloc[0]
        opponent_matchup = str(last_game['MATCHUP'])
        if '@' in opponent_matchup:
            opponent = opponent_matchup.split('@')[1].strip()
            is_home = False
            location = 'Extérieur'
        else:
            opponent = opponent_matchup.split('vs.')[1].strip() if 'vs.' in opponent_matchup else 'Unknown'
            is_home = True
            location = 'Domicile'
        return jsonify({
            'team': team_code,
            'roster': roster_list,
            'next_game': {
                'opponent': opponent,
                'is_home': is_home,
                'location': location,
                'last_game_date': str(last_game['GAME_DATE'])
            },
            'status': 'SUCCESS'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'ERROR'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
