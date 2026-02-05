#!/usr/bin/env python3
"""
The Odds API Client
https://the-odds-api.com/
Prix: $10/mois pour 500 requêtes
"""

import requests
from datetime import datetime
import os

class OddsAPIClient:
    """
    Client pour récupérer les odds NBA en temps réel
    
    Plans:
    - FREE: 500 requêtes/mois
    - $10/mois: 5,000 requêtes/mois
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        
        # Bookmakers à suivre (en ordre de préférence)
        self.bookmakers = [
            'fanduel',      # FanDuel
            'draftkings',   # DraftKings  
            'betmgm',       # BetMGM
            'pointsbetus',  # PointsBet
            'williamhill_us', # William Hill
            'bovada',       # Bovada
            'betonlineag'   # BetOnline (comparaison)
        ]
        
    def get_nba_games(self):
        """
        Récupère les matchs NBA disponibles aujourd'hui
        """
        if not self.api_key:
            print("⚠️ ODDS_API_KEY manquant - utilise mode simulation")
            return self._simulate_games()
        
        try:
            url = f"{self.base_url}/sports/basketball_nba/odds/"
            
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american',
                'bookmakers': ','.join(self.bookmakers)
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ Erreur API: {response.status_code}")
                return self._simulate_games()
            
            data = response.json()
            
            # Affiche le nombre de requêtes restantes
            remaining = response.headers.get('x-requests-remaining', 'N/A')
            used = response.headers.get('x-requests-used', 'N/A')
            print(f"📊 Requêtes: {used} utilisées, {remaining} restantes")
            
            return self._parse_games(data)
            
        except Exception as e:
            print(f"❌ Erreur The Odds API: {e}")
            return self._simulate_games()
    
    def _parse_games(self, data):
        """Parse les données de l'API"""
        games = []
        
        for game in data:
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = game['commence_time']
            
            # Extrait les bookmakers
            bookmakers_data = {}
            
            for bookmaker in game.get('bookmakers', []):
                bm_name = bookmaker['key']
                
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':  # Moneyline
                        bookmakers_data[bm_name] = {
                            'home_odds': market['outcomes'][0]['price'],
                            'away_odds': market['outcomes'][1]['price']
                        }
            
            games.append({
                'home_team': self._normalize_team(home_team),
                'away_team': self._normalize_team(away_team),
                'commence_time': commence_time,
                'bookmakers': bookmakers_data
            })
        
        return games
    
    def get_player_props(self):
        """
        Récupère les props de joueurs (points, assists, rebounds)
        """
        if not self.api_key:
            return self._simulate_player_props()
        
        try:
            url = f"{self.base_url}/sports/basketball_nba/odds/"
            
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'player_points,player_assists,player_rebounds',
                'oddsFormat': 'american',
                'bookmakers': ','.join(self.bookmakers)
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return self._simulate_player_props()
            
            data = response.json()
            return self._parse_player_props(data)
            
        except Exception as e:
            print(f"❌ Erreur props: {e}")
            return self._simulate_player_props()
    
    def _parse_player_props(self, data):
        """Parse les props de joueurs"""
        props = []
        
        for game in data:
            home_team = self._normalize_team(game['home_team'])
            away_team = self._normalize_team(game['away_team'])
            
            for bookmaker in game.get('bookmakers', []):
                bm_name = bookmaker['key']
                
                for market in bookmaker.get('markets', []):
                    market_type = market['key']  # player_points, player_assists, etc.
                    
                    # Mapping
                    stat_mapping = {
                        'player_points': 'points',
                        'player_assists': 'assists', 
                        'player_rebounds': 'rebounds'
                    }
                    
                    stat_type = stat_mapping.get(market_type)
                    if not stat_type:
                        continue
                    
                    for outcome in market.get('outcomes', []):
                        player_name = outcome.get('description', '')
                        point = outcome.get('point')  # La ligne (ex: 25.5)
                        
                        # Over/Under
                        over_odds = None
                        under_odds = None
                        
                        if outcome['name'] == 'Over':
                            over_odds = outcome['price']
                        elif outcome['name'] == 'Under':
                            under_odds = outcome['price']
                        
                        if player_name and point:
                            props.append({
                                'player': player_name,
                                'stat_type': stat_type,
                                'line': float(point),
                                'over_odds': over_odds,
                                'under_odds': under_odds,
                                'bookmaker': bm_name,
                                'home_team': home_team,
                                'away_team': away_team
                            })
        
        return props
    
    def _normalize_team(self, team_name):
        """Normalise les noms d'équipes"""
        team_map = {
            'Los Angeles Lakers': 'LAL',
            'Golden State Warriors': 'GSW',
            'Boston Celtics': 'BOS',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Phoenix Suns': 'PHX',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Los Angeles Clippers': 'LAC',
            'Philadelphia 76ers': 'PHI',
            'Brooklyn Nets': 'BKN',
            'Atlanta Hawks': 'ATL',
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Detroit Pistons': 'DET',
            'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND',
            'Memphis Grizzlies': 'MEM',
            'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL',
            'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA',
            'Washington Wizards': 'WAS',
            'Charlotte Hornets': 'CHA'
        }
        return team_map.get(team_name, team_name)
    
    def _simulate_games(self):
        """Simule des matchs pour testing"""
        return [
            {
                'home_team': 'LAL',
                'away_team': 'GSW',
                'commence_time': datetime.now().isoformat(),
                'bookmakers': {
                    'fanduel': {'home_odds': -150, 'away_odds': 130},
                    'draftkings': {'home_odds': -145, 'away_odds': 125}
                }
            }
        ]
    
    def _simulate_player_props(self):
        """Simule des props pour testing"""
        return [
            {
                'player': 'LeBron James',
                'stat_type': 'points',
                'line': 25.5,
                'over_odds': -110,
                'under_odds': -110,
                'bookmaker': 'fanduel',
                'home_team': 'LAL',
                'away_team': 'GSW'
            },
            {
                'player': 'Stephen Curry',
                'stat_type': 'points',
                'line': 27.5,
                'over_odds': -115,
                'under_odds': -105,
                'bookmaker': 'draftkings',
                'home_team': 'GSW',
                'away_team': 'LAL'
            },
            {
                'player': 'LeBron James',
                'stat_type': 'assists',
                'line': 7.5,
                'over_odds': -110,
                'under_odds': -110,
                'bookmaker': 'fanduel',
                'home_team': 'LAL',
                'away_team': 'GSW'
            }
        ]
    
    def get_usage_stats(self):
        """Récupère les stats d'utilisation de l'API"""
        if not self.api_key:
            return {'status': 'API key manquant', 'used': 0, 'remaining': 500}
        
        try:
            # Fait une petite requête pour récupérer les headers
            url = f"{self.base_url}/sports/basketball_nba/odds/"
            params = {
                'apiKey': self.api_key,
                'regions': 'us'
            }
            response = requests.get(url, params=params, timeout=5)
            
            return {
                'status': 'ok',
                'used': response.headers.get('x-requests-used', 'N/A'),
                'remaining': response.headers.get('x-requests-remaining', 'N/A')
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


if __name__ == '__main__':
    print("🎲 Test The Odds API Client")
    print("=" * 60)
    
    client = OddsAPIClient()
    
    print("\n📊 Matchs NBA disponibles:")
    games = client.get_nba_games()
    for game in games[:3]:
        print(f"   {game['away_team']} @ {game['home_team']}")
    
    print(f"\n🎯 Props disponibles:")
    props = client.get_player_props()
    for prop in props[:5]:
        print(f"   {prop['player']}: {prop['stat_type']} O/U {prop['line']}")
    
    print("\n📈 Usage API:")
    stats = client.get_usage_stats()
    print(f"   {stats}")
