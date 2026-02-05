#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Odds API Client - VERSION 2 JOURS
Récupère les matchs d'AUJOURD'HUI et DEMAIN
"""

import os
import requests
from datetime import datetime, timedelta

class OddsAPIClient:
    def __init__(self):
        self.api_key = os.environ.get('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not found in environment variables")
        
        self.base_url = 'https://api.the-odds-api.com/v4'
        self.sport = 'basketball_nba'
        
    def get_player_props(self, days=2):
        """
        Récupère les player props pour X jours
        
        Args:
            days (int): Nombre de jours (1 ou 2, max supporté par l'API gratuite)
        
        Returns:
            list: Liste de toutes les props trouvées
        """
        all_props = []
        
        # Aujourd'hui
        today_props = self._fetch_props_for_date(datetime.now())
        all_props.extend(today_props)
        
        # Demain (si days >= 2)
        if days >= 2:
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow_props = self._fetch_props_for_date(tomorrow)
            all_props.extend(tomorrow_props)
        
        print(f"Total props retrieved: {len(all_props)} (over {days} day(s))")
        
        return all_props
    
    def _fetch_props_for_date(self, date):
        """Récupère les props pour une date spécifique"""
        
        # Étape 1: Récupérer les matchs du jour
        games_url = f'{self.base_url}/sports/{self.sport}/odds'
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(games_url, params=params, timeout=10)
            response.raise_for_status()
            games = response.json()
            
            print(f"Games found for {date.strftime('%Y-%m-%d')}: {len(games)}")
            
            if not games:
                return []
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching games for {date.strftime('%Y-%m-%d')}: {e}")
            return []
        
        # Étape 2: Récupérer les player props
        props_url = f'{self.base_url}/sports/{self.sport}/events/{games[0]["id"]}/odds'
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'player_points,player_rebounds,player_assists',
            'oddsFormat': 'american'
        }
        
        all_props = []
        
        for game in games:
            event_id = game['id']
            
            try:
                props_response = requests.get(
                    f'{self.base_url}/sports/{self.sport}/events/{event_id}/odds',
                    params=params,
                    timeout=10
                )
                props_response.raise_for_status()
                event_data = props_response.json()
                
                # Parse les props
                props = self._parse_props(event_data, game, date)
                all_props.extend(props)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching props for game {event_id}: {e}")
                continue
        
        return all_props
    
    def _parse_props(self, event_data, game, date):
        """Parse les props d'un événement"""
        props = []
        
        home_team = game['home_team']
        away_team = game['away_team']
        game_time = game.get('commence_time', '')
        
        if 'bookmakers' not in event_data:
            return props
        
        for bookmaker in event_data['bookmakers']:
            bookmaker_name = bookmaker['key']
            
            for market in bookmaker.get('markets', []):
                market_type = market['key']
                
                # Mapper les markets vers nos stat_types
                stat_type_map = {
                    'player_points': 'points',
                    'player_rebounds': 'rebounds',
                    'player_assists': 'assists'
                }
                
                stat_type = stat_type_map.get(market_type)
                if not stat_type:
                    continue
                
                for outcome in market.get('outcomes', []):
                    props.append({
                        'player': outcome['description'],
                        'stat_type': stat_type,
                        'line': outcome['point'],
                        'over_odds': outcome.get('price', -110),
                        'under_odds': -110,  # Approximation
                        'bookmaker': bookmaker_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_time': game_time,
                        'date': date.strftime('%Y-%m-%d')
                    })
        
        return props
    
    def get_usage_stats(self):
        """Vérifie l'utilisation de l'API"""
        url = f'{self.base_url}/sports/{self.sport}/odds'
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            # Les headers contiennent l'info d'usage
            requests_used = response.headers.get('X-Requests-Used', 'unknown')
            requests_remaining = response.headers.get('X-Requests-Remaining', 'unknown')
            
            return {
                'used': requests_used,
                'remaining': requests_remaining,
                'total': 500  # Plan gratuit
            }
        except Exception as e:
            return {'error': str(e)}


# Test si exécuté directement
if __name__ == '__main__':
    client = OddsAPIClient()
    
    print("\n=== TEST: Récupération props 2 jours ===\n")
    
    props = client.get_player_props(days=2)
    
    print(f"\nTotal props: {len(props)}")
    
    if props:
        print("\nExemple de prop:")
        print(f"  Joueur: {props[0]['player']}")
        print(f"  Stat: {props[0]['stat_type']}")
        print(f"  Ligne: {props[0]['line']}")
        print(f"  Date: {props[0]['date']}")
        print(f"  Match: {props[0]['away_team']} @ {props[0]['home_team']}")
        print(f"  Heure: {props[0]['game_time']}")
    
    print("\n=== Utilisation API ===\n")
    usage = client.get_usage_stats()
    print(f"Utilisées: {usage.get('used')}")
    print(f"Restantes: {usage.get('remaining')}")

