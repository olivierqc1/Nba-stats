# BACKEND MODIFICATION - PART 1/2

## CHERCHE LA FONCTION scan_daily_opportunities DANS nba_analyzer_improved.py

## REMPLACE-LA PAR CE CODE (PART 1):

```python
def scan_daily_opportunities(min_edge=5.0, min_confidence='MEDIUM', days=2):
    """Scanne les opportunités sur X jours"""
    if not ODDS_API_AVAILABLE or not odds_client:
        return {
            'status': 'ERROR',
            'message': 'Odds API not available',
            'opportunities': []
        }
    
    print("\n" + "="*70)
    print(f"SCANNING OPPORTUNITIES - {days} DAY(S)")
    print("="*70)
    
    # Récupère les props pour X jours
    props = odds_client.get_player_props(days=days)
    print(f"Props retrieved: {len(props)}")
    
    # Groupe par date
    opportunities_by_date = {}
    analyzed_count = 0
    
    for prop in props:
        player = prop['player']
        stat_type = prop['stat_type']
        line = prop['line']
        bookmaker = prop['bookmaker']
        date = prop['date']
        game_time = prop.get('game_time', '')
        
        is_home = True
        opponent = prop['away_team'] if is_home else prop['home_team']
        
        try:
            result = analyzer.analyze_stat(
                player, stat_type, opponent, is_home, line, 
                remove_outliers=True
            )
            
            analyzed_count += 1
            
            if result.get('status') != 'SUCCESS':
                continue
            
            edge = result['line_analysis']['edge']
            recommendation = result['line_analysis']['recommendation']
            
            if recommendation == 'SKIP' or edge < min_edge:
                continue
            
            # Ajoute infos match
            result['game_info'] = {
                'date': date,
                'time': game_time,
                'home_team': prop['home_team'],
                'away_team': prop['away_team']
            }
            
            result['bookmaker_info'] = {
                'bookmaker': bookmaker,
                'line': line,
                'over_odds': prop.get('over_odds', -110),
                'under_odds': prop.get('under_odds', -110)
            }
            
            result['odds_comparison'] = {
                'primary': bookmaker,
                'betonline_different': False
            }
            
            # Groupe par date
            if date not in opportunities_by_date:
                opportunities_by_date[date] = []
            
            opportunities_by_date[date].append(result)
            
        except Exception as e:
            print(f"ERROR {player} {stat_type}: {e}")
            continue

# Continue avec PART 2
```

# BACKEND MODIFICATION - PART 2/2

## SUITE DU CODE (colle après PART 1):

```python
    # Trie par edge dans chaque jour
    for date in opportunities_by_date:
        opportunities_by_date[date].sort(
            key=lambda x: x['line_analysis']['edge'], 
            reverse=True
        )
    
    # Compte total
    total_opportunities = sum(len(opps) for opps in opportunities_by_date.values())
    
    print(f"OK: {analyzed_count} props analyzed")
    print(f"Opportunities found: {total_opportunities} (edge >= {min_edge}%)")
    
    # Résumé par jour
    for date in sorted(opportunities_by_date.keys()):
        count = len(opportunities_by_date[date])
        print(f"  {date}: {count} opportunities")
    
    print("="*70 + "\n")
    
    return {
        'status': 'SUCCESS',
        'total_props_available': len(props),
        'total_analyzed': analyzed_count,
        'opportunities_found': total_opportunities,
        'scan_time': datetime.now().isoformat(),
        'days_scanned': days,
        'filters': {
            'min_edge': min_edge,
            'min_confidence': min_confidence
        },
        'opportunities_by_date': opportunities_by_date,
        'opportunities': [opp for opps in opportunities_by_date.values() for opp in opps]
    }
```

---

## PUIS CHERCHE L'ENDPOINT @app.route('/api/daily-opportunities')

## REMPLACE PAR:

```python
@app.route('/api/daily-opportunities', methods=['GET'])
def daily_opportunities():
    """Endpoint PRINCIPAL - SUPPORT 2 JOURS"""
    min_edge = request.args.get('min_edge', 5.0, type=float)
    min_confidence = request.args.get('min_confidence', 'MEDIUM', type=str)
    days = request.args.get('days', 2, type=int)
    
    # Limite à 2 jours max
    if days > 2:
        days = 2
    
    result = scan_daily_opportunities(min_edge, min_confidence, days)
    return jsonify(result)
```

---

## C'EST TOUT POUR LE BACKEND!

Passe au dashboard (DASHBOARD_MOD_PART1.txt)
