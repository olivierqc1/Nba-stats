# 🏀 NBA Betting Analyzer v4.0

Système d'analyse avancé pour paris sportifs NBA avec:
- ✅ **Points, Assists, Rebounds** séparés
- ✅ **Test du Chi-carré (χ²)** pour valider le modèle
- ✅ **Détection avancée d'outliers** (IQR + Z-score + MAD)
- ✅ **Exclusion optionnelle** des données extrêmes
- ✅ **Régression linéaire** avec p-values et intervalles de confiance
- ✅ **Kelly Criterion** pour la gestion de bankroll

---

## 🚀 Installation Rapide

### Option 1: Déploiement sur Render (GRATUIT)

1. **Crée un compte sur [Render.com](https://render.com)**

2. **Crée un nouveau Web Service:**
   - Repository: Ton repo GitHub
   - Branch: `main`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python nba_analyzer_improved.py`

3. **Variables d'environnement:**
   ```
   PORT = 10000
   DEBUG = False
   ```

4. **Déploie!** Render va build et démarrer automatiquement.

---

### Option 2: Local (pour développement)

```bash
# 1. Clone le repo
git clone https://github.com/ton-username/Nba-betting.git
cd Nba-betting

# 2. Installe les dépendances
pip install flask flask-cors numpy pandas scipy scikit-learn nba-api --break-system-packages

# 3. Lance le serveur
python3 nba_analyzer_improved.py

# 4. Ouvre le frontend
# Ouvre index_v4.html dans ton navigateur
```

---

## 📊 Fonctionnalités

### 1. **Analyse Multi-Stats**

Analyse séparée pour:
- 🎯 **Points** (prédiction de la performance scoreur)
- 🎯 **Assists** (prédiction de la distribution)
- 🎯 **Rebounds** (prédiction du rebond)

### 2. **Test du Chi-carré (χ²)**

Valide la qualité du modèle:
- **H0:** Distribution observée = Distribution attendue
- **H1:** Différence significative
- **p-value < 0.05:** Modèle inadapté ⚠️
- **p-value ≥ 0.05:** Modèle conforme ✅

### 3. **Détection d'Outliers**

3 méthodes combinées:
- **IQR (Interquartile Range):** Q1 - 1.5×IQR à Q3 + 1.5×IQR
- **Z-score:** |z| > 2.5
- **MAD (Median Absolute Deviation):** Modified Z-score > 3.5

Un point est considéré **outlier** s'il est détecté par **≥2 méthodes**.

### 4. **Exclusion d'Outliers**

Option pour exclure les outliers du modèle de régression:
- ✅ **ACTIVÉ:** Modèle plus stable (R² souvent meilleur)
- ❌ **DÉSACTIVÉ:** Toutes les données incluses

### 5. **Régression Linéaire**

Variables prédictives:
- `is_home`: Domicile/Extérieur
- `opponent_def_rating`: Qualité défensive adverse
- `minutes`: Temps de jeu estimé
- `rest_days`: Jours de repos
- `back_to_back`: Match consécutif
- `team_pace`: Rythme de jeu

Statistiques:
- **R²** et **R² ajusté**
- **RMSE** (Root Mean Square Error)
- **p-values** pour chaque coefficient

---

## 🔧 API Endpoints

### `POST /api/analyze`

Analyse **UNE** statistique.

**Request:**
```json
{
  "player": "LeBron James",
  "opponent": "GSW",
  "is_home": true,
  "stat_type": "points",
  "line": 25.5,
  "remove_outliers": true
}
```

**Response:**
```json
{
  "status": "SUCCESS",
  "prediction": 27.3,
  "confidence_interval": {"lower": 22.1, "upper": 32.5},
  "line_analysis": {
    "recommendation": "OVER",
    "over_probability": 67.2,
    "edge": 14.7,
    "kelly_criterion": 5.3
  },
  "chi_square_test": {
    "chi2_statistic": 8.42,
    "p_value": 0.392,
    "significant": false
  },
  "outlier_analysis": {
    "outliers_detected": 2,
    "outliers_pct": 8.0,
    "data_used": "CLEANED"
  }
}
```

---

### `POST /api/analyze-all`

Analyse **LES 3** statistiques en un seul appel.

**Request:**
```json
{
  "player": "Luka Doncic",
  "opponent": "BOS",
  "is_home": false,
  "lines": {
    "points": 28.5,
    "assists": 8.5,
    "rebounds": 7.5
  },
  "remove_outliers": true
}
```

**Response:**
```json
{
  "status": "SUCCESS",
  "analyses": {
    "points": { /* analyse complète */ },
    "assists": { /* analyse complète */ },
    "rebounds": { /* analyse complète */ }
  }
}
```

---

## 📈 Interprétation des Résultats

### **Recommandation:**

- **OVER:** Prédiction > Ligne + Edge significatif
- **UNDER:** Prédiction < Ligne + Edge significatif  
- **SKIP:** Edge insuffisant (< 5%)

### **Edge:**

Avantage par rapport à la cote implicite:
- **< 5%:** SKIP
- **5-10%:** Confiance MOYENNE
- **> 10%:** Confiance HAUTE

### **Kelly Criterion:**

Pourcentage de bankroll à miser:
- **0-2%:** Mise conservatrice
- **2-5%:** Mise modérée
- **5%+:** Forte opportunité

### **Test χ²:**

- **p < 0.05:** ⚠️ Modèle inadapté (données ne suivent pas la distribution attendue)
- **p ≥ 0.05:** ✅ Modèle conforme

---

## 🎯 Exemple d'Utilisation

### Scénario: LeBron James vs Warriors

1. **Entre les paramètres:**
   - Joueur: LeBron James
   - Adversaire: GSW (Warriors)
   - Localisation: Domicile
   - ☑️ Exclure outliers: OUI

2. **Clique "Analyser"**

3. **Résultats affichés:**

   **📊 POINTS:**
   - Prédiction: **27.3 pts**
   - Ligne bookmaker: **25.5 pts**
   - Recommandation: **OVER 25.5** ✅
   - Edge: **+12.4%**
   - Kelly: **4.2%** de la bankroll

   **🎯 ASSISTS:**
   - Prédiction: **6.8 asts**
   - Ligne: **7.5 asts**
   - Recommandation: **UNDER 7.5** ✅
   - Edge: **+8.1%**

   **🏀 REBOUNDS:**
   - Prédiction: **7.2 rebs**
   - Ligne: **7.5 rebs**
   - Recommandation: **SKIP** ⏸️
   - Edge: **+2.3%** (trop faible)

   **🔍 Outliers:**
   - 2 matchs exclus (8%)
   - R² amélioré de 0.68 → 0.81

   **📊 Chi-carré:**
   - χ² = 11.24, p = 0.187
   - ✅ Modèle conforme

---

## ⚙️ Configuration Avancée

### Ajuster les Seuils

Dans `nba_analyzer_improved.py`:

```python
# Seuils d'outliers
z_threshold = 2.5        # Z-score
iqr_multiplier = 1.5     # IQR
mad_threshold = 3.5      # MAD

# Seuils de recommandation
edge_threshold = 0.05    # 5% minimum
```

### Ajouter des Variables

Pour enrichir le modèle:

```python
# Dans get_player_games()
df_clean['fatigue_index'] = ...
df_clean['opponent_pace'] = ...

# Dans analyze_stat()
X = df_model[['is_home', ..., 'fatigue_index', 'opponent_pace']]
```

---

## 🐛 Troubleshooting

### Backend ne démarre pas
```bash
# Vérifie les dépendances
pip list | grep -E "flask|numpy|pandas|scipy|sklearn|nba-api"

# Réinstalle si nécessaire
pip install --upgrade flask flask-cors numpy pandas scipy scikit-learn nba-api
```

### Erreur nba_api
```bash
# Si nba_api échoue, le système utilise des données simulées
# Pour forcer les vraies données:
pip install nba-api --upgrade
```

### Frontend ne se connecte pas
```javascript
// Change l'URL dans index_v4.html
const API_URL = 'http://localhost:5000';  // Local
// OU
const API_URL = 'https://ton-app.onrender.com';  // Render
```

---

## 📝 TODO / Améliorations Futures

- [ ] Ajouter **turnovers** et **steals** dans l'analyse
- [ ] Implémenter **boosting (XGBoost)** pour améliorer les prédictions
- [ ] Ajouter **feature importance** pour identifier les variables clés
- [ ] Créer **dashboard historique** des performances du modèle
- [ ] Ajouter **backtesting** sur saisons passées
- [ ] Intégrer **live odds** de plusieurs bookmakers
- [ ] Créer **système de tracking** des paris réels

---

## 📄 Licence

MIT License - Utilise librement pour usage personnel.

---

## 👨‍💻 Auteur

Créé par **Olivier** 🇨🇦🇪🇸

Pour questions: [GitHub Issues](https://github.com/ton-username/Nba-betting/issues)
