#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Model Trainer - NBA Predictions - PART 1/2
Train, validate et sauvegarde les modèles pour Points, Assists, Rebounds

⚠️ CE FICHIER EST INCOMPLET!
📋 INSTRUCTIONS:
1. Télécharge PART1 et PART2
2. Copie TOUT le contenu de PART1
3. Copie TOUT le contenu de PART2 EN DESSOUS
4. Sauvegarde comme "xgboost_model.py"
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from advanced_data_collector import AdvancedDataCollector


class XGBoostNBAModel:
    """
    Modèle XGBoost pour prédictions NBA
    """
    
    def __init__(self, stat_type='points'):
        """
        Args:
            stat_type: 'points', 'assists' ou 'rebounds'
        """
        self.stat_type = stat_type
        self.stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}[stat_type]
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_stats = {}
        
        # Hyperparamètres optimisés
        self.model_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(self, player_name, season='2024-25', save_model=True):
        """
        Entraîne le modèle sur les données du joueur
        
        Args:
            player_name: Nom du joueur
            season: Saison à utiliser
            save_model: Si True, sauvegarde le modèle
        
        Returns:
            dict avec métriques d'entraînement
        """
        
        print(f"\n{'='*70}")
        print(f"ENTRAÎNEMENT XGBOOST - {player_name} ({self.stat_type.upper()})")
        print(f"{'='*70}\n")
        
        # 1. Collecte données
        collector = AdvancedDataCollector()
        df = collector.get_complete_player_data(player_name, season)
        
        if df is None or len(df) < 10:
            return {'status': 'ERROR', 'message': 'Pas assez de données'}
        
        # 2. Prépare features et target
        X, y, feature_names = self._prepare_training_data(df)
        
        if X is None:
            return {'status': 'ERROR', 'message': 'Erreur préparation données'}
        
        self.feature_names = feature_names
        
        print(f"📊 Données préparées:")
        print(f"  Échantillons: {len(X)}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Target: {self.stat_col} (moyenne: {y.mean():.1f})\n")
        
        # 3. Standardisation
        X_scaled = self.scaler.fit_transform(X)
        
        # 4. Cross-validation temporelle
        print("🔄 Cross-Validation (Time Series Split)...")
        cv_results = self._cross_validate(X_scaled, y)
        
        # 5. Entraînement final
        print("\n🤖 Entraînement modèle final...")
        self.model = XGBRegressor(**self.model_params)
        self.model.fit(X_scaled, y, verbose=False)
        
        # 6. Évaluation
        train_metrics = self._evaluate_model(X_scaled, y, X, y)
        
        # 7. Feature importance
        feature_importance = self._get_feature_importance()
        
        # 8. Sauvegarde
        if save_model:
            self._save_model(player_name, season)
        
        # Résultats
        results = {
            'status': 'SUCCESS',
            'player': player_name,
            'stat_type': self.stat_type,
            'training_samples': len(X),
            'features_count': len(feature_names),
            'cv_results': cv_results,
            'train_metrics': train_metrics,
            'feature_importance': feature_importance,
            'model_saved': save_model
        }
        
        self.training_stats = results
        
        self._print_summary(results)
        
        return results
    
    def _prepare_training_data(self, df):
        """Prépare X et y pour entraînement"""
        
        exclude_cols = [
            'GAME_DATE', 'MATCHUP', 'WL', 'opponent',
            'PTS', 'AST', 'REB', 'FGM', 'FGA', 
            'FG3M', 'FG3A', 'FTM', 'FTA', 'STL', 'BLK', 'TOV'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        df_clean = df[feature_cols + [self.stat_col]].dropna()
        
        if len(df_clean) < 10:
            return None, None, None
        
        X = df_clean[feature_cols].values
        y = df_clean[self.stat_col].values
        
        return X, y, feature_cols
    
    def _cross_validate(self, X, y, n_splits=5):
        """Cross-validation temporelle"""
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        r2_scores = []
        rmse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model_fold = XGBRegressor(**self.model_params)
            model_fold.fit(X_train, y_train, verbose=False)
            
            y_pred = model_fold.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            
            print(f"  Fold {fold+1}/{n_splits}: R² = {r2:.3f}, RMSE = {rmse:.2f}")
        
        cv_results = {
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'rmse_mean': float(np.mean(rmse_scores)),
            'rmse_std': float(np.std(rmse_scores)),
            'folds': n_splits
        }
        
        print(f"\n  📊 Moyenne CV: R² = {cv_results['r2_mean']:.3f} (±{cv_results['r2_std']:.3f})")
        print(f"                 RMSE = {cv_results['rmse_mean']:.2f} (±{cv_results['rmse_std']:.2f})")
        
        return cv_results
    
    def _evaluate_model(self, X_scaled, y, X_original, y_original):
        """Évalue le modèle final"""
        
        y_pred = self.model.predict(X_scaled)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        within_3 = np.sum(np.abs(y - y_pred) <= 3) / len(y) * 100
        
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'accuracy_within_3': float(within_3)
        }
    
    def _get_feature_importance(self, top_n=15):
        """Récupère feature importance"""
        
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        
        indices = np.argsort(importance)[::-1][:top_n]
        
        feature_importance = {
            self.feature_names[i]: float(importance[i])
            for i in indices
        }
        
        return feature_importance

# ============================================================================
# FIN DE PART 1/2
# ⚠️ CONTINUE AVEC xgboost_model_PART2.py
# ============================================================================

# ============================================================================
# XGBOOST_MODEL.PY - PART 2/2
# ============================================================================
#
# ⚠️ CE FICHIER COMMENCE AU MILIEU DU CODE!
#
# 📋 INSTRUCTIONS:
# 1. Ouvre xgboost_model_PART1.py
# 2. Copie TOUT son contenu
# 3. Copie TOUT le contenu ci-dessous EN DESSOUS
# 4. Sauvegarde le tout comme "xgboost_model.py"
#
# ============================================================================

    def predict(self, features_dict):
        """
        Fait une prédiction
        
        Args:
            features_dict: Dict avec toutes les features
        
        Returns:
            dict avec prédiction et intervalle de confiance
        """
        
        if self.model is None:
            return {'error': 'Model not trained'}
        
        feature_vector = [features_dict.get(fname, 0) for fname in self.feature_names]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(feature_vector_scaled)[0]
        
        confidence_interval = self._estimate_confidence_interval(prediction)
        
        return {
            'prediction': round(float(prediction), 1),
            'confidence_interval': confidence_interval,
            'model_type': 'XGBoost',
            'features_used': len(self.feature_names)
        }
    
    def _estimate_confidence_interval(self, prediction, confidence=0.95):
        """Estime intervalle de confiance"""
        
        rmse = self.training_stats.get('train_metrics', {}).get('rmse', 3.0)
        
        margin = 1.96 * rmse
        
        return {
            'lower': round(float(prediction - margin), 1),
            'upper': round(float(prediction + margin), 1),
            'width': round(float(2 * margin), 1)
        }
    
    def _save_model(self, player_name, season):
        """Sauvegarde le modèle"""
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        player_slug = player_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_path = models_dir / f"{player_slug}_{self.stat_type}_{season}.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'stat_type': self.stat_type,
            'player': player_name,
            'season': season,
            'training_stats': self.training_stats,
            'trained_at': timestamp
        }
        
        joblib.dump(model_data, model_path)
        
        print(f"\n💾 Modèle sauvegardé: {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """Charge un modèle sauvegardé"""
        
        model_data = joblib.load(model_path)
        
        instance = cls(stat_type=model_data['stat_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.training_stats = model_data['training_stats']
        
        return instance
    
    def _print_summary(self, results):
        """Affiche résumé d'entraînement"""
        
        print(f"\n{'='*70}")
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print(f"{'='*70}\n")
        
        print("📊 RÉSULTATS FINAUX:")
        print(f"  R² Train: {results['train_metrics']['r2']:.3f}")
        print(f"  RMSE: {results['train_metrics']['rmse']:.2f}")
        print(f"  MAE: {results['train_metrics']['mae']:.2f}")
        print(f"  Précision ±3: {results['train_metrics']['accuracy_within_3']:.1f}%")
        
        print(f"\n🎯 TOP 5 FEATURES PRÉDICTIVES:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5], 1):
            print(f"  {i}. {feature}: {importance:.3f}")
        
        print(f"\n{'='*70}\n")


# ============================================================================
# MODEL MANAGER - Gestion de plusieurs modèles
# ============================================================================

class ModelManager:
    """
    Gère l'entraînement et le chargement de modèles pour plusieurs joueurs
    """
    
    def __init__(self):
        self.models = {}
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
    
    def train_player_all_stats(self, player_name, season='2024-25'):
        """
        Entraîne les 3 modèles (points, assists, rebounds) pour un joueur
        """
        
        print(f"\n{'='*70}")
        print(f"ENTRAÎNEMENT COMPLET: {player_name}")
        print(f"{'='*70}\n")
        
        results = {}
        
        for stat_type in ['points', 'assists', 'rebounds']:
            print(f"\n--- {stat_type.upper()} ---")
            
            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player_name, season, save_model=True)
            
            if result['status'] == 'SUCCESS':
                key = f"{player_name}_{stat_type}"
                self.models[key] = model
                results[stat_type] = result
            else:
                print(f"❌ Échec entraînement {stat_type}")
        
        return results
    
    def load_player_models(self, player_name, season='2024-25'):
        """Charge les modèles d'un joueur"""
        
        player_slug = player_name.lower().replace(' ', '_')
        
        for stat_type in ['points', 'assists', 'rebounds']:
            model_path = self.models_dir / f"{player_slug}_{stat_type}_{season}.pkl"
            
            if model_path.exists():
                try:
                    model = XGBoostNBAModel.load_model(model_path)
                    key = f"{player_name}_{stat_type}"
                    self.models[key] = model
                    print(f"✅ Modèle chargé: {stat_type}")
                except Exception as e:
                    print(f"❌ Erreur chargement {stat_type}: {e}")
    
    def predict(self, player_name, stat_type, opponent, is_home, season='2024-25'):
        """Fait une prédiction avec le modèle approprié"""
        
        key = f"{player_name}_{stat_type}"
        
        if key not in self.models:
            self.load_player_models(player_name, season)
        
        if key not in self.models:
            print(f"⚠️  Modèle non trouvé, entraînement...")
            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player_name, season)
            if result['status'] == 'SUCCESS':
                self.models[key] = model
            else:
                return {'error': 'Unable to train model'}
        
        collector = AdvancedDataCollector()
        features = collector.prepare_features_for_prediction(
            player_name, opponent, is_home, season
        )
        
        if features is None:
            return {'error': 'Unable to prepare features'}
        
        return self.models[key].predict(features)


# ============================================================================
# TEST & EXEMPLES
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("TEST: ENTRAÎNEMENT XGBOOST")
    print("="*70)
    
    player = "LeBron James"
    
    model = XGBoostNBAModel(stat_type='points')
    results = model.train(player, '2024-25', save_model=True)
    
    if results['status'] == 'SUCCESS':
        print("\n" + "="*70)
        print("TEST: PRÉDICTION")
        print("="*70 + "\n")
        
        collector = AdvancedDataCollector()
        features = collector.prepare_features_for_prediction(
            player, 'GSW', is_home=True
        )
        
        if features:
            prediction = model.predict(features)
            
            print(f"🎯 {player} vs GSW (Domicile)")
            print(f"  Prédiction: {prediction['prediction']} points")
            print(f"  IC 95%: [{prediction['confidence_interval']['lower']} - {prediction['confidence_interval']['upper']}]")
    
    print("\n" + "="*70 + "\n")

# ============================================================================
# FIN DE PART 2/2
# ✅ FICHIER COMPLET!
# ============================================================================