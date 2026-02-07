"""
Professional AutoML Engine
Enterprise-grade automated machine learning with comprehensive features
Author: AutoML Pro
Version: 3.0.0
"""

import pandas as pd
import numpy as np
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Generator, Any
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_absolute_error
)

# Model imports
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# =====================================================
# ENUMS AND DATA CLASSES
# =====================================================

class ProblemType(Enum):
    """Problem type enumeration"""
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"

class OptimizationMode(Enum):
    """Optimization mode for model training"""
    FAST = "fast"
    BALANCED = "balanced"
    PRECISE = "precise"

@dataclass
class ModelResult:
    """Data class for model training results"""
    name: str
    score: float
    secondary_metric: float
    training_time: float
    status: str
    model_object: Any = None
    predictions: np.ndarray = None

@dataclass
class DatasetInfo:
    """Data class for dataset information"""
    n_rows: int
    n_features: int
    n_classes: Optional[int]
    feature_types: Dict[str, int]
    missing_values: Dict[str, int]
    target_distribution: Dict[str, int]

# =====================================================
# DATA ANALYZER
# =====================================================

class DataAnalyzer:
    """Analyze and profile datasets"""
    
    @staticmethod
    def detect_problem_type(df: pd.DataFrame, target_col: str) -> ProblemType:
        """Automatically detect if problem is classification or regression"""
        target = df[target_col]
        unique_count = target.nunique()
        dtype = target.dtype
        
        # Classification if categorical or few unique values
        if dtype == 'object' or unique_count < 20:
            return ProblemType.CLASSIFICATION
        else:
            return ProblemType.REGRESSION
    
    @staticmethod
    def profile_dataset(df: pd.DataFrame, target_col: str) -> DatasetInfo:
        """Generate comprehensive dataset profile"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Feature types
        feature_types = {
            'numeric': len(X.select_dtypes(include=['int64', 'float64']).columns),
            'categorical': len(X.select_dtypes(include=['object', 'bool']).columns)
        }
        
        # Missing values
        missing_values = {col: X[col].isnull().sum() for col in X.columns if X[col].isnull().sum() > 0}
        
        # Target distribution
        target_dist = y.value_counts().to_dict() if y.dtype == 'object' else {}
        
        # Number of classes
        n_classes = y.nunique() if y.dtype == 'object' else None
        
        return DatasetInfo(
            n_rows=len(df),
            n_features=len(X.columns),
            n_classes=n_classes,
            feature_types=feature_types,
            missing_values=missing_values,
            target_distribution=target_dist
        )
    
    @staticmethod
    def detect_imbalance(y: pd.Series) -> Tuple[bool, float]:
        """Detect class imbalance in target variable"""
        if y.dtype != 'object' and y.nunique() > 20:
            return False, 1.0
        
        value_counts = y.value_counts()
        max_count = value_counts.max()
        min_count = value_counts.min()
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        is_imbalanced = imbalance_ratio > 3.0
        
        return is_imbalanced, imbalance_ratio

# =====================================================
# DATA PREPROCESSOR
# =====================================================

class DataPreprocessor:
    """Advanced data preprocessing pipeline"""
    
    def __init__(self, max_rows: int = 50000, high_cardinality_threshold: int = 50):
        self.max_rows = max_rows
        self.high_cardinality_threshold = high_cardinality_threshold
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
    
    def preprocess(self, df: pd.DataFrame, target_col: str, 
                   problem_type: ProblemType) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
        """Comprehensive data preprocessing"""
        # Remove rows with missing target
        df = df.dropna(subset=[target_col]).copy()
        
        # Sample for performance if dataset is large
        if len(df) > self.max_rows:
            print(f"⚡ Sampling {self.max_rows} rows for faster processing...")
            df = df.sample(self.max_rows, random_state=42)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove high cardinality categorical features
        high_card_cols = [
            col for col in X.columns 
            if X[col].dtype == 'object' and X[col].nunique() > self.high_cardinality_threshold
        ]
        
        if high_card_cols:
            print(f"🗑️  Dropping high cardinality columns: {high_card_cols}")
            X = X.drop(columns=high_card_cols)
        
        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Store feature names
        self.feature_names = numeric_features + categorical_features
        
        # Build preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Apply preprocessing
        try:
            X_processed = self.preprocessor.fit_transform(X)
        except Exception as e:
            print(f"⚠️  Preprocessing failed: {e}")
            print("🔄 Falling back to numeric-only preprocessing...")
            X = X.select_dtypes(include=['number'])
            X_processed = X.fillna(0).values
            self.preprocessor = None
        
        # Encode target for classification
        self.label_encoder = None
        if problem_type == ProblemType.CLASSIFICATION and y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = y.values
        
        return X_processed, y_processed, self.preprocessor, self.label_encoder

# =====================================================
# MODEL FACTORY
# =====================================================

class ModelFactory:
    """Factory for creating ML models with different configurations"""
    
    @staticmethod
    def get_classification_models(mode: OptimizationMode = OptimizationMode.BALANCED,
                                  is_imbalanced: bool = False) -> List[Tuple[str, Any]]:
        """Get classification models based on mode and data characteristics"""
        class_weight = 'balanced' if is_imbalanced else None
        
        # Configure based on mode
        if mode == OptimizationMode.FAST:
            n_estimators = 50
            max_depth = 10
        elif mode == OptimizationMode.BALANCED:
            n_estimators = 100
            max_depth = 20
        else:  # PRECISE
            n_estimators = 200
            max_depth = None
        
        models = [
            ("Logistic Regression", LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=42
            )),
            ("Decision Tree", DecisionTreeClassifier(
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=42
            )),
            ("Random Forest", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=42
            )),
            ("Naive Bayes", GaussianNB()),
        ]
        
        # Add more complex models for balanced and precise modes
        if mode in [OptimizationMode.BALANCED, OptimizationMode.PRECISE]:
            models.extend([
                ("Gradient Boosting", GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=5,
                    random_state=42
                )),
                ("K-Nearest Neighbors", KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=-1
                ))
            ])
        
        # Add SVM only for precise mode (slow on large datasets)
        if mode == OptimizationMode.PRECISE:
            models.append(
                ("Support Vector Machine", SVC(
                    max_iter=1000,
                    probability=True,
                    class_weight=class_weight,
                    random_state=42
                ))
            )
        
        return models
    
    @staticmethod
    def get_regression_models(mode: OptimizationMode = OptimizationMode.BALANCED) -> List[Tuple[str, Any]]:
        """Get regression models based on mode"""
        if mode == OptimizationMode.FAST:
            n_estimators = 50
            max_depth = 10
        elif mode == OptimizationMode.BALANCED:
            n_estimators = 100
            max_depth = 20
        else:  # PRECISE
            n_estimators = 200
            max_depth = None
        
        models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge(random_state=42)),
            ("Decision Tree", DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=42
            )),
            ("Random Forest", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=-1,
                random_state=42
            ))
        ]
        
        # Add more models for balanced and precise
        if mode in [OptimizationMode.BALANCED, OptimizationMode.PRECISE]:
            models.extend([
                ("Gradient Boosting", GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=5,
                    random_state=42
                )),
                ("K-Nearest Neighbors", KNeighborsRegressor(
                    n_neighbors=5,
                    n_jobs=-1
                ))
            ])
        
        if mode == OptimizationMode.PRECISE:
            models.append(
                ("Support Vector Machine", SVR(max_iter=1000))
            )
        
        return models

# =====================================================
# MODEL TRAINER
# =====================================================

class ModelTrainer:
    """Advanced model training with ensemble creation"""
    
    def __init__(self, problem_type: ProblemType, mode: OptimizationMode = OptimizationMode.BALANCED):
        self.problem_type = problem_type
        self.mode = mode
        self.results = []
        self.best_model = None
        self.best_score = -np.inf
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2,
                     is_imbalanced: bool = False) -> Generator:
        """Train multiple models and yield progress updates"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if self.problem_type == ProblemType.CLASSIFICATION else None
        )
        
        # Get models
        if self.problem_type == ProblemType.CLASSIFICATION:
            models = ModelFactory.get_classification_models(self.mode, is_imbalanced)
        else:
            models = ModelFactory.get_regression_models(self.mode)
        
        trained_estimators = []
        
        # Train each model
        for i, (name, model) in enumerate(models):
            start_time = time.time()
            status = "Success"
            
            try:
                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if self.problem_type == ProblemType.CLASSIFICATION:
                    score = accuracy_score(y_test, y_pred)
                    secondary = f1_score(y_test, y_pred, average='weighted')
                else:
                    score = r2_score(y_test, y_pred)
                    secondary = mean_absolute_error(y_test, y_pred)
                
                training_time = time.time() - start_time
                
                # Store result
                result = ModelResult(
                    name=name,
                    score=score,
                    secondary_metric=secondary,
                    training_time=training_time,
                    status=status,
                    model_object=model,
                    predictions=y_pred
                )
                
                self.results.append(result)
                trained_estimators.append((name, model))
                
                # Update best model
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                
            except Exception as e:
                status = "Failed"
                print(f"❌ {name} failed: {e}")
                training_time = 0
            
            # Yield progress
            yield {
                'step': i + 1,
                'total': len(models) + 1,
                'model_name': name,
                'time': training_time if status == "Success" else 0,
                'status': status
            }
        
        # Create ensemble if we have enough models
        if len(trained_estimators) >= 3:
            yield from self._create_ensemble(
                trained_estimators, X_train, y_train, X_test, y_test
            )
        
        # Final yield with complete results
        yield {
            'type': 'final',
            'results': self._format_results(),
            'best_model': self.best_model,
            'X_train': X_train,
            'y_test': y_test,
            'y_pred': self.best_model.predict(X_test) if self.best_model else None
        }
    
    def _create_ensemble(self, trained_estimators: List, X_train, y_train, X_test, y_test):
        """Create and train ensemble model"""
        start_time = time.time()
        
        try:
            # Sort by score and get top 3
            sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)
            top_3_names = [r.name for r in sorted_results[:3]]
            
            # Get corresponding models
            estimators = [(name, model) for name, model in trained_estimators if name in top_3_names]
            
            # Create ensemble
            if self.problem_type == ProblemType.CLASSIFICATION:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
            else:
                ensemble = VotingRegressor(estimators=estimators)
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            
            # Calculate metrics
            if self.problem_type == ProblemType.CLASSIFICATION:
                score = accuracy_score(y_test, y_pred)
                secondary = f1_score(y_test, y_pred, average='weighted')
            else:
                score = r2_score(y_test, y_pred)
                secondary = mean_absolute_error(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            # Store result
            result = ModelResult(
                name="🏆 Smart Ensemble",
                score=score,
                secondary_metric=secondary,
                training_time=training_time,
                status="Success",
                model_object=ensemble,
                predictions=y_pred
            )
            
            self.results.append(result)
            
            # Update best if ensemble is better
            if score > self.best_score:
                self.best_score = score
                self.best_model = ensemble
            
            yield {
                'step': len(trained_estimators) + 1,
                'total': len(trained_estimators) + 1,
                'model_name': 'Smart Ensemble',
                'time': training_time,
                'status': 'Success'
            }
            
        except Exception as e:
            print(f"❌ Ensemble creation failed: {e}")
    
    def _format_results(self) -> pd.DataFrame:
        """Format results as DataFrame"""
        if self.problem_type == ProblemType.CLASSIFICATION:
            data = [{
                'Algorithm': r.name,
                'Accuracy': r.score,
                'F1 Score': r.secondary_metric,
                'Training Time': f"{r.training_time:.2f}s"
            } for r in self.results]
            
            df = pd.DataFrame(data)
            df = df.sort_values('Accuracy', ascending=False)
            
        else:
            data = [{
                'Algorithm': r.name,
                'R² Score': r.score,
                'MAE': r.secondary_metric,
                'Training Time': f"{r.training_time:.2f}s"
            } for r in self.results]
            
            df = pd.DataFrame(data)
            df = df.sort_values('R² Score', ascending=False)
        
        return df

# =====================================================
# HYPERPARAMETER OPTIMIZER
# =====================================================

class HyperparameterOptimizer:
    """Grid search hyperparameter optimization"""
    
    @staticmethod
    def get_param_grid(model_name: str) -> Dict:
        """Get parameter grid for model"""
        grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "Support Vector Machine": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            "Decision Tree": {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "Ridge": {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            "Logistic Regression": {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        }
        
        return grids.get(model_name, {})
    
    @staticmethod
    def optimize(X: np.ndarray, y: np.ndarray, model_name: str, 
                 problem_type: ProblemType, cv: int = 3) -> Tuple[Dict, float, bytes]:
        """Optimize hyperparameters for a specific model"""
        # Get base model
        if problem_type == ProblemType.CLASSIFICATION:
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Support Vector Machine": SVC(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
        else:
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Support Vector Machine": SVR(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Ridge": Ridge(random_state=42),
                "K-Nearest Neighbors": KNeighborsRegressor()
            }
        
        model = models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        param_grid = HyperparameterOptimizer.get_param_grid(model_name)
        
        if not param_grid:
            # No grid available, just train the model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            return {}, score, pickle.dumps(model)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, n_jobs=-1, 
            verbose=0, scoring='accuracy' if problem_type == ProblemType.CLASSIFICATION else 'r2'
        )
        
        grid_search.fit(X, y)
        
        return (
            grid_search.best_params_,
            grid_search.best_score_,
            pickle.dumps(grid_search.best_estimator_)
        )

# =====================================================
# MAIN AUTOML CLASS
# =====================================================

class AutoML:
    """
    Main AutoML orchestrator class
    Coordinates all components for end-to-end automated machine learning
    """
    
    def __init__(self, mode: OptimizationMode = OptimizationMode.BALANCED):
        self.mode = mode
        self.analyzer = DataAnalyzer()
        self.preprocessor = DataPreprocessor()
        self.trainer = None
        self.optimizer = HyperparameterOptimizer()
        
        self.dataset_info = None
        self.problem_type = None
        self.is_imbalanced = False
        self.results_df = None
        self.best_model = None
        
    def analyze(self, df: pd.DataFrame, target_col: str) -> DatasetInfo:
        """Analyze dataset and detect problem type"""
        self.problem_type = self.analyzer.detect_problem_type(df, target_col)
        self.dataset_info = self.analyzer.profile_dataset(df, target_col)
        self.is_imbalanced, imbalance_ratio = self.analyzer.detect_imbalance(df[target_col])
        
        if self.is_imbalanced:
            print(f"⚠️  Detected class imbalance (ratio: {imbalance_ratio:.2f})")
        
        return self.dataset_info
    
    def fit(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Generator:
        """
        Fit AutoML pipeline
        Yields progress updates during training
        """
        # Analyze if not done already
        if self.problem_type is None:
            self.analyze(df, target_col)
        
        # Preprocess data
        X, y, preprocessor, label_encoder = self.preprocessor.preprocess(
            df, target_col, self.problem_type
        )
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.problem_type, self.mode)
        
        # Train models (generator)
        for update in self.trainer.train_models(X, y, test_size, self.is_imbalanced):
            if update.get('type') == 'final':
                self.results_df = update['results']
                self.best_model = update['best_model']
            yield update
    
    def tune(self, X: np.ndarray, y: np.ndarray, model_name: str, cv: int = 3):
        """Tune hyperparameters for specific model"""
        return self.optimizer.optimize(X, y, model_name, self.problem_type, cv)
    
    def get_results(self) -> pd.DataFrame:
        """Get training results DataFrame"""
        return self.results_df
    
    def get_best_model(self):
        """Get best trained model"""
        return self.best_model