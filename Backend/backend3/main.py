from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, LargeBinary, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
import pandas as pd
import numpy as np
import io
import os
import joblib
import time
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from pydantic import BaseModel

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration with retry logic
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql_db3")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "maintenance_data")
DB_RETRY_ATTEMPTS = int(os.getenv("DB_RETRY_ATTEMPTS", "10"))
DB_RETRY_DELAY = int(os.getenv("DB_RETRY_DELAY", "5"))

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

def create_engine_with_retry():
    """Create database engine with retry logic"""
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            logger.info(f"Attempting to connect to database (attempt {attempt + 1}/{DB_RETRY_ATTEMPTS})")
            engine = create_engine(
                DATABASE_URL, 
                echo=True,
                pool_pre_ping=True,  # Enables automatic reconnection
                pool_recycle=3600,   # Recycle connections every hour
                connect_args={
                    "connect_timeout": 60,
                    "read_timeout": 60,
                    "write_timeout": 60,
                }
            )
            
            # Test the connection
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            logger.info("Successfully connected to database")
            return engine
            
        except OperationalError as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < DB_RETRY_ATTEMPTS - 1:
                logger.info(f"Retrying in {DB_RETRY_DELAY} seconds...")
                time.sleep(DB_RETRY_DELAY)
            else:
                logger.error("All database connection attempts failed")
                raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {e}")
            raise

def create_tables_with_retry():
    """Create database tables with retry logic"""
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            logger.info(f"Attempting to create tables (attempt {attempt + 1}/{DB_RETRY_ATTEMPTS})")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
            return True
            
        except OperationalError as e:
            logger.warning(f"Table creation attempt {attempt + 1} failed: {e}")
            if attempt < DB_RETRY_ATTEMPTS - 1:
                logger.info(f"Retrying in {DB_RETRY_DELAY} seconds...")
                time.sleep(DB_RETRY_DELAY)
            else:
                logger.error("All table creation attempts failed")
                return False
        except Exception as e:
            logger.error(f"Unexpected error creating tables: {e}")
            return False

# SQLAlchemy setup
Base = declarative_base()

# Initialize database connection with retry
try:
    engine = create_engine_with_retry()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    sys.exit(1)

# Pydantic models for API requests/responses
class TrainingRequest(BaseModel):
    model_type: str
    test_size: float = 0.2
    hyperparameters: Dict[str, Any] = {}
    model_name: Optional[str] = None
    description: Optional[str] = None

class PredictionRequest(BaseModel):
    model_id: int
    data: List[Dict[str, float]]

class ModelResponse(BaseModel):
    id: int
    name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    created_at: str
    description: Optional[str] = None

# Database Models - Updated for new dataset format
class SensorData(Base):
    __tablename__ = "sensor_data"
    
    id = Column(Integer, primary_key=True, index=True)
    # Basic sensor measurements
    temperature = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    flow_rate = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    voltage = Column(Float, nullable=False)
    
    # FFT features for Temperature (10 components)
    fft_temp_0 = Column(Float, nullable=False)
    fft_temp_1 = Column(Float, nullable=False)
    fft_temp_2 = Column(Float, nullable=False)
    fft_temp_3 = Column(Float, nullable=False)
    fft_temp_4 = Column(Float, nullable=False)
    fft_temp_5 = Column(Float, nullable=False)
    fft_temp_6 = Column(Float, nullable=False)
    fft_temp_7 = Column(Float, nullable=False)
    fft_temp_8 = Column(Float, nullable=False)
    fft_temp_9 = Column(Float, nullable=False)
    
    # FFT features for Vibration (10 components)
    fft_vib_0 = Column(Float, nullable=False)
    fft_vib_1 = Column(Float, nullable=False)
    fft_vib_2 = Column(Float, nullable=False)
    fft_vib_3 = Column(Float, nullable=False)
    fft_vib_4 = Column(Float, nullable=False)
    fft_vib_5 = Column(Float, nullable=False)
    fft_vib_6 = Column(Float, nullable=False)
    fft_vib_7 = Column(Float, nullable=False)
    fft_vib_8 = Column(Float, nullable=False)
    fft_vib_9 = Column(Float, nullable=False)
    
    # FFT features for Pressure (10 components)
    fft_pres_0 = Column(Float, nullable=False)
    fft_pres_1 = Column(Float, nullable=False)
    fft_pres_2 = Column(Float, nullable=False)
    fft_pres_3 = Column(Float, nullable=False)
    fft_pres_4 = Column(Float, nullable=False)
    fft_pres_5 = Column(Float, nullable=False)
    fft_pres_6 = Column(Float, nullable=False)
    fft_pres_7 = Column(Float, nullable=False)
    fft_pres_8 = Column(Float, nullable=False)
    fft_pres_9 = Column(Float, nullable=False)
    
    # Target variable
    fault_type = Column(Integer, nullable=False)  # 0 for normal, other numbers for fault types
    
    # Metadata
    upload_datetime = Column(DateTime, default=datetime.utcnow)

class UploadLog(Base):
    __tablename__ = "upload_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_datetime = Column(DateTime, default=datetime.utcnow)
    records_count = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False)
    error_message = Column(Text, nullable=True)

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)
    model_data = Column(LargeBinary(length=16777216), nullable=False)  # 16MB LONGBLOB
    scaler_data = Column(LargeBinary(length=16777216), nullable=False)  # 16MB LONGBLOB
    feature_columns = Column(Text, nullable=False)  # JSON string of feature column names
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    hyperparameters = Column(Text, nullable=False)  # JSON string
    training_data_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    num_classes = Column(Integer, nullable=False, default=2)  # Number of fault types

class PredictionResult(Base):
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    input_data = Column(Text, nullable=False)  # JSON string of input features
    predicted_fault_type = Column(Integer, nullable=False)  # Predicted fault type
    prediction_probability = Column(Text, nullable=True)  # JSON string of class probabilities
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingHistory(Base):
    __tablename__ = "training_history"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    training_started = Column(DateTime, nullable=False)
    training_completed = Column(DateTime, nullable=False)
    training_duration_seconds = Column(Float, nullable=False)
    dataset_size = Column(Integer, nullable=False)
    test_size = Column(Float, nullable=False)
    confusion_matrix = Column(Text, nullable=False)  # JSON string
    classification_report = Column(Text, nullable=False)
    feature_importance = Column(Text, nullable=True)  # JSON string
    status = Column(String(50), nullable=False)
    error_message = Column(Text, nullable=True)

# Create tables function
def create_tables():
    """Create tables using retry logic"""
    return create_tables_with_retry()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ML Trainer Class - Updated for new dataset
class MaintenancePredictionTrainer:
    def __init__(self):
        self.models_config = {
            'random_forest': {
                'name': 'Random Forest',
                'class': RandomForestClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            },
            'svm': {
                'name': 'Support Vector Machine',
                'class': SVC,
                'default_params': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'probability': True,  # Enable probability predictions
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'class': GradientBoostingClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'class': LogisticRegression,
                'default_params': {
                    'C': 1.0,
                    'solver': 'liblinear',
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            'neural_network': {
                'name': 'Neural Network (MLP)',
                'class': MLPClassifier,
                'default_params': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'constant',
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }
        
        # Define all feature columns for the new dataset
        self.feature_columns = [
            'temperature', 'vibration', 'pressure', 'flow_rate', 'current', 'voltage',
            'fft_temp_0', 'fft_temp_1', 'fft_temp_2', 'fft_temp_3', 'fft_temp_4',
            'fft_temp_5', 'fft_temp_6', 'fft_temp_7', 'fft_temp_8', 'fft_temp_9',
            'fft_vib_0', 'fft_vib_1', 'fft_vib_2', 'fft_vib_3', 'fft_vib_4',
            'fft_vib_5', 'fft_vib_6', 'fft_vib_7', 'fft_vib_8', 'fft_vib_9',
            'fft_pres_0', 'fft_pres_1', 'fft_pres_2', 'fft_pres_3', 'fft_pres_4',
            'fft_pres_5', 'fft_pres_6', 'fft_pres_7', 'fft_pres_8', 'fft_pres_9'
        ]

    def load_data_from_db(self, db: Session):
        """Load sensor data from database"""
        try:
            # Query all sensor data
            sensor_data = db.query(SensorData).all()
            
            if not sensor_data:
                raise ValueError("No sensor data found in database")
            
            # Convert to DataFrame
            data_list = []
            for record in sensor_data:
                data_dict = {
                    'temperature': record.temperature,
                    'vibration': record.vibration,
                    'pressure': record.pressure,
                    'flow_rate': record.flow_rate,
                    'current': record.current,
                    'voltage': record.voltage,
                    'fault_type': record.fault_type
                }
                
                # Add FFT features
                for i in range(10):
                    data_dict[f'fft_temp_{i}'] = getattr(record, f'fft_temp_{i}')
                    data_dict[f'fft_vib_{i}'] = getattr(record, f'fft_vib_{i}')
                    data_dict[f'fft_pres_{i}'] = getattr(record, f'fft_pres_{i}')
                
                data_list.append(data_dict)
            
            df = pd.DataFrame(data_list)
            logger.info(f"Loaded {len(df)} records from database")
            logger.info(f"Fault type distribution: {df['fault_type'].value_counts().to_dict()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise

    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Separate features and target
        X = df[self.feature_columns]
        y = df['fault_type']
        
        # Check for missing values
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        
        return X, y, self.feature_columns

    def train_model(self, db: Session, model_type: str, hyperparameters: Dict, 
                   test_size: float = 0.2, model_name: str = None, description: str = None):
        """Train a maintenance prediction model"""
        
        if model_type not in self.models_config:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        training_start = datetime.utcnow()
        
        try:
            # Load data
            df = self.load_data_from_db(db)
            X, y, feature_columns = self.preprocess_data(df)
            
            # Check if we have enough samples for each class
            class_counts = y.value_counts()
            if len(class_counts) < 2:
                raise ValueError("Need at least 2 different fault types for classification")
            
            # Split data - stratify to maintain class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Prepare model parameters
            model_config = self.models_config[model_type]
            params = model_config['default_params'].copy()
            params.update(hyperparameters)
            
            # Train model
            model = model_config['class'](**params)
            model.fit(X_train_scaled, y_train)
            
            training_end = datetime.utcnow()
            training_duration = (training_end - training_start).total_seconds()
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Generate reports
            cm = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_.tolist()))
            
            # Serialize model and scaler using BytesIO
            model_buffer = io.BytesIO()
            scaler_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            joblib.dump(scaler, scaler_buffer)
            model_data = model_buffer.getvalue()
            scaler_data = scaler_buffer.getvalue()
            
            # Generate model name if not provided
            if not model_name:
                model_name = f"{model_config['name']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Save to database
            ml_model = MLModel(
                name=model_name,
                model_type=model_type,
                model_data=model_data,
                scaler_data=scaler_data,
                feature_columns=str(feature_columns),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                hyperparameters=str(params),
                training_data_count=len(df),
                description=description,
                num_classes=len(np.unique(y))
            )
            
            db.add(ml_model)
            db.flush()  # Get the ID
            
            # Save training history
            training_history = TrainingHistory(
                model_id=ml_model.id,
                training_started=training_start,
                training_completed=training_end,
                training_duration_seconds=training_duration,
                dataset_size=len(df),
                test_size=test_size,
                confusion_matrix=str(cm.tolist()),
                classification_report=class_report,
                feature_importance=str(feature_importance) if feature_importance else None,
                status="completed"
            )
            
            db.add(training_history)
            db.commit()
            
            return {
                'model_id': ml_model.id,
                'model_name': model_name,
                'model_type': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_duration': training_duration,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'feature_importance': feature_importance,
                'num_classes': len(np.unique(y)),
                'class_distribution': class_counts.to_dict()
            }
            
        except Exception as e:
            # Rollback the session first
            db.rollback()
            
            # Log error in training history
            training_history = TrainingHistory(
                model_id=0,  # Will be updated if model was created
                training_started=training_start,
                training_completed=datetime.utcnow(),
                training_duration_seconds=0,
                dataset_size=0,
                test_size=test_size,
                confusion_matrix="[]",
                classification_report="",
                status="failed",
                error_message=str(e)
            )
            db.add(training_history)
            db.commit()
            raise

    def predict(self, db: Session, model_id: int, input_data: List[Dict]):
        """Make predictions using a trained model"""
        try:
            # Load model from database
            ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
            if not ml_model:
                raise ValueError(f"Model with ID {model_id} not found")
            
            # Deserialize model and scaler using BytesIO
            model = joblib.load(io.BytesIO(ml_model.model_data))
            scaler = joblib.load(io.BytesIO(ml_model.scaler_data))
            
            # Get feature columns
            feature_columns = eval(ml_model.feature_columns)
            
            # Prepare input data
            df_input = pd.DataFrame(input_data)
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in df_input.columns:
                    raise ValueError(f"Missing feature: {col}")
            
            # Select and order features correctly
            X_input = df_input[feature_columns]
            
            # Scale input data
            X_input_scaled = scaler.transform(X_input)
            
            # Make predictions
            predictions = model.predict(X_input_scaled)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_input_scaled)
            
            # Save prediction results
            results = []
            for i, pred in enumerate(predictions):
                prob_dict = None
                if probabilities is not None:
                    # Get unique classes and their probabilities
                    classes = model.classes_
                    prob_dict = {int(cls): float(prob) for cls, prob in zip(classes, probabilities[i])}
                
                prediction_result = PredictionResult(
                    model_id=model_id,
                    input_data=str(input_data[i]),
                    predicted_fault_type=int(pred),
                    prediction_probability=str(prob_dict) if prob_dict else None
                )
                db.add(prediction_result)
                
                fault_name = "Normal" if pred == 0 else f"Fault Type {pred}"
                
                results.append({
                    'input': input_data[i],
                    'predicted_fault_type': int(pred),
                    'fault_name': fault_name,
                    'is_fault': bool(pred != 0),  # True if any fault detected
                    'probabilities': prob_dict
                })
            
            db.commit()
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

# FastAPI app
app = FastAPI(
    title="Industrial Maintenance Prediction API",
    description="API for uploading sensor data with FFT features, training ML models, and predicting fault types",
    version="3.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8001", "http://localhost:8002", "http://localhost:8003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trainer = MaintenancePredictionTrainer()

@app.on_event("startup")
async def startup_event():
    """Startup event with better error handling"""
    logger.info("Starting application...")
    
    # Try to create tables
    success = create_tables()
    if not success:
        logger.error("Failed to create database tables. Application may not work properly.")
    
    logger.info("Application startup completed")

@app.get("/")
async def root():
    return {"message": "Industrial Maintenance Prediction API v3.0 is running"}

@app.get("/health")
async def health_check():
    """Enhanced health check with database status"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        # Check table existence
        table_count = 0
        try:
            db = SessionLocal()
            tables_result = db.execute(text("SHOW TABLES")).fetchall()
            table_count = len(tables_result)
            db.close()
        except:
            pass
        
        return {
            "status": "healthy",
            "database": "connected",
            "tables_created": table_count > 0,
            "table_count": table_count,
            "database_url_host": MYSQL_HOST,
            "database_name": MYSQL_DATABASE,
            "supported_features": len(trainer.feature_columns)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "database_url_host": MYSQL_HOST,
            "database_name": MYSQL_DATABASE
        }

@app.get("/health/db")
async def database_health_check():
    """Check database connectivity"""
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1 as health_check")).fetchone()
        db.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "test_query_result": result[0] if result else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }

# Updated CSV upload endpoint for new data format
# Replace the upload_csv function starting at line ~705 with this flexible version:

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    upload_log = UploadLog(
        filename=file.filename,
        records_count=0,
        status="processing"
    )
    db.add(upload_log)
    db.commit()
    db.refresh(upload_log)
    
    try:
        content = await file.read()
        csv_data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        logger.info(f"📊 CSV uploaded with columns: {list(csv_data.columns)}")
        
        # Expected columns for the dataset format
        required_columns = [
            'Temperature', 'Vibration', 'Pressure', 'Flow_Rate', 'Current', 'Voltage',
            'FFT_Temp_0', 'FFT_Temp_1', 'FFT_Temp_2', 'FFT_Temp_3', 'FFT_Temp_4',
            'FFT_Temp_5', 'FFT_Temp_6', 'FFT_Temp_7', 'FFT_Temp_8', 'FFT_Temp_9',
            'FFT_Vib_0', 'FFT_Vib_1', 'FFT_Vib_2', 'FFT_Vib_3', 'FFT_Vib_4',
            'FFT_Vib_5', 'FFT_Vib_6', 'FFT_Vib_7', 'FFT_Vib_8', 'FFT_Vib_9',
            'FFT_Pres_0', 'FFT_Pres_1', 'FFT_Pres_2', 'FFT_Pres_3', 'FFT_Pres_4',
            'FFT_Pres_5', 'FFT_Pres_6', 'FFT_Pres_7', 'FFT_Pres_8', 'FFT_Pres_9',
            'Fault_Type'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in csv_data.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Missing columns: {missing_columns}")
            logger.info("🔧 Attempting to create missing columns from available data...")
            
            # Try to map existing columns to required ones
            col_map = {}
            for col in csv_data.columns:
                col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                if 'temp' in col_lower and 'fft' not in col_lower:
                    col_map['Temperature'] = col
                elif 'vib' in col_lower and 'fft' not in col_lower:
                    col_map['Vibration'] = col
                elif 'press' in col_lower and 'fft' not in col_lower:
                    col_map['Pressure'] = col
                elif 'flow' in col_lower:
                    col_map['Flow_Rate'] = col
                elif 'current' in col_lower or 'curr' in col_lower:
                    col_map['Current'] = col
                elif 'volt' in col_lower:
                    col_map['Voltage'] = col
                elif 'fault' in col_lower or 'label' in col_lower or 'class' in col_lower:
                    col_map['Fault_Type'] = col
            
            # Rename mapped columns
            csv_data = csv_data.rename(columns=col_map)
            logger.info(f"✅ Mapped columns: {col_map}")
            
            # Create missing basic columns with defaults
            if 'Temperature' not in csv_data.columns:
                csv_data['Temperature'] = 25.0
                logger.info("Created Temperature with default value 25.0")
            if 'Vibration' not in csv_data.columns:
                csv_data['Vibration'] = 0.1
                logger.info("Created Vibration with default value 0.1")
            if 'Pressure' not in csv_data.columns:
                csv_data['Pressure'] = 100.0
                logger.info("Created Pressure with default value 100.0")
            if 'Flow_Rate' not in csv_data.columns:
                csv_data['Flow_Rate'] = 50.0
                logger.info("Created Flow_Rate with default value 50.0")
            if 'Current' not in csv_data.columns:
                csv_data['Current'] = 10.0
                logger.info("Created Current with default value 10.0")
            if 'Voltage' not in csv_data.columns:
                csv_data['Voltage'] = 220.0
                logger.info("Created Voltage with default value 220.0")
            
            # Create missing FFT columns with zeros
            for fft_type in ['Temp', 'Vib', 'Pres']:
                for i in range(10):
                    col_name = f'FFT_{fft_type}_{i}'
                    if col_name not in csv_data.columns:
                        csv_data[col_name] = 0.0
            
            logger.info("✅ Created all missing FFT columns with zeros")
            
            # Handle Fault_Type
            if 'Fault_Type' not in csv_data.columns:
                csv_data['Fault_Type'] = 0
                logger.info("Created Fault_Type with default value 0 (Normal)")
        
        # Validate numeric columns
        numeric_columns = [col for col in required_columns if col != 'Fault_Type']
        for col in numeric_columns:
            if col in csv_data.columns:
                try:
                    csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce')
                    # Fill NaN with 0
                    csv_data[col] = csv_data[col].fillna(0.0)
                except Exception as e:
                    logger.warning(f"Error converting {col}: {e}")
        
        # Validate fault_type column
        try:
            csv_data['Fault_Type'] = csv_data['Fault_Type'].astype(int)
        except Exception as e:
            logger.warning(f"Error converting Fault_Type to integer: {e}. Setting all to 0.")
            csv_data['Fault_Type'] = 0
        
        # Insert records
        records_inserted = 0
        for _, row in csv_data.iterrows():
            try:
                sensor_record = SensorData(
                    temperature=float(row['Temperature']),
                    vibration=float(row['Vibration']),
                    pressure=float(row['Pressure']),
                    flow_rate=float(row['Flow_Rate']),
                    current=float(row['Current']),
                    voltage=float(row['Voltage']),
                    
                    # FFT Temperature features
                    fft_temp_0=float(row['FFT_Temp_0']),
                    fft_temp_1=float(row['FFT_Temp_1']),
                    fft_temp_2=float(row['FFT_Temp_2']),
                    fft_temp_3=float(row['FFT_Temp_3']),
                    fft_temp_4=float(row['FFT_Temp_4']),
                    fft_temp_5=float(row['FFT_Temp_5']),
                    fft_temp_6=float(row['FFT_Temp_6']),
                    fft_temp_7=float(row['FFT_Temp_7']),
                    fft_temp_8=float(row['FFT_Temp_8']),
                    fft_temp_9=float(row['FFT_Temp_9']),
                    
                    # FFT Vibration features
                    fft_vib_0=float(row['FFT_Vib_0']),
                    fft_vib_1=float(row['FFT_Vib_1']),
                    fft_vib_2=float(row['FFT_Vib_2']),
                    fft_vib_3=float(row['FFT_Vib_3']),
                    fft_vib_4=float(row['FFT_Vib_4']),
                    fft_vib_5=float(row['FFT_Vib_5']),
                    fft_vib_6=float(row['FFT_Vib_6']),
                    fft_vib_7=float(row['FFT_Vib_7']),
                    fft_vib_8=float(row['FFT_Vib_8']),
                    fft_vib_9=float(row['FFT_Vib_9']),
                    
                    # FFT Pressure features
                    fft_pres_0=float(row['FFT_Pres_0']),
                    fft_pres_1=float(row['FFT_Pres_1']),
                    fft_pres_2=float(row['FFT_Pres_2']),
                    fft_pres_3=float(row['FFT_Pres_3']),
                    fft_pres_4=float(row['FFT_Pres_4']),
                    fft_pres_5=float(row['FFT_Pres_5']),
                    fft_pres_6=float(row['FFT_Pres_6']),
                    fft_pres_7=float(row['FFT_Pres_7']),
                    fft_pres_8=float(row['FFT_Pres_8']),
                    fft_pres_9=float(row['FFT_Pres_9']),
                    
                    # Target
                    fault_type=int(row['Fault_Type'])
                )
                db.add(sensor_record)
                records_inserted += 1
            except Exception as row_err:
                logger.warning(f"Skipping row: {row_err}")
                continue
        
        upload_log.records_count = records_inserted
        upload_log.status = "completed"
        db.commit()
        
        # Get fault type distribution
        fault_distribution = csv_data['Fault_Type'].value_counts().to_dict()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "CSV uploaded successfully",
                "filename": file.filename,
                "records_inserted": records_inserted,
                "fault_type_distribution": fault_distribution,
                "upload_time": datetime.utcnow().isoformat(),
                "features_count": len(trainer.feature_columns),
                "note": "Missing columns were auto-created with default values"
            }
        )
    
    except HTTPException as http_exc:
        db.rollback()
        upload_log.status = "failed"
        upload_log.error_message = str(http_exc.detail)
        db.commit()
        raise http_exc
    except Exception as e:
        db.rollback()
        error_msg = f"Error processing CSV: {str(e)}"
        logger.error(f"CSV Processing Error: {error_msg}")
        upload_log.status = "failed"
        upload_log.error_message = error_msg
        db.commit()
        raise HTTPException(status_code=500, detail=error_msg)

# ML endpoints
@app.get("/ml/models/")
async def get_available_models():
    """Get available ML model types and their parameters"""
    return {
        "available_models": {
            key: {
                "name": config["name"],
                "default_parameters": config["default_params"]
            }
            for key, config in trainer.models_config.items()
        },
        "feature_count": len(trainer.feature_columns),
        "features": trainer.feature_columns
    }

@app.post("/ml/train/")
async def train_model(request: TrainingRequest, db: Session = Depends(get_db)):
    """Train a new ML model on the sensor data"""
    try:
        result = trainer.train_model(
            db=db,
            model_type=request.model_type,
            hyperparameters=request.hyperparameters,
            test_size=request.test_size,
            model_name=request.model_name,
            description=request.description
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Model trained successfully",
                "result": result
            }
        )
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/models/trained/", response_model=List[ModelResponse])
async def get_trained_models(db: Session = Depends(get_db)):
    """Get list of all trained models"""
    try:
        models = db.query(MLModel).filter(MLModel.is_active == True).order_by(MLModel.created_at.desc()).all()
        
        return [
            ModelResponse(
                id=model.id,
                name=model.name,
                model_type=model.model_type,
                accuracy=model.accuracy,
                precision=model.precision,
                recall=model.recall,
                f1_score=model.f1_score,
                created_at=model.created_at.isoformat(),
                description=model.description
            )
            for model in models
        ]
    
    except Exception as e:
        logger.error(f"Error retrieving models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/models/{model_id}/")
async def get_model_details(model_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific model"""
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get training history
        training_history = db.query(TrainingHistory).filter(TrainingHistory.model_id == model_id).first()
        
        return {
            "model": {
                "id": model.id,
                "name": model.name,
                "model_type": model.model_type,
                "accuracy": model.accuracy,
                "precision": model.precision,
                "recall": model.recall,
                "f1_score": model.f1_score,
                "created_at": model.created_at.isoformat(),
                "description": model.description,
                "hyperparameters": eval(model.hyperparameters),
                "feature_columns": eval(model.feature_columns),
                "training_data_count": model.training_data_count,
                "num_classes": model.num_classes
            },
            "training_history": {
                "training_duration_seconds": training_history.training_duration_seconds if training_history else None,
                "confusion_matrix": eval(training_history.confusion_matrix) if training_history else None,
                "classification_report": training_history.classification_report if training_history else None,
                "feature_importance": eval(training_history.feature_importance) if training_history and training_history.feature_importance else None
            }
        }
    
    except Exception as e:
        logger.error(f"Error retrieving model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict/")
async def make_prediction(request: PredictionRequest, db: Session = Depends(get_db)):
    """Make fault type predictions using a trained model"""
    try:
        results = trainer.predict(db, request.model_id, request.data)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Predictions made successfully",
                "model_id": request.model_id,
                "predictions": results
            }
        )
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict-single/")
async def predict_single(
    model_id: int,
    temperature: float,
    vibration: float,
    pressure: float,
    flow_rate: float,
    current: float,
    voltage: float,
    fft_temp_0: float = 0, fft_temp_1: float = 0, fft_temp_2: float = 0, fft_temp_3: float = 0, fft_temp_4: float = 0,
    fft_temp_5: float = 0, fft_temp_6: float = 0, fft_temp_7: float = 0, fft_temp_8: float = 0, fft_temp_9: float = 0,
    fft_vib_0: float = 0, fft_vib_1: float = 0, fft_vib_2: float = 0, fft_vib_3: float = 0, fft_vib_4: float = 0,
    fft_vib_5: float = 0, fft_vib_6: float = 0, fft_vib_7: float = 0, fft_vib_8: float = 0, fft_vib_9: float = 0,
    fft_pres_0: float = 0, fft_pres_1: float = 0, fft_pres_2: float = 0, fft_pres_3: float = 0, fft_pres_4: float = 0,
    fft_pres_5: float = 0, fft_pres_6: float = 0, fft_pres_7: float = 0, fft_pres_8: float = 0, fft_pres_9: float = 0,
    db: Session = Depends(get_db)
):
    """Make a single fault type prediction with all features"""
    try:
        input_data = [{
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "flow_rate": flow_rate,
            "current": current,
            "voltage": voltage,
            "fft_temp_0": fft_temp_0, "fft_temp_1": fft_temp_1, "fft_temp_2": fft_temp_2, "fft_temp_3": fft_temp_3, "fft_temp_4": fft_temp_4,
            "fft_temp_5": fft_temp_5, "fft_temp_6": fft_temp_6, "fft_temp_7": fft_temp_7, "fft_temp_8": fft_temp_8, "fft_temp_9": fft_temp_9,
            "fft_vib_0": fft_vib_0, "fft_vib_1": fft_vib_1, "fft_vib_2": fft_vib_2, "fft_vib_3": fft_vib_3, "fft_vib_4": fft_vib_4,
            "fft_vib_5": fft_vib_5, "fft_vib_6": fft_vib_6, "fft_vib_7": fft_vib_7, "fft_vib_8": fft_vib_8, "fft_vib_9": fft_vib_9,
            "fft_pres_0": fft_pres_0, "fft_pres_1": fft_pres_1, "fft_pres_2": fft_pres_2, "fft_pres_3": fft_pres_3, "fft_pres_4": fft_pres_4,
            "fft_pres_5": fft_pres_5, "fft_pres_6": fft_pres_6, "fft_pres_7": fft_pres_7, "fft_pres_8": fft_pres_8, "fft_pres_9": fft_pres_9
        }]
        
        results = trainer.predict(db, model_id, input_data)
        
        if results:
            result = results[0]
            return {
                "model_id": model_id,
                "input": result["input"],
                "predicted_fault_type": result["predicted_fault_type"],
                "fault_name": result["fault_name"],
                "is_fault": result["is_fault"],
                "confidence": result["probabilities"][result["predicted_fault_type"]] if result["probabilities"] else None,
                "all_probabilities": result["probabilities"]
            }
        else:
            raise HTTPException(status_code=500, detail="No prediction result returned")
    
    except Exception as e:
        logger.error(f"Error making single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/predictions/")
async def get_prediction_history(
    model_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get prediction history"""
    try:
        query = db.query(PredictionResult)
        
        if model_id:
            query = query.filter(PredictionResult.model_id == model_id)
        
        predictions = query.order_by(PredictionResult.created_at.desc()).offset(offset).limit(limit).all()
        
        return {
            "predictions": [
                {
                    "id": pred.id,
                    "model_id": pred.model_id,
                    "input_data": eval(pred.input_data),
                    "predicted_fault_type": pred.predicted_fault_type,
                    "fault_name": "Normal" if pred.predicted_fault_type == 0 else f"Fault Type {pred.predicted_fault_type}",
                    "is_fault": bool(pred.predicted_fault_type != 0),
                    "prediction_probability": eval(pred.prediction_probability) if pred.prediction_probability else None,
                    "created_at": pred.created_at.isoformat()
                }
                for pred in predictions
            ],
            "total_records": len(predictions),
            "offset": offset,
            "limit": limit
        }
    
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/ml/models/{model_id}/")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Soft delete a model (mark as inactive)"""
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model.is_active = False
        db.commit()
        
        return {"message": f"Model {model_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data retrieval endpoints
@app.get("/sensor-data/")
async def get_sensor_data(
    limit: int = 100, 
    offset: int = 0,
    fault_type: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Retrieve sensor data with optional filtering"""
    query = db.query(SensorData)
    
    if fault_type is not None:
        query = query.filter(SensorData.fault_type == fault_type)
    
    data = query.offset(offset).limit(limit).all()
    
    return {
        "data": [
            {
                "id": record.id,
                "temperature": record.temperature,
                "vibration": record.vibration,
                "pressure": record.pressure,
                "flow_rate": record.flow_rate,
                "current": record.current,
                "voltage": record.voltage,
                "fault_type": record.fault_type,
                "fault_name": "Normal" if record.fault_type == 0 else f"Fault Type {record.fault_type}",
                "is_fault": bool(record.fault_type != 0),
                "upload_datetime": record.upload_datetime.isoformat(),
                "fft_features": {
                    "temperature": [getattr(record, f'fft_temp_{i}') for i in range(10)],
                    "vibration": [getattr(record, f'fft_vib_{i}') for i in range(10)],
                    "pressure": [getattr(record, f'fft_pres_{i}') for i in range(10)]
                }
            }
            for record in data
        ],
        "total_records": len(data),
        "offset": offset,
        "limit": limit
    }

@app.get("/upload-logs/")
async def get_upload_logs(db: Session = Depends(get_db)):
    """Get upload history"""
    logs = db.query(UploadLog).order_by(UploadLog.upload_datetime.desc()).limit(50).all()
    
    return {
        "logs": [
            {
                "id": log.id,
                "filename": log.filename,
                "upload_datetime": log.upload_datetime.isoformat(),
                "records_count": log.records_count,
                "status": log.status,
                "error_message": log.error_message
            }
            for log in logs
        ]
    }

# Data Management & Cleanup Endpoints
@app.get("/database-stats/")
async def get_database_stats(db: Session = Depends(get_db)):
    """Get comprehensive statistics about data in the database"""
    try:
        stats = {
            "sensor_data": {
                "total_records": db.query(SensorData).count(),
                "fault_type_distribution": {}
            },
            "ml_models": {
                "total_models": db.query(MLModel).filter(MLModel.is_active == True).count(),
                "inactive_models": db.query(MLModel).filter(MLModel.is_active == False).count()
            },
            "predictions": {
                "total_predictions": db.query(PredictionResult).count()
            },
            "uploads": {
                "total_uploads": db.query(UploadLog).count(),
                "successful_uploads": db.query(UploadLog).filter(UploadLog.status == "completed").count(),
                "failed_uploads": db.query(UploadLog).filter(UploadLog.status == "failed").count()
            },
            "training_history": {
                "total_trainings": db.query(TrainingHistory).count(),
                "successful_trainings": db.query(TrainingHistory).filter(TrainingHistory.status == "completed").count(),
                "failed_trainings": db.query(TrainingHistory).filter(TrainingHistory.status == "failed").count()
            }
        }
        
        # Get fault type distribution
        fault_counts = db.execute(
            text("SELECT fault_type, COUNT(*) as count FROM sensor_data GROUP BY fault_type ORDER BY fault_type")
        ).fetchall()
        
        for fault_type, count in fault_counts:
            label = "Normal" if fault_type == 0 else f"Fault Type {fault_type}"
            stats["sensor_data"]["fault_type_distribution"][label] = count
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@app.delete("/clear-everything/")
async def clear_everything(confirm: str = Query(..., description="Type 'YES_DELETE_EVERYTHING' to confirm")):
    """Clear ALL data from database - sensor data, models, predictions, everything"""
    if confirm != "YES_DELETE_EVERYTHING":
        raise HTTPException(
            status_code=400, 
            detail="Confirmation required. Add ?confirm=YES_DELETE_EVERYTHING to the URL"
        )
    
    try:
        db = SessionLocal()
        
        # Count all records
        sensor_count = db.query(SensorData).count()
        upload_count = db.query(UploadLog).count()
        models_count = db.query(MLModel).count()
        predictions_count = db.query(PredictionResult).count()
        training_count = db.query(TrainingHistory).count()
        
        # Delete everything in correct order (foreign keys)
        db.query(PredictionResult).delete()
        db.query(TrainingHistory).delete()
        db.query(MLModel).delete()
        db.query(SensorData).delete()
        db.query(UploadLog).delete()
        db.commit()
        db.close()
        
        return {
            "message": "EVERYTHING CLEARED! Database is now completely empty.",
            "deleted_records": {
                "sensor_data": sensor_count,
                "upload_logs": upload_count,
                "ml_models": models_count,
                "predictions": predictions_count,
                "training_history": training_count,
                "total_records_deleted": sensor_count + upload_count + models_count + predictions_count + training_count
            },
            "warning": "All data has been permanently deleted!"
        }
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)