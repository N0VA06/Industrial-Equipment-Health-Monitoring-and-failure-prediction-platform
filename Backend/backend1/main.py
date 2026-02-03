from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, LargeBinary, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import io
import os
import joblib
import base64
import logging
import traceback
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

# Database configuration
MYSQL_USER = os.getenv("MYSQL_USER", "app_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "app_password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql_db1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "sensor_data")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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

# Database Models
class SensorData(Base):
    __tablename__ = "sensor_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    vibration = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    rms_vibration = Column(Float, nullable=False)
    mean_temp = Column(Float, nullable=False)
    fault_label = Column(Integer, nullable=False)
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
    model_data = Column(LargeBinary(length=16777216), nullable=False)
    scaler_data = Column(LargeBinary(length=16777216), nullable=False)
    feature_columns = Column(Text, nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    hyperparameters = Column(Text, nullable=False)
    training_data_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

class PredictionResult(Base):
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    input_data = Column(Text, nullable=False)
    predicted_fault_label = Column(Integer, nullable=False)
    prediction_probability = Column(Text, nullable=True)
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
    confusion_matrix = Column(Text, nullable=False)
    classification_report = Column(Text, nullable=False)
    feature_importance = Column(Text, nullable=True)
    status = Column(String(50), nullable=False)
    error_message = Column(Text, nullable=True)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    password = Column(String(255))
    role = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Authentication schemas
class UserCreate(BaseModel):
    username: str
    password: str
    role: str
    class Config:
        orm_mode = True

class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

class LoginRequest(BaseModel):
    username: str
    password: str
    role: str

# Create tables
def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication settings
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Authentication utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# ML Trainer Class
class FaultDetectionTrainer:
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
                    'probability': True,
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
                    'hidden_layer_sizes': (100,),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'constant',
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }

    def load_data_from_db(self, db: Session):
        try:
            sensor_data = db.query(SensorData).all()
            
            if not sensor_data:
                raise ValueError("No sensor data found in database")
            
            data_list = []
            for record in sensor_data:
                data_list.append({
                    'vibration': record.vibration,
                    'temperature': record.temperature,
                    'pressure': record.pressure,
                    'rms_vibration': record.rms_vibration,
                    'mean_temp': record.mean_temp,
                    'fault_label': record.fault_label
                })
            
            df = pd.DataFrame(data_list)
            logger.info(f"Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise

    def preprocess_data(self, df):
        feature_columns = ['vibration', 'temperature', 'pressure', 'rms_vibration', 'mean_temp']
        X = df[feature_columns]
        y = df['fault_label']
        
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        
        return X, y, feature_columns

    def train_model(self, db: Session, model_type: str, hyperparameters: Dict, 
                   test_size: float = 0.2, model_name: str = None, description: str = None):
        
        if model_type not in self.models_config:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        training_start = datetime.utcnow()
        
        try:
            df = self.load_data_from_db(db)
            X, y, feature_columns = self.preprocess_data(df)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_config = self.models_config[model_type]
            params = model_config['default_params'].copy()
            params.update(hyperparameters)
            
            model = model_config['class'](**params)
            model.fit(X_train_scaled, y_train)
            
            training_end = datetime.utcnow()
            training_duration = (training_end - training_start).total_seconds()
            
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            cm = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_.tolist()))
            
            model_buffer = io.BytesIO()
            scaler_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            joblib.dump(scaler, scaler_buffer)
            model_data = model_buffer.getvalue()
            scaler_data = scaler_buffer.getvalue()
            
            if not model_name:
                model_name = f"{model_config['name']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
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
                description=description
            )
            
            db.add(ml_model)
            db.flush()
            
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
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            db.rollback()
            
            training_history = TrainingHistory(
                model_id=0,
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
        try:
            ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
            if not ml_model:
                raise ValueError(f"Model with ID {model_id} not found")
            
            model = joblib.load(io.BytesIO(ml_model.model_data))
            scaler = joblib.load(io.BytesIO(ml_model.scaler_data))
            
            feature_columns = eval(ml_model.feature_columns)
            
            df_input = pd.DataFrame(input_data)
            
            for col in feature_columns:
                if col not in df_input.columns:
                    raise ValueError(f"Missing feature: {col}")
            
            X_input = df_input[feature_columns]
            X_input_scaled = scaler.transform(X_input)
            
            predictions = model.predict(X_input_scaled)
            
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_input_scaled)
            
            results = []
            for i, pred in enumerate(predictions):
                prob_dict = None
                if probabilities is not None:
                    classes = model.classes_
                    prob_dict = {int(cls): float(prob) for cls, prob in zip(classes, probabilities[i])}
                
                prediction_result = PredictionResult(
                    model_id=model_id,
                    input_data=str(input_data[i]),
                    predicted_fault_label=int(pred),
                    prediction_probability=str(prob_dict) if prob_dict else None
                )
                db.add(prediction_result)
                
                results.append({
                    'input': input_data[i],
                    'predicted_fault_label': int(pred),
                    'probabilities': prob_dict
                })
            
            db.commit()
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

# FastAPI app
app = FastAPI(
    title="Industrial Fault Detection API",
    description="API for uploading sensor data, training ML models, and making fault predictions",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trainer = FaultDetectionTrainer()

@app.on_event("startup")
async def startup_event():
    create_tables()
    
    try:
        db = SessionLocal()
        try:
            existing_admin = db.query(User).filter(User.username == "admin").first()
            if not existing_admin:
                new_user = User(
                    username="admin",
                    password=pwd_context.hash("admin123"),
                    role="admin",
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                db.add(new_user)
                db.commit()
                logger.info("Default admin user created: username=admin, password=admin123, role=admin")
            else:
                logger.info(f"Admin user already exists: role={existing_admin.role}, active={existing_admin.is_active}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error in startup_event: {e}")

@app.get("/")
async def root():
    return {"message": "Industrial Fault Detection API is running"}

@app.get("/health")
async def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "database": "disconnected",
            "error": str(e)
        }

# Helper function to map CSV columns flexibly
def map_csv_columns(df):
    """Map various CSV column formats to standardized names"""
    column_mapping = {}
    
    # Define possible variations for each required column
    column_variations = {
        'timestamp': ['timestamp', 'time', 'date', 'datetime'],
        'vibration': ['vibration', 'vibration (mm/s)', 'vibration_mm_s', 'vib', 'vib_change'],
        'temperature': ['temperature', 'temperature (°c)', 'temperature (c)', 'temp', 'temperature_c', 'temp_change'],
        'pressure': ['pressure', 'pressure (bar)', 'pressure (pa)', 'pressure_bar', 'press', 'pressure_pa'],
        'rms_vibration': ['rms vibration', 'rms_vibration', 'rms', 'rms vib', 'rpm', 'vibration (mm/s)'],
        'mean_temp': ['mean temp', 'mean_temp', 'avg temp', 'avg_temp', 'mean temperature', 'temperature (°c)', 'temperature (c)'],
        'fault_label': ['fault label', 'fault_label', 'fault', 'label', 'class', 'maintenance required', 'maintenance_required']
    }
    
    # Normalize all column names in the dataframe
    normalized_cols = {col: col.strip().lower() for col in df.columns}
    
    # Try to match each required column
    for standard_name, variations in column_variations.items():
        for col_name, normalized in normalized_cols.items():
            if normalized in [v.lower() for v in variations]:
                column_mapping[col_name] = standard_name
                break
    
    # If we still don't have all required fields, try intelligent mapping based on available columns
    # Your CSV has: Timestamp, Temperature (°C), Vibration (mm/s), Pressure (Pa), RPM, Maintenance Required, Temp_Change, Vib_Change
    if 'timestamp' not in column_mapping.values():
        for col in df.columns:
            if 'time' in col.lower():
                column_mapping[col] = 'timestamp'
                break
    
    if 'vibration' not in column_mapping.values():
        for col in df.columns:
            if 'vibration' in col.lower() and 'change' not in col.lower():
                column_mapping[col] = 'vibration'
                break
    
    if 'temperature' not in column_mapping.values():
        for col in df.columns:
            if 'temp' in col.lower() and 'change' not in col.lower():
                column_mapping[col] = 'temperature'
                break
    
    if 'pressure' not in column_mapping.values():
        for col in df.columns:
            if 'pressure' in col.lower():
                column_mapping[col] = 'pressure'
                break
    
    if 'rms_vibration' not in column_mapping.values():
        # Use RPM or Vib_Change as proxy for RMS vibration
        for col in df.columns:
            if col.lower() in ['rpm', 'vib_change', 'vib change']:
                column_mapping[col] = 'rms_vibration'
                break
    
    if 'mean_temp' not in column_mapping.values():
        # Use Temp_Change or duplicate Temperature as mean_temp
        for col in df.columns:
            if col.lower() in ['temp_change', 'temp change']:
                column_mapping[col] = 'mean_temp'
                break
        # If still not found, use temperature column
        if 'mean_temp' not in column_mapping.values():
            for col in df.columns:
                if 'temp' in col.lower() and col not in column_mapping:
                    column_mapping[col] = 'mean_temp'
                    break
    
    if 'fault_label' not in column_mapping.values():
        for col in df.columns:
            if any(word in col.lower() for word in ['maintenance', 'fault', 'label', 'class', 'required']):
                column_mapping[col] = 'fault_label'
                break
    
    return column_mapping

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
        # Read CSV content
        content = await file.read()
        csv_data = pd.read_csv(io.StringIO(content.decode('utf-8')))

        logger.info(f"Original CSV columns: {list(csv_data.columns)}")

        # Map columns flexibly
        column_mapping = map_csv_columns(csv_data)
        
        logger.info(f"Initial column mapping: {column_mapping}")

        # Rename mapped columns
        csv_data = csv_data.rename(columns=column_mapping)
        
        # CREATE MISSING COLUMNS WITH INTELLIGENT DEFAULTS
        
        # 1. Create timestamp if missing
        if 'timestamp' not in csv_data.columns:
            logger.info("Creating timestamp column (missing)")
            csv_data['timestamp'] = pd.date_range(
                start=datetime.utcnow(), 
                periods=len(csv_data), 
                freq='1min'
            )
        
        # 2. Create fault_label from Fault_Type or other columns
        if 'fault_label' not in csv_data.columns:
            logger.info("Creating fault_label column (missing)")
            # Check if there's a Fault_Type or similar column
            fault_type_cols = [col for col in csv_data.columns if 'fault' in col.lower() or 'type' in col.lower()]
            if fault_type_cols:
                # Convert categorical fault types to numeric labels
                fault_col = fault_type_cols[0]
                csv_data['fault_label'] = pd.Categorical(csv_data[fault_col]).codes
                logger.info(f"Created fault_label from {fault_col}: {csv_data['fault_label'].unique()}")
            else:
                # No fault column at all - create dummy labels (all 0 = no fault)
                csv_data['fault_label'] = 0
                logger.info("No fault column found - setting all to 0 (no fault)")
        
        # 3. Create vibration from available data
        if 'vibration' not in csv_data.columns:
            logger.info("Creating vibration column (missing)")
            vib_cols = [col for col in csv_data.columns if 'vib' in col.lower()]
            if vib_cols:
                # Use first vibration-related column or average of FFT vibration features
                if any('fft' in col.lower() for col in vib_cols):
                    csv_data['vibration'] = csv_data[vib_cols].mean(axis=1)
                else:
                    csv_data['vibration'] = csv_data[vib_cols[0]]
            else:
                csv_data['vibration'] = 0.0
        
        # 4. Create temperature from available data
        if 'temperature' not in csv_data.columns:
            logger.info("Creating temperature column (missing)")
            temp_cols = [col for col in csv_data.columns if 'temp' in col.lower()]
            if temp_cols:
                if any('fft' in col.lower() for col in temp_cols):
                    csv_data['temperature'] = csv_data[temp_cols].mean(axis=1)
                else:
                    csv_data['temperature'] = csv_data[temp_cols[0]]
            else:
                csv_data['temperature'] = 0.0
        
        # 5. Create pressure from available data
        if 'pressure' not in csv_data.columns:
            logger.info("Creating pressure column (missing)")
            press_cols = [col for col in csv_data.columns if 'pres' in col.lower()]
            if press_cols:
                if any('fft' in col.lower() for col in press_cols):
                    csv_data['pressure'] = csv_data[press_cols].mean(axis=1)
                else:
                    csv_data['pressure'] = csv_data[press_cols[0]]
            else:
                # Try Flow_Rate as proxy
                flow_cols = [col for col in csv_data.columns if 'flow' in col.lower()]
                if flow_cols:
                    csv_data['pressure'] = csv_data[flow_cols[0]]
                else:
                    csv_data['pressure'] = 0.0
        
        # 6. Create rms_vibration
        if 'rms_vibration' not in csv_data.columns:
            logger.info("Creating rms_vibration column (missing)")
            # Use RPM, Current, or calculate from vibration data
            if 'Current' in csv_data.columns:
                csv_data['rms_vibration'] = csv_data['Current']
            elif 'vibration' in csv_data.columns:
                # Calculate RMS from vibration
                csv_data['rms_vibration'] = csv_data['vibration'].rolling(window=5, min_periods=1).std()
            else:
                csv_data['rms_vibration'] = 0.0
        
        # 7. Create mean_temp
        if 'mean_temp' not in csv_data.columns:
            logger.info("Creating mean_temp column (missing)")
            if 'temperature' in csv_data.columns:
                # Calculate rolling mean
                csv_data['mean_temp'] = csv_data['temperature'].rolling(window=5, min_periods=1).mean()
            else:
                csv_data['mean_temp'] = 0.0

        logger.info(f"Final columns after processing: {list(csv_data.columns)}")
        
        # Convert timestamp column
        csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'], errors='coerce')

        # Insert records into DB
        records_inserted = 0
        for _, row in csv_data.iterrows():
            try:
                sensor_record = SensorData(
                    timestamp=row['timestamp'] if pd.notna(row['timestamp']) else datetime.utcnow(),
                    vibration=float(row['vibration']) if pd.notna(row['vibration']) else 0.0,
                    temperature=float(row['temperature']) if pd.notna(row['temperature']) else 0.0,
                    pressure=float(row['pressure']) if pd.notna(row['pressure']) else 0.0,
                    rms_vibration=float(row['rms_vibration']) if pd.notna(row['rms_vibration']) else 0.0,
                    mean_temp=float(row['mean_temp']) if pd.notna(row['mean_temp']) else 0.0,
                    fault_label=int(row['fault_label']) if pd.notna(row['fault_label']) else 0,
                )
                db.add(sensor_record)
                records_inserted += 1
            except Exception as row_err:
                logger.warning(f"Skipping row due to error: {row_err}")
                continue

        upload_log.records_count = records_inserted
        upload_log.status = "completed"
        db.commit()

        return JSONResponse(
            status_code=200,
            content={
                "message": "CSV uploaded successfully",
                "filename": file.filename,
                "records_inserted": records_inserted,
                "upload_time": datetime.utcnow().isoformat(),
                "columns_created": {
                    "original_columns": len([col for col in csv_data.columns]),
                    "note": "Missing columns were created automatically from available data"
                }
            }
        )

    except HTTPException as http_exc:
        db.rollback()
        upload_log.status = "failed"
        upload_log.error_message = http_exc.detail
        db.commit()
        raise http_exc
    except Exception as e:
        db.rollback()
        error_msg = f"Error processing CSV: {type(e).__name__} - {str(e)}"
        logger.error(f"CSV Processing Error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        upload_log.status = "failed"
        upload_log.error_message = error_msg
        db.commit()
        raise HTTPException(status_code=500, detail=error_msg)

# ML endpoints
@app.get("/ml/models/")
async def get_available_models():
    return {
        "available_models": {
            key: {
                "name": config["name"],
                "default_parameters": config["default_params"]
            }
            for key, config in trainer.models_config.items()
        }
    }

@app.post("/ml/train/")
async def train_model(request: TrainingRequest, db: Session = Depends(get_db)):
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
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
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
                "training_data_count": model.training_data_count
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

@app.get("/ml/predictions/")
async def get_prediction_history(
    model_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
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
                    "predicted_fault_label": pred.predicted_fault_label,
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

@app.get("/sensor-data/")
async def get_sensor_data(
    limit: int = 100, 
    offset: int = 0,
    fault_label: Optional[int] = None,
    db: Session = Depends(get_db)
):
    query = db.query(SensorData)
    
    if fault_label is not None:
        query = query.filter(SensorData.fault_label == fault_label)
    
    data = query.offset(offset).limit(limit).all()
    
    return {
        "data": [
            {
                "id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "vibration": record.vibration,
                "temperature": record.temperature,
                "pressure": record.pressure,
                "rms_vibration": record.rms_vibration,
                "mean_temp": record.mean_temp,
                "fault_label": record.fault_label,
                "upload_datetime": record.upload_datetime.isoformat()
            }
            for record in data
        ],
        "total_records": len(data),
        "offset": offset,
        "limit": limit
    }

@app.get("/upload-logs/")
async def get_upload_logs(db: Session = Depends(get_db)):
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

@app.get("/database-stats/")
async def get_database_stats(db: Session = Depends(get_db)):
    try:
        stats = {
            "sensor_data": {
                "total_records": db.query(SensorData).count(),
                "fault_distribution": {}
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
        
        fault_counts = db.execute(
            text("SELECT fault_label, COUNT(*) as count FROM sensor_data GROUP BY fault_label ORDER BY fault_label")
        ).fetchall()
        
        for fault_label, count in fault_counts:
            stats["sensor_data"]["fault_distribution"][f"fault_{fault_label}"] = count
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@app.delete("/sensor-data/clear/")
async def clear_sensor_data_only(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(SensorData).count()
        db.query(SensorData).delete()
        db.commit()
        
        return {
            "message": f"Cleared {deleted_count} sensor data records",
            "deleted_count": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting sensor data: {str(e)}")

@app.delete("/upload-logs/clear/")
async def clear_upload_logs(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(UploadLog).count()
        db.query(UploadLog).delete()
        db.commit()
        
        return {
            "message": f"Cleared {deleted_count} upload log records",
            "deleted_count": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting upload logs: {str(e)}")

@app.delete("/ml/predictions/clear/")
async def clear_predictions_only(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(PredictionResult).count()
        db.query(PredictionResult).delete()
        db.commit()
        
        return {
            "message": f"Cleared {deleted_count} prediction records",
            "deleted_count": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting predictions: {str(e)}")

@app.delete("/ml/training-history/clear/")
async def clear_training_history(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(TrainingHistory).count()
        db.query(TrainingHistory).delete()
        db.commit()
        
        return {
            "message": f"Cleared {deleted_count} training history records",
            "deleted_count": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting training history: {str(e)}")

@app.delete("/ml/models/clear/")
async def clear_models_only(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(MLModel).count()
        db.query(MLModel).delete()
        db.commit()
        
        return {
            "message": f"Cleared {deleted_count} ML models",
            "deleted_count": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting models: {str(e)}")

@app.delete("/ml/clear-all/")
async def clear_all_ml_data(db: Session = Depends(get_db)):
    try:
        models_count = db.query(MLModel).count()
        predictions_count = db.query(PredictionResult).count()
        training_count = db.query(TrainingHistory).count()
        
        db.query(PredictionResult).delete()
        db.query(TrainingHistory).delete()
        db.query(MLModel).delete()
        db.commit()
        
        return {
            "message": "ALL ML data cleared successfully",
            "deleted_records": {
                "models": models_count,
                "predictions": predictions_count,
                "training_history": training_count,
                "total_ml_records": models_count + predictions_count + training_count
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error clearing ML data: {str(e)}")

@app.delete("/clear-everything/")
async def clear_everything(confirm: str = Query(..., description="Type 'YES_DELETE_EVERYTHING' to confirm")):
    if confirm != "YES_DELETE_EVERYTHING":
        raise HTTPException(
            status_code=400, 
            detail="Confirmation required. Add ?confirm=YES_DELETE_EVERYTHING to the URL"
        )
    
    try:
        db = SessionLocal()
        
        sensor_count = db.query(SensorData).count()
        upload_count = db.query(UploadLog).count()
        models_count = db.query(MLModel).count()
        predictions_count = db.query(PredictionResult).count()
        training_count = db.query(TrainingHistory).count()
        
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
            }
        }
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.delete("/sensor-data/")
async def clear_sensor_data_legacy(db: Session = Depends(get_db)):
    return await clear_sensor_data_only(db)

@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user = User(
        username=user.username,
        password=get_password_hash(user.password),
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/auth/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"Login attempt: username={request.username}, role={request.role}")
        
        user = db.query(User).filter(User.username == request.username).first()
        
        if not user:
            logger.warning(f"User not found: {request.username}")
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        logger.info(f"User found: username={user.username}, role={user.role}, active={user.is_active}")
        
        if not verify_password(request.password, user.password):
            logger.warning(f"Invalid password for user: {request.username}")
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        logger.info(f"Password verified for user: {request.username}")
        
        if not user.is_active:
            logger.warning(f"User inactive: {request.username}")
            raise HTTPException(status_code=401, detail="Account is inactive")
        
        if user.role.lower() != request.role.lower():
            logger.warning(f"Role mismatch: DB={user.role}, Request={request.role}")
            raise HTTPException(status_code=401, detail=f"Role mismatch. Your role is '{user.role}'")
        
        logger.info(f"Role verified: {user.role}")
        
        access_token = create_access_token(data={"sub": user.username, "role": user.role})
        
        logger.info(f"Login successful: user={user.username}, role={user.role}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "username": user.username,
                "role": user.role
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)