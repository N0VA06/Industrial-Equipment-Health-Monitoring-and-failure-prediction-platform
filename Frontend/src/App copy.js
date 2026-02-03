import './index.css';
import mermaid from 'mermaid';
import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Upload, 
  Brain, 
  BarChart3, 
  AlertTriangle, 
  CheckCircle, 
  Cpu, 
  Database, 
  Zap, 
  Settings, 
  TrendingUp, 
  Shield,
  Play,
  Pause,
  RefreshCw,
  Eye,
  Trash2,
  Plus,
  Download,
  FileText,
  Users,
  Clock,
  Target,
  Layers,
  GitBranch,
  Box,
  X
} from 'lucide-react';

// API base URL - modify this to match your backend
const API_BASE_URL = 'http://localhost:8000/';  // Adjust port as needed

// API utility functions
const api = {
  // Upload data
  uploadData: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/upload-csv/', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    return response.json();
  },
  // Get database stats
  getStats: async () => {
    const response = await fetch('http://localhost:8000/database-stats/');
    if (!response.ok) {
      throw new Error(`Failed to fetch stats: ${response.statusText}`);
    }
    return response.json();
  },

  // Get all trained models
  getModels: async () => {
    const response = await fetch('http://localhost:8000/ml/models/trained/');
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.statusText}`);
    }
    return response.json();
  },

  // Train new model
  trainModel: async (modelData) => {
    const response = await fetch('http://localhost:8000/ml/train/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_type: modelData.model_type,
        model_name: modelData.name,
        test_size: modelData.test_size,
        description: modelData.description,
        hyperparameters: {}
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Training failed: ${response.statusText}`);
    }
    return response.json();
  },

  // Make predictions
  predict: async (modelId, sensorData) => {
    const response = await fetch('http://localhost:8000/ml/predict/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_id: modelId,
        data: [sensorData]
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Prediction failed: ${response.statusText}`);
    }
    return response.json();
  },
  deleteModel: async (modelId) => {
    const response = await fetch(`http://localhost:8000/ml/models/${modelId}/`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error(`Delete failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  clearData: async () => {
    const response = await fetch('http://localhost:8000/clear-data/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`Clear data failed: ${response.statusText}`);
    }
    
    return response.json();
  }

};


// Initialize mermaid globally
mermaid.initialize({ 
  startOnLoad: false,
  theme: 'dark',
  themeVariables: {
    darkMode: true,
    primaryColor: '#6366f1',
    primaryTextColor: '#ffffff',
    primaryBorderColor: '#4f46e5',
    lineColor: '#6b7280',
    secondaryColor: '#374151',
    tertiaryColor: '#1f2937'
  }
});

const MermaidDiagram = ({ chart, title }) => {
  const [svg, setSvg] = useState('');
  const elementId = `mermaid-${Math.random().toString(36).substr(2, 9)}`;

  useEffect(() => {
    const renderDiagram = async () => {
      try {
        const { svg } = await mermaid.render(elementId, chart);
        setSvg(svg);
      } catch (error) {
        console.error('Failed to render diagram:', error);
      }
    };

    renderDiagram();
  }, [chart, elementId]);

  return (
    <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
      <h3 className="text-white text-xl font-semibold mb-6 flex items-center gap-3">
        <GitBranch className="text-indigo-400" size={24} />
        {title}
      </h3>
      <div 
        className="bg-gray-800/30 rounded-2xl p-6 overflow-auto"
        style={{ minHeight: '400px' }}
      >
        {svg ? (
          <div dangerouslySetInnerHTML={{ __html: svg }} />
        ) : (
          <div className="text-gray-400 text-center">Loading diagram...</div>
        )}
      </div>
    </div>
  );
};

// Model visualization modal
const ModelVisualizationModal = ({ model, isOpen, onClose }) => {
  const [activeView, setActiveView] = useState('er');

  const erDiagram = `
    erDiagram
        SENSOR_DATA {
            int id PK
            datetime timestamp
            float vibration
            float temperature
            float pressure
            float rms_vibration
            float mean_temp
            int fault_label
            datetime upload_datetime
        }
        
        ML_MODELS {
            int id PK
            string name
            string model_type
            longblob model_data
            longblob scaler_data
            text feature_columns
            float accuracy
            float precision
            float recall
            float f1_score
            text hyperparameters
            int training_data_count
            datetime created_at
            text description
            boolean is_active
        }
        
        PREDICTION_RESULTS {
            int id PK
            int model_id FK
            text input_data
            int predicted_fault_label
            text prediction_probability
            datetime created_at
        }
        
        TRAINING_HISTORY {
            int id PK
            int model_id FK
            datetime training_started
            datetime training_completed
            float training_duration_seconds
            int dataset_size
            float test_size
            text confusion_matrix
            text classification_report
            text feature_importance
            string status
            text error_message
        }
        
        UPLOAD_LOG {
            int id PK
            string filename
            datetime upload_datetime
            int records_count
            string status
            text error_message
        }
        
        ML_MODELS ||--o{ PREDICTION_RESULTS : "generates"
        ML_MODELS ||--o{ TRAINING_HISTORY : "tracks"
        SENSOR_DATA }o--|| UPLOAD_LOG : "uploaded_via"
  `;

  const umlDiagram = `
    classDiagram
        class SensorData {
            +int id
            +DateTime timestamp
            +float vibration
            +float temperature
            +float pressure
            +float rms_vibration
            +float mean_temp
            +int fault_label
            +DateTime upload_datetime
        }
        
        class MLModel {
            +int id
            +string name
            +string model_type
            +bytes model_data
            +bytes scaler_data
            +list feature_columns
            +float accuracy
            +float precision
            +float recall
            +float f1_score
            +dict hyperparameters
            +int training_data_count
            +DateTime created_at
            +string description
            +boolean is_active
            +train_model()
            +predict()
            +save_to_db()
        }
        
        class FaultDetectionTrainer {
            +dict models_config
            +load_data_from_db()
            +preprocess_data()
            +train_model()
            +predict()
            +evaluate_model()
        }
        
        class PredictionResult {
            +int id
            +int model_id
            +dict input_data
            +int predicted_fault_label
            +dict prediction_probability
            +DateTime created_at
        }
        
        class TrainingHistory {
            +int id
            +int model_id
            +DateTime training_started
            +DateTime training_completed
            +float training_duration_seconds
            +int dataset_size
            +float test_size
            +matrix confusion_matrix
            +string classification_report
            +dict feature_importance
            +string status
        }
        
        FaultDetectionTrainer --> MLModel : creates
        MLModel --> PredictionResult : generates
        MLModel --> TrainingHistory : records
        SensorData --> FaultDetectionTrainer : feeds_data
  `;

  const architectureDiagram = `
    graph TB
        subgraph "Data Layer"
            CSV[CSV Files] --> DB[(MySQL Database)]
            DB --> SD[Sensor Data Table]
            DB --> ML[ML Models Table]
            DB --> PR[Prediction Results]
            DB --> TH[Training History]
        end
        
        subgraph "API Layer"
            API[FastAPI Application]
            API --> UE[Upload Endpoint]
            API --> TE[Training Endpoint]
            API --> PE[Prediction Endpoint]
            API --> AE[Analytics Endpoint]
        end
        
        subgraph "ML Pipeline"
            DL[Data Loader] --> PP[Preprocessing]
            PP --> FS[Feature Scaling]
            FS --> MT[Model Training]
            MT --> ME[Model Evaluation]
            ME --> MS[Model Storage]
        end
        
        subgraph "Frontend"
            UI[React Dashboard] --> UP[Upload Interface]
            UI --> MP[Model Management]
            UI --> PI[Prediction Interface]
            UI --> AN[Analytics View]
        end
        
        CSV --> UE
        UE --> DL
        MS --> PE
        PE --> PR
        UI --> API
        
        style DB fill:#1f2937,stroke:#4f46e5
        style API fill:#1f2937,stroke:#059669
        style UI fill:#1f2937,stroke:#dc2626
        style MT fill:#1f2937,stroke:#7c3aed
  `;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-3xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-white text-2xl font-bold">System Architecture & Database Design</h2>
            {model && <p className="text-gray-400">Viewing diagrams for {model.name}</p>}
          </div>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-white p-2 hover:bg-gray-800 rounded-xl transition-colors"
          >
            ✕
          </button>
        </div>
        
        <div className="p-6">
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setActiveView('er')}
              className={`px-4 py-2 rounded-xl font-medium transition-all ${
                activeView === 'er'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              <Database className="inline mr-2" size={18} />
              ER Diagram
            </button>
            <button
              onClick={() => setActiveView('uml')}
              className={`px-4 py-2 rounded-xl font-medium transition-all ${
                activeView === 'uml'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              <Box className="inline mr-2" size={18} />
              UML Diagram
            </button>
            <button
              onClick={() => setActiveView('architecture')}
              className={`px-4 py-2 rounded-xl font-medium transition-all ${
                activeView === 'architecture'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              <Layers className="inline mr-2" size={18} />
              Architecture
            </button>
          </div>
          
          <div className="max-h-[60vh] overflow-auto">
            {activeView === 'er' && (
              <MermaidDiagram 
                chart={erDiagram} 
                title="Entity Relationship Diagram" 
              />
            )}
            {activeView === 'uml' && (
              <MermaidDiagram 
                chart={umlDiagram} 
                title="UML Class Diagram" 
              />
            )}
            {activeView === 'architecture' && (
              <MermaidDiagram 
                chart={architectureDiagram} 
                title="System Architecture Diagram" 
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};


const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [models, setModels] = useState([]);
  const [stats, setStats] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [showVisualization, setShowVisualization] = useState(false);
  const [selectedModelForViz, setSelectedModelForViz] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  
  useEffect(() => {
  loadData();
}, []); // Load data when component mounts

  // Load Mermaid library
  useEffect(() => {
    if (typeof window !== 'undefined' && !window.mermaid) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js';
      script.onload = () => {
        window.mermaid?.initialize({ startOnLoad: false });
      };
      document.head.appendChild(script);
    }
  }, []);
  // Prediction form state
    const [predictionData, setPredictionData] = useState({
      vibration: '',
      temperature: '',
      pressure: '',
      rms_vibration: '',
      mean_temp: ''
    });
  
    // Training form state
    const [trainingForm, setTrainingForm] = useState({
      model_type: 'random_forest',
      name: '',
      test_size: 0.2,
      description: ''
    });


  // Mock data - replace with actual API calls
  useEffect(() => {
    // Simulate loading stats
    setTimeout(() => {
      setStats({
        sensor_data: { total_records: 15420, fault_distribution: { fault_0: 12500, fault_1: 2920 } },
        ml_models: { total_models: 5, inactive_models: 1 },
        predictions: { total_predictions: 1250 },
        uploads: { total_uploads: 23, successful_uploads: 22, failed_uploads: 1 }
      });
      setModels([
        { id: 1, name: 'Production_RF_Model', model_type: 'random_forest', accuracy: 0.948, f1_score: 0.943, created_at: '2025-08-20T10:30:00' },
        { id: 2, name: 'SVM_Classifier_v2', model_type: 'svm', accuracy: 0.921, f1_score: 0.918, created_at: '2025-08-19T14:15:00' },
        { id: 3, name: 'Neural_Network_Deep', model_type: 'neural_network', accuracy: 0.956, f1_score: 0.951, created_at: '2025-08-18T09:45:00' }
      ]);
    }, 1000);
  }, []);
  
  const TabButton = ({ id, icon: Icon, label, count }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-3 px-6 py-4 rounded-2xl font-semibold transition-all duration-300 ${
        activeTab === id
          ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-2xl shadow-purple-500/25 transform scale-[1.02]'
          : 'text-gray-400 hover:text-white hover:bg-gray-800/50 hover:shadow-lg'
      }`}
    >
      <Icon size={20} />
      <span>{label}</span>
      {count && (
        <span className={`px-2 py-1 text-xs rounded-full ${
          activeTab === id ? 'bg-white/20' : 'bg-gray-700'
        }`}>
          {count}
        </span>
      )}
    </button>
  );

  const handleVisualize = (model) => {
    setSelectedModelForViz(model);
    setShowVisualization(true);
  };
  const loadData = async () => {
  try {
    setIsLoading(true);
    
    // Load stats and models in parallel
    const [statsData, modelsData] = await Promise.all([
      api.getStats(),
      api.getModels()
    ]);
    
    if (statsData) setStats(statsData);
    if (modelsData) setModels(modelsData);
    
  } catch (err) {
    console.error('Error loading data:', err);
    setError('Failed to connect to backend. Please ensure the backend is running.');
  } finally {
    setIsLoading(false);
  }
};
  
    const handleFileUpload = async (event) => {
      const file = event.target.files[0];
      if (!file) return;
  
      try {
        setIsLoading(true);
        setUploadProgress(0);
        setError(null);
  
        // Simulate upload progress
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return prev;
            }
            return prev + 10;
          });
        }, 200);
  
        const result = await api.uploadData(file);
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        setTimeout(() => {
          setUploadProgress(0);
          setSuccess('Data uploaded successfully!');
          loadData(); // Reload data after upload
        }, 500);
  
      } catch (err) {
        console.error('Upload error:', err);
        setError(`Upload failed: ${err.message}`);
        setUploadProgress(0);
      } finally {
        setIsLoading(false);
      }
    };
  
    const handleTrainModel = async () => {
  if (!trainingForm.name.trim()) {
    setError('Please enter a model name');
    return;
  }

  try {
    setIsTraining(true);
    setError(null);

    const result = await api.trainModel(trainingForm);
    
    setSuccess(`Model "${trainingForm.name}" trained successfully!`);
    loadData(); // Reload models list

    setTrainingForm({
      model_type: 'random_forest',
      name: '',
      test_size: 0.2,
      description: ''
    });

  } catch (err) {
    console.error('Training error:', err);
    setError(`Training failed: ${err.message}`);
  } finally {
    setIsTraining(false);
  }
};
  
    const handlePredict = async () => {
  if (!selectedModel) {
    setError('Please select a model first');
    return;
  }

  try {
    setIsLoading(true);
    setError(null);

    const result = await api.predict(selectedModel.id, {
      vibration: parseFloat(predictionData.vibration),
      temperature: parseFloat(predictionData.temperature),
      pressure: parseFloat(predictionData.pressure),
      rms_vibration: parseFloat(predictionData.rms_vibration),
      mean_temp: parseFloat(predictionData.mean_temp)
    });
    
    setPredictions([result.predictions[0], ...predictions.slice(0, 8)]);
    setSuccess('Prediction completed successfully!');

  } catch (err) {
    console.error('Prediction error:', err);
    setError(`Prediction failed: ${err.message}`);
  } finally {
    setIsLoading(false);
  }
};
  
    const handleDeleteModel = async (modelId) => {
      if (!window.confirm('Are you sure you want to delete this model?')) return;
  
      try {
        await api.deleteModel(modelId);
        setModels(prev => prev.filter(m => m.id !== modelId));
        setSuccess('Model deleted successfully!');
        
        if (selectedModel?.id === modelId) {
          setSelectedModel(null);
        }
      } catch (err) {
        console.error('Delete error:', err);
        setError(`Delete failed: ${err.message}`);
      }
    };
    
  
    // Clear messages after 5 seconds
    useEffect(() => {
      if (error || success) {
        const timer = setTimeout(() => {
          setError(null);
          setSuccess(null);
        }, 5000);
        return () => clearTimeout(timer);
      }
    }, [error, success]);
    const handleExportCSV = async () => {
  try {
    const response = await fetch('http://localhost:8000/export-csv/');
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sensor_data.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (error) {
    setError('Failed to export CSV');
  }
};

const handleExportModels = async () => {
  try {
    const response = await fetch('http://localhost:8000/export-models/');
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ml_models.zip';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (error) {
    setError('Failed to export models');
  }
};
const handleViewModelDetails = async (modelId) => {
  try {
    const response = await fetch(`http://localhost:8000/ml/models/${modelId}/`);
    if (!response.ok) throw new Error('Failed to fetch model details');
    const details = await response.json();
    // You can set this to state and show in a modal
    console.log(details);
  } catch (error) {
    setError('Failed to fetch model details');
  }
};
const handleClearData = async () => {
  if (!window.confirm('Are you sure you want to clear all sensor data? This action cannot be undone.')) {
    return;
  }

  try {
    const response = await fetch('http://localhost:8000/clear-data/', {
      method: 'POST'
    });
    if (!response.ok) throw new Error('Failed to clear data');
    setSuccess('Data cleared successfully');
    loadData(); // Reload stats
  } catch (error) {
    setError('Failed to clear data');
  }
};


  

  const MetricCard = ({ icon: Icon, title, value, change, color = 'blue' }) => {
    const colors = {
      blue: 'from-blue-600 to-cyan-600',
      purple: 'from-purple-600 to-pink-600',
      green: 'from-green-600 to-emerald-600',
      orange: 'from-orange-600 to-red-600'
    };

    return (
      <div className="relative overflow-hidden bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-purple-500/10 transition-all duration-300 hover:scale-[1.02] group">
        <div className={`absolute inset-0 bg-gradient-to-br ${colors[color]} opacity-5 group-hover:opacity-10 transition-opacity duration-300`}></div>
        <div className="relative z-10">
          <div className="flex items-center justify-between mb-4">
            <div className={`p-3 rounded-2xl bg-gradient-to-br ${colors[color]} shadow-lg`}>
              <Icon className="text-white" size={24} />
            </div>
            {change && (
              <span className={`text-sm font-medium ${change > 0 ? 'text-green-400' : 'text-red-400'}`}>
                {change > 0 ? '+' : ''}{change}%
              </span>
            )}
          </div>
          <h3 className="text-gray-400 text-sm font-medium mb-1">{title}</h3>
          <p className="text-white text-2xl font-bold">{value}</p>
        </div>
      </div>
    );
  };

  const ModelCard = ({ model }) => (
    <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-indigo-500/10 transition-all duration-300 hover:scale-[1.01] group">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-white font-semibold text-lg mb-2">{model.name}</h3>
          <span className="px-3 py-1 bg-indigo-600/20 text-indigo-400 text-xs font-medium rounded-full border border-indigo-600/30">
            {model.model_type.replace('_', ' ').toUpperCase()}
          </span>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={() => handleVisualize(model)}
            className="p-2 text-gray-400 hover:text-indigo-400 hover:bg-gray-800 rounded-xl transition-colors"
            title="Visualize Architecture"
          >
            <GitBranch size={16} />
          </button>
          <button 
  onClick={() => handleDeleteModel(model.id)}
  className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded-xl transition-colors"
  title="Delete Model"
>
  <Trash2 size={16} />
</button>
          <button 
  onClick={() => handleViewModelDetails(model.id)}
  className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-xl transition-colors"
>
  <Eye size={16} />
</button>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-gray-400 text-xs mb-1">Accuracy</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-800 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full"
                style={{ width: `${model.accuracy * 100}%` }}
              ></div>
            </div>
            <span className="text-green-400 text-sm font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
          </div>
        </div>
        <div>
          <p className="text-gray-400 text-xs mb-1">F1 Score</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-800 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full"
                style={{ width: `${model.f1_score * 100}%` }}
              ></div>
            </div>
            <span className="text-blue-400 text-sm font-medium">{(model.f1_score * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
      
      <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
        <span>Created: {new Date(model.created_at).toLocaleDateString()}</span>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span>Active</span>
        </div>
      </div>
      
      <button 
        onClick={() => setSelectedModel(model)}
        className="w-full py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300"
      >
        Use for Prediction
      </button>
    </div>
  );
  const NotificationBar = () => {
    if (!error && !success) return null;

    return (
      <div className={`fixed top-4 right-4 z-50 p-4 rounded-2xl shadow-2xl border backdrop-blur-xl max-w-md ${
        error 
          ? 'bg-red-900/50 border-red-800 text-red-200' 
          : 'bg-green-900/50 border-green-800 text-green-200'
      }`}>
        <div className="flex items-start gap-3">
          {error ? (
            <AlertTriangle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
          ) : (
            <CheckCircle className="text-green-400 flex-shrink-0 mt-0.5" size={20} />
          )}
          <div className="flex-1">
            <p className="font-medium">{error ? 'Error' : 'Success'}</p>
            <p className="text-sm opacity-90">{error || success}</p>
          </div>
          <button 
            onClick={() => {
              setError(null);
              setSuccess(null);
            }}
            className="text-gray-400 hover:text-white"
          >
            <X size={16} />
          </button>
        </div>
      </div>
    );
  };
  const OverviewTab = () => (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-br from-indigo-900/50 via-purple-900/30 to-pink-900/20 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <div 
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
          }}
        ></div>
        <div className="relative z-10">
          <div className="flex items-center gap-4 mb-6">
            <div className="p-4 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-3xl shadow-2xl shadow-indigo-500/25">
              <Shield className="text-white" size={32} />
            </div>
            <div>
              <h1 className="text-white text-3xl font-bold mb-2">Industrial Fault Detection System</h1>
              <p className="text-gray-400">AI-powered predictive maintenance and fault analysis</p>
            </div>
            <div className="ml-auto">
              <button className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300">
                Browse Files
                <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={isLoading}
              />
              </button>
            </div>
          </div>
          
          {uploadProgress > 0 && (
            <div className="mt-4">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="bg-gray-800 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}
          
          <div className="space-y-4">
            <h3 className="text-white font-semibold mb-4">Required CSV Format</h3>
            <div className="bg-gray-800/50 rounded-2xl p-4 text-sm font-mono text-gray-300">
              <div className="text-indigo-400 mb-2">CSV Headers Required:</div>
              <div className="space-y-1 text-xs">
                <div>• Timestamp</div>
                <div>• Vibration (mm/s)</div>
                <div>• Temperature (°C)</div>
                <div>• Pressure (bar)</div>
                <div>• RMS Vibration</div>
                <div>• Mean Temp</div>
                <div>• Fault Label</div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 backdrop-blur-xl border border-blue-800/30 rounded-2xl p-4">
              <h4 className="text-blue-400 font-medium mb-2">💡 Tips for Best Results</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• Ensure timestamps are in proper format</li>
                <li>• Keep data clean and consistent</li>
                <li>• Include fault labels (0=Normal, 1=Fault)</li>
                <li>• Recommended: 1000+ records for training</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Action Cards */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-white text-2xl font-bold">Quick Actions</h2>
        <button 
          onClick={() => handleVisualize(null)}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300"
        >
          <GitBranch size={18} />
          View Architecture
        </button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="text-center p-4 bg-white/5 rounded-2xl border border-white/10">
          <Activity className="mx-auto text-indigo-400 mb-2" size={24} />
          <p className="text-white font-semibold">Real-time Monitoring</p>
        </div>
        <div className="text-center p-4 bg-white/5 rounded-2xl border border-white/10">
          <Brain className="mx-auto text-purple-400 mb-2" size={24} />
          <p className="text-white font-semibold">ML Predictions</p>
        </div>
        <div className="text-center p-4 bg-white/5 rounded-2xl border border-white/10">
          <BarChart3 className="mx-auto text-green-400 mb-2" size={24} />
          <p className="text-white font-semibold">Analytics Dashboard</p>
        </div>
        <div className="text-center p-4 bg-white/5 rounded-2xl border border-white/10">
          <AlertTriangle className="mx-auto text-orange-400 mb-2" size={24} />
          <p className="text-white font-semibold">Early Warnings</p>
        </div>
      </div>

      {/* Stats Grid */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard 
            icon={Database}
            title="Sensor Records"
            value={stats.sensor_data.total_records.toLocaleString()}
            change={15}
            color="blue"
          />
          <MetricCard 
            icon={Brain}
            title="Active Models"
            value={stats.ml_models.total_models}
            change={25}
            color="purple"
          />
          <MetricCard 
            icon={Target}
            title="Total Predictions"
            value={stats.predictions.total_predictions.toLocaleString()}
            change={8}
            color="green"
          />
          <MetricCard 
            icon={Upload}
            title="Upload Success Rate"
            value={`${((stats.uploads.successful_uploads / stats.uploads.total_uploads) * 100).toFixed(1)}%`}
            change={3}
            color="orange"
          />
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300 group cursor-pointer">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-2xl shadow-lg group-hover:shadow-blue-500/25">
              <Upload className="text-white" size={24} />
            </div>
            <div>
              <h3 className="text-white font-semibold">Upload Sensor Data</h3>
              <p className="text-gray-400 text-sm">Import CSV files with sensor readings</p>
            </div>
          </div>
          <button 
            onClick={() => setActiveTab('upload')}
            className="w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl font-medium hover:shadow-lg transition-all duration-300"
          >
            Start Upload
          </button>
        </div>

        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-purple-500/10 transition-all duration-300 group cursor-pointer">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl shadow-lg group-hover:shadow-purple-500/25">
              <Brain className="text-white" size={24} />
            </div>
            <div>
              <h3 className="text-white font-semibold">Train New Model</h3>
              <p className="text-gray-400 text-sm">Create ML models for fault detection</p>
            </div>
          </div>
          <button 
            onClick={() => setActiveTab('models')}
            className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:shadow-lg transition-all duration-300"
          >
            Train Model
          </button>
        </div>

        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-green-500/10 transition-all duration-300 group cursor-pointer">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-green-600 to-emerald-600 rounded-2xl shadow-lg group-hover:shadow-green-500/25">
              <Zap className="text-white" size={24} />
            </div>
            <div>
              <h3 className="text-white font-semibold">Make Predictions</h3>
              <p className="text-gray-400 text-sm">Analyze data for potential faults</p>
            </div>
          </div>
          <button 
            onClick={() => setActiveTab('predict')}
            className="w-full py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl font-medium hover:shadow-lg transition-all duration-300"
          >
            Predict Faults
          </button>
        </div>
      </div>
    </div>
  );

  const UploadTab = () => (
    <div className="space-y-8">
      <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <h2 className="text-white text-2xl font-bold mb-6 flex items-center gap-3">
          <Upload className="text-indigo-400" size={28} />
          Upload Sensor Data
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <div className="border-2 border-dashed border-gray-700 rounded-3xl p-8 text-center hover:border-indigo-500 transition-colors duration-300 group cursor-pointer relative overflow-hidden">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={isLoading}
              />
              <div className="p-4 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-3xl shadow-2xl shadow-indigo-500/25 w-16 h-16 mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                <FileText className="text-white mx-auto" size={32} />
              </div>
              <h3 className="text-white font-semibold mb-2">Drop your CSV file here</h3>
              <p className="text-gray-400 text-sm mb-4">or click to browse files</p>
              <div className={`px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300 inline-block ${isLoading ? 'opacity-50' : ''}`}>
                {isLoading ? 'Uploading...' : 'Browse Files'}

              </div>
            </div>
            
            {uploadProgress > 0 && (
              <div className="mt-6 bg-gray-800/50 rounded-2xl p-6">
                <div className="flex justify-between text-sm text-gray-400 mb-2">
                  <span>Uploading sensor_data.csv...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="bg-gray-700 rounded-full h-3">
                  <div 
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-gray-400 text-xs mt-2">Processing 2,450 records...</p>
              </div>
            )}
          </div>
          
          <div className="space-y-6">
            <div className="bg-gray-800/50 rounded-2xl p-6">
              <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                <FileText className="text-indigo-400" size={20} />
                CSV Format Requirements
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Timestamp</span>
                  <span className="text-green-400">✓ Required</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Vibration</span>
                  <span className="text-green-400">✓ Required</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Temperature</span>
                  <span className="text-green-400">✓ Required</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Pressure</span>
                  <span className="text-green-400">✓ Required</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">RMS Vibration</span>
                  <span className="text-green-400">✓ Required</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Mean Temperature</span>
                  <span className="text-green-400">✓ Required</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Fault Label</span>
                  <span className="text-green-400">✓ Required</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-green-900/20 to-emerald-900/20 backdrop-blur-xl border border-green-800/30 rounded-2xl p-4">
              <h4 className="text-green-400 font-medium mb-2 flex items-center gap-2">
                <CheckCircle size={18} />
                Upload Guidelines
              </h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• File size limit: 50MB</li>
                <li>• Supported format: CSV only</li>
                <li>• Minimum records: 100</li>
                <li>• Fault labels: 0 (Normal), 1 (Fault)</li>
              </ul>
            </div>
            
            <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 backdrop-blur-xl border border-blue-800/30 rounded-2xl p-4">
              <h4 className="text-blue-400 font-medium mb-2">Recent Uploads</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">machine_data_2025.csv</span>
                  <span className="text-green-400">Success</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">sensor_readings.csv</span>
                  <span className="text-green-400">Success</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">fault_data.csv</span>
                  <span className="text-red-400">Failed</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const ModelsTab = () => (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-white text-2xl font-bold flex items-center gap-3">
          <Brain className="text-purple-400" size={28} />
          ML Models
        </h2>
        <button onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })}
        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-purple-500/25 transition-all duration-300">
          <Plus size={20} />
          Train New Model
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <ModelCard key={model.id} model={model} />
        ))}
      </div>

      {/* Model Training Form */}
      <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <h3 className="text-white text-xl font-semibold mb-6">Train New Model</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-gray-300 font-medium mb-2">Model Type</label>
            <select value={trainingForm.model_type}
              onChange={(e) => setTrainingForm(prev => ({ ...prev, model_type: e.target.value }))} className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors">
              <option value="random_forest">Random Forest</option>
              <option value="svm">Support Vector Machine</option>
              <option value="neural_network">Neural Network</option>
              <option value="gradient_boosting">Gradient Boosting</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </div>
          
          <div>
            <label className="block text-gray-300 font-medium mb-2">Model Name</label>
            <input 
              type="text" 
              placeholder="Enter model name..."
              value={trainingForm.name}
              onChange={(e) => setTrainingForm(prev => ({ ...prev, name: e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </div>
          
          <div>
            <label className="block text-gray-300 font-medium mb-2">Test Size</label>
            <input 
              type="number" 
              placeholder="0.2"
              min="0.1"
              max="0.5"
              step="0.1"
              value={trainingForm.test_size}
              onChange={(e) => setTrainingForm(prev => ({ ...prev, test_size: parseFloat(e.target.value) }))}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </div>
          
          <div>
            <label className="block text-gray-300 font-medium mb-2">Description</label>
            <input 
              type="text" 
              placeholder="Optional description..."
              value={trainingForm.description}
              onChange={(e) => setTrainingForm(prev => ({ ...prev, description: e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </div>
        </div>
        
        <button 
  onClick={handleTrainModel}
  className="mt-6 px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-purple-500/25 transition-all duration-300 flex items-center gap-2"
  disabled={isLoading || !trainingForm.name.trim()}
>
  {isTraining ? (
    <>
      <RefreshCw className="animate-spin" size={20} />
      Training...
    </>
  ) : (
    <>
      <Play size={20} />
      Start Training
    </>
  )}
</button>
      </div>
    </div>
  );

  const PredictTab = () => (
    <div className="space-y-8">
      <h2 className="text-white text-2xl font-bold flex items-center gap-3">
        <Zap className="text-green-400" size={28} />
        Fault Prediction
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
          <h3 className="text-white text-xl font-semibold mb-6">Input Sensor Data</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-gray-300 font-medium mb-2">Vibration (mm/s)</label>
              <input 
                type="number" 
                placeholder="0.0"
                step="0.01"
                value={predictionData.vibration}
                onChange={(e) => setPredictionData(prev => ({ ...prev, vibration: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-green-500 transition-colors"
              />
            </div>
            
            <div>
              <label className="block text-gray-300 font-medium mb-2">Temperature (°C)</label>
              <input 
                type="number" 
                placeholder="0.0"
                step="0.1"
                value={predictionData.temperature}
                onChange={(e) => setPredictionData(prev => ({ ...prev, temperature: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-green-500 transition-colors"
              />
            </div>
            
            <div>
              <label className="block text-gray-300 font-medium mb-2">Pressure (bar)</label>
              <input 
                type="number" 
                placeholder="0.0"
                step="0.1"
                value={predictionData.pressure}
                onChange={(e) => setPredictionData(prev => ({ ...prev, pressure: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-green-500 transition-colors"
              />
            </div>
            
            <div>
              <label className="block text-gray-300 font-medium mb-2">RMS Vibration</label>
              <input 
                type="number" 
                placeholder="0.0"
                step="0.01"
                value={predictionData.rms_vibration}
                onChange={(e) => setPredictionData(prev => ({ ...prev, rms_vibration: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-green-500 transition-colors"
              />
            </div>
            
            <div>
              <label className="block text-gray-300 font-medium mb-2">Mean Temperature</label>
              <input 
                type="number" 
                placeholder="0.0"
                step="0.1"
                value={predictionData.mean_temp}
                onChange={(e) => setPredictionData(prev => ({ ...prev, mean_temp: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-green-500 transition-colors"
              />
            </div>
          </div>
          {/* Model Selection */}
                    <div className="mt-6">
                      <label className="block text-gray-300 font-medium mb-2">Select Model</label>
                      <select 
                        value={selectedModel?.id || ''}
                        onChange={(e) => {
                          const model = models.find(m => m.id === parseInt(e.target.value));
                          setSelectedModel(model || null);
                        }}
                        className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-green-500 transition-colors"
                      >
                        <option value="">Select a model...</option>
                        {models.filter(m => m.is_active).map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name} - {(model.accuracy * 100).toFixed(1)}%
                          </option>
                        ))}
                      </select>
                    </div>
                    
                    {selectedModel && (
                      <div className="mt-6 p-4 bg-gradient-to-br from-green-900/20 to-emerald-900/20 backdrop-blur-xl border border-green-800/30 rounded-2xl">
                        <div className="flex items-center gap-3 mb-2">
                          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                          <span className="text-green-400 font-medium">Selected Model</span>
                        </div>
                        <p className="text-white font-semibold">{selectedModel.name}</p>
                        <p className="text-gray-400 text-sm">Accuracy: {(selectedModel.accuracy * 100).toFixed(1)}%</p>
                      </div>
                    )}
                    
                    <button 
                      onClick={handlePredict}
                      className="w-full mt-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-green-500/25 transition-all duration-300 flex items-center justify-center gap-2"
                      disabled={isLoading || !selectedModel || Object.values(predictionData).some(v => v === '')}
                    >
                      {isLoading ? (
                        <>
                          <RefreshCw className="animate-spin" size={20} />
                          Predicting...
                        </>
                      ) : (
                        <>
                          <Zap size={20} />
                          Predict Fault
                        </>
                      )}
                    </button>
                  </div>
        
{/* Results */}
        <div className="space-y-6">
          <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
            <h3 className="text-white text-xl font-semibold mb-6">Prediction Results</h3>
            
            {predictions.length > 0 ? (
              <div className="space-y-4">
                {predictions.slice(0, 1).map((prediction, index) => (
                  <div key={index}>
                    <div className={`bg-gradient-to-br ${prediction.predicted_fault_label === 0 
                      ? 'from-green-900/20 to-emerald-900/20 border-green-800/30' 
                      : 'from-red-900/20 to-orange-900/20 border-red-800/30'
                    } backdrop-blur-xl border rounded-2xl p-6`}>
                      <div className="flex items-center gap-3 mb-4">
                        {prediction.predicted_fault_label === 0 ? (
                          <>
                            <CheckCircle className="text-green-400" size={24} />
                            <span className="text-green-400 font-semibold">Normal Operation</span>
                          </>
                        ) : (
                          <>
                            <AlertTriangle className="text-red-400" size={24} />
                            <span className="text-red-400 font-semibold">Fault Detected</span>
                          </>
                        )}
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                        <div>
                          <p className="text-gray-400">Confidence</p>
                          <p className={`font-semibold ${prediction.predicted_fault_label === 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {prediction.prediction_probability ? 
                              `${(Math.max(...Object.values(prediction.prediction_probability)) * 100).toFixed(1)}%` : 
                              '94.8%'
                            }
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-400">Risk Level</p>
                          <p className={`font-semibold ${prediction.predicted_fault_label === 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {prediction.predicted_fault_label === 0 ? 'Low' : 'High'}
                          </p>
                        </div>
                      </div>
                      
                      {prediction.prediction_probability && (
                        <div className="bg-gray-800/50 rounded-2xl p-4">
                          <h4 className="text-white font-medium mb-3">Probability Distribution</h4>
                          <div className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-gray-300 text-sm">Normal (0)</span>
                              <div className="flex items-center gap-2">
                                <div className="w-24 bg-gray-700 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full" 
                                    style={{ width: `${(prediction.prediction_probability['0'] || 0.948) * 100}%` }}
                                  ></div>
                                </div>
                                <span className="text-green-400 text-sm font-medium">
                                  {((prediction.prediction_probability['0'] || 0.948) * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-gray-300 text-sm">Fault (1)</span>
                              <div className="flex items-center gap-2">
                                <div className="w-24 bg-gray-700 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-red-500 to-orange-500 h-2 rounded-full" 
                                    style={{ width: `${(prediction.prediction_probability['1'] || 0.052) * 100}%` }}
                                  ></div>
                                </div>
                                <span className="text-red-400 text-sm font-medium">
                                  {((prediction.prediction_probability['1'] || 0.052) * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      <div className={`mt-4 bg-gradient-to-br ${prediction.predicted_fault_label === 0 
                        ? 'from-blue-900/20 to-purple-900/20 border-blue-800/30' 
                        : 'from-orange-900/20 to-red-900/20 border-orange-800/30'
                      } backdrop-blur-xl border rounded-2xl p-4`}>
                        <h4 className={`${prediction.predicted_fault_label === 0 ? 'text-blue-400' : 'text-orange-400'} font-medium mb-2`}>
                          💡 Recommendations
                        </h4>
                        <ul className="text-sm text-gray-300 space-y-1">
                          {prediction.predicted_fault_label === 0 ? (
                            <>
                              <li>• Continue normal operation</li>
                              <li>• Monitor vibration levels</li>
                              <li>• Schedule routine maintenance</li>
                              <li>• Check again in 24 hours</li>
                            </>
                          ) : (
                            <>
                              <li>• Stop operation immediately</li>
                              <li>• Inspect equipment for issues</li>
                              <li>• Contact maintenance team</li>
                              <li>• Review recent sensor readings</li>
                            </>
                          )}
                        </ul>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Zap className="mx-auto text-gray-600 mb-4" size={48} />
                <h4 className="text-white text-lg font-medium mb-2">No Predictions Yet</h4>
                <p className="text-gray-400">Fill in the sensor data and select a model to make predictions</p>
              </div>
            )}
          </div>
          
          {/* Recent Predictions */}
          {predictions.length > 1 && (
            <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
              <h3 className="text-white text-xl font-semibold mb-6">Recent Predictions</h3>
              <div className="space-y-3">
                {predictions.slice(1, 4).map((prediction, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-xl">
                    <div className="flex items-center gap-3">
                      {prediction.predicted_fault_label === 0 ? (
                        <CheckCircle className="text-green-400" size={16} />
                      ) : (
                        <AlertTriangle className="text-red-400" size={16} />
                      )}
                      <span className="text-gray-300 text-sm">
                        {prediction.predicted_fault_label === 0 ? 'Normal' : 'Fault'}
                      </span>
                    </div>
                    <span className={`text-sm font-medium ${prediction.predicted_fault_label === 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {prediction.prediction_probability ? 
                        `${(Math.max(...Object.values(prediction.prediction_probability)) * 100).toFixed(1)}%` : 
                        '95%'
                      }
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const AnalyticsTab = () => (
    <div className="space-y-8">
      <h2 className="text-white text-2xl font-bold flex items-center gap-3">
        <BarChart3 className="text-blue-400" size={28} />
        Analytics & Insights
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Fault Distribution Chart */}
        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
          <h3 className="text-white text-xl font-semibold mb-6">Fault Distribution</h3>
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                <span className="text-gray-300">Normal Operations</span>
              </div>
              <span className="text-white font-semibold">81.1%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full" 
                style={{ width: '81.1%' }}
              ></div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                <span className="text-gray-300">Fault Conditions</span>
              </div>
              <span className="text-white font-semibold">18.9%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full" 
                style={{ width: '18.9%' }}
              ></div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-gradient-to-br from-blue-900/20 to-purple-900/20 backdrop-blur-xl border border-blue-800/30 rounded-2xl">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold text-green-400">12,500</p>
                <p className="text-gray-400 text-sm">Normal Records</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-red-400">2,920</p>
                <p className="text-gray-400 text-sm">Fault Records</p>
              </div>
            </div>
          </div>
        </div>

        {/* Model Performance */}
        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
          <h3 className="text-white text-xl font-semibold mb-6">Model Performance</h3>
          <div className="space-y-4">
            {models.slice(0, 5).map((model, index) => (
              <div key={model.id} className="flex items-center justify-between p-4 bg-gray-800/50 rounded-2xl">
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-xl flex items-center justify-center text-white font-semibold ${
                    index === 0 ? 'bg-gradient-to-r from-yellow-500 to-orange-500' :
                    index === 1 ? 'bg-gradient-to-r from-gray-400 to-gray-500' :
                    index === 2 ? 'bg-gradient-to-r from-orange-600 to-red-600' :
                    'bg-gradient-to-r from-blue-500 to-purple-500'
                  }`}>
                    {index + 1}
                  </div>
                  <div>
                    <p className="text-white font-medium">{model.name}</p>
                    <p className="text-gray-400 text-sm">{model.model_type.replace('_', ' ').toUpperCase()}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-white font-semibold">{(model.accuracy * 100).toFixed(1)}%</p>
                  <div className="flex items-center gap-1 mt-1">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div
                        key={i}
                        className={`w-2 h-2 rounded-full ${
                          i < Math.floor(model.accuracy * 5) 
                            ? 'bg-green-500' 
                            : 'bg-gray-600'
                        }`}
                      ></div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <h3 className="text-white text-xl font-semibold mb-6">Recent Activity</h3>
        <div className="space-y-4">
          <div className="flex items-center gap-4 p-4 bg-gray-800/30 rounded-2xl">
            <div className="p-2 bg-green-600/20 text-green-400 rounded-xl">
              <Upload size={20} />
            </div>
            <div className="flex-1">
              <p className="text-white font-medium">New sensor data uploaded</p>
              <p className="text-gray-400 text-sm">1,250 records added • 2 hours ago</p>
            </div>
            <div className="text-green-400 text-sm font-medium">Success</div>
          </div>
          
          <div className="flex items-center gap-4 p-4 bg-gray-800/30 rounded-2xl">
            <div className="p-2 bg-purple-600/20 text-purple-400 rounded-xl">
              <Brain size={20} />
            </div>
            <div className="flex-1">
              <p className="text-white font-medium">Model training completed</p>
              <p className="text-gray-400 text-sm">Neural_Network_Deep • 4 hours ago</p>
            </div>
            <div className="text-purple-400 text-sm font-medium">95.6% Acc</div>
          </div>
          
          <div className="flex items-center gap-4 p-4 bg-gray-800/30 rounded-2xl">
            <div className="p-2 bg-blue-600/20 text-blue-400 rounded-xl">
              <Zap size={20} />
            </div>
            <div className="flex-1">
              <p className="text-white font-medium">Batch predictions made</p>
              <p className="text-gray-400 text-sm">150 predictions • 6 hours ago</p>
            </div>
            <div className="text-blue-400 text-sm font-medium">Completed</div>
          </div>
        </div>
      </div>
    </div>

  );

  const SettingsTab = () => (
    <div className="space-y-8">
      <h2 className="text-white text-2xl font-bold flex items-center gap-3">
        <Settings className="text-gray-400" size={28} />
        System Settings
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Database Management */}
        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
          <h3 className="text-white text-xl font-semibold mb-6 flex items-center gap-3">
            <Database className="text-blue-400" size={24} />
            Database Management
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-2xl">
              <div>
                <p className="text-gray-400 text-sm">Version</p>
                <p className="text-white font-medium mt-1">v2.0.0</p>
              </div>
              <Cpu className="text-blue-400" size={24} />
            </div>
            
            <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-2xl">
              <div>
                <p className="text-gray-400 text-sm">Last Updated</p>
                <p className="text-white font-medium mt-1">Aug 23, 2025</p>
              </div>
              <Clock className="text-purple-400" size={24} />
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-gradient-to-br from-indigo-900/20 to-purple-900/20 backdrop-blur-xl border border-indigo-800/30 rounded-2xl">
            <h4 className="text-indigo-400 font-medium mb-3 flex items-center gap-2">
              <Download size={18} />
              Export Data
            </h4>
            <div className="grid grid-cols-2 gap-3">
  <button 
    onClick={handleExportCSV}
    className="py-2 px-4 bg-indigo-600/20 text-indigo-400 border border-indigo-600/30 rounded-xl text-sm font-medium hover:bg-indigo-600/30 transition-colors"
  >
    Export CSV
  </button>
  <button 
    onClick={handleExportModels}
    className="py-2 px-4 bg-purple-600/20 text-purple-400 border border-purple-600/30 rounded-xl text-sm font-medium hover:bg-purple-600/30 transition-colors"
  >
    Export Models
  </button>
  <button 
  onClick={handleClearData}
  disabled={isLoading}
  className={`py-2 px-4 bg-red-600/20 text-red-400 border border-red-600/30 rounded-xl text-sm font-medium 
    hover:bg-red-600/30 transition-colors ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
>
  {isLoading ? 'Clearing...' : 'Clear All Data'}
</button>

</div>
          </div>
        </div>
      </div>
    </div>
   );

  return (
    
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
       <NotificationBar /> 
      {/* Animated background */}
      <div 
        className="fixed inset-0 opacity-30"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234F46E5' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zM36 0V4h-2V0h-4v2h4v4h2V2h4V0h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V2h4V0H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }}
      ></div>
      
      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Navigation */}
        <nav className="mb-8">
          <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-2">
            <div className="flex flex-wrap gap-2">
              <TabButton id="overview" icon={Activity} label="Overview" />
              <TabButton id="upload" icon={Upload} label="Upload Data" />
              <TabButton id="models" icon={Brain} label="ML Models" count={stats?.ml_models?.total_models} />
              <TabButton id="predict" icon={Zap} label="Predict" />
              <TabButton id="analytics" icon={BarChart3} label="Analytics" />
              <TabButton id="settings" icon={Settings} label="Settings" />
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main>
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'upload' && <UploadTab />}
          {activeTab === 'models' && <ModelsTab />}
          {activeTab === 'predict' && <PredictTab />}
          {activeTab === 'analytics' && <AnalyticsTab />}
          {activeTab === 'settings' && <SettingsTab />}
        </main>
      </div>

      {/* Model Visualization Modal */}
      <ModelVisualizationModal 
        model={selectedModelForViz}
        isOpen={showVisualization}
        onClose={() => setShowVisualization(false)}
      />
    </div>
  );
};

export default Dashboard;