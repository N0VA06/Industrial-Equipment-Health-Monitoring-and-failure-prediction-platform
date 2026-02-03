import React, { useState, useRef, useEffect } from 'react';
import './index.css';
import DatabaseViewer from './DatabaseViewer';
import mermaid from 'mermaid';
import LoginPage from './Login';
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
  X,
  Edit3,
  Save,
  Search,
  Filter,
  Server,
  MessageCircle
} from 'lucide-react';

// Machine Configuration
// Machine Configuration - Using proxy paths
// Machine Configuration - Direct URLs for development
// For production with Vercel rewrites, these paths work
// For local development, update to full URLs or use proper proxy
// Machine Configuration - MODIFY THESE URLs FOR YOUR DOCKER INSTANCES

// Machine Configuration - MODIFY THESE URLs FOR YOUR DOCKER INSTANCES
const MACHINES = {
  machine1: {
    id: 'machine1',
    name: 'Machine 1',
    url: 'https://52.90.145.182/backend1',
    color: 'blue',
    description: 'Production Line A'
  },
  machine2: {
    id: 'machine2',
    name: 'Machine 2',
    url: 'https://52.90.145.182/backend2',
    color: 'green',
    description: 'Production Line B'
  },
  machine3: {
    id: 'machine3',
    name: 'Machine 3',
    url: 'https://52.90.145.182/backend3',
    color: 'purple',
    description: 'Production Line C'
  }
};

// Authentication URL - Only backend on 8001 has auth configured
const AUTH_BASE_URL = 'https://52.90.145.182/backend1';

// RAG Chat Configuration
const RAG_API_URL = 'https://ehffsqpp4z.us-east-1.awsapprunner.com/';
const RAG_AUTH_TOKEN = 'Bearer 9fb2eed12d2b4def6242c0e16708fe60fa1d99fd7fa2c6323d1913cd4303b446';

// Dynamic API utility functions
const createApi = (baseUrl) => ({
  uploadData: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${baseUrl}/upload-csv/`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  getStats: async () => {
    const response = await fetch(`${baseUrl}/database-stats/`);
    if (!response.ok) {
      throw new Error(`Failed to fetch stats: ${response.statusText}`);
    }
    return response.json();
  },

  getModels: async () => {
    const response = await fetch(`${baseUrl}/ml/models/trained/`);
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.statusText}`);
    }
    return response.json();
  },

  trainModel: async (modelData) => {
    const response = await fetch(`${baseUrl}/ml/train/`, {
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

  predict: async (modelId, sensorData) => {
    const response = await fetch(`${baseUrl}/ml/predict/`, {
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
    const response = await fetch(`${baseUrl}/ml/models/${modelId}/`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error(`Delete failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  clearData: async () => {
    const response = await fetch(`${baseUrl}/clear-everything/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`Clear data failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  getSensorData: async () => {
    const response = await fetch(`${baseUrl}/sensor-data/`);
    if (!response.ok) throw new Error('Failed to fetch sensor data');
    return response.json();
  },

  updateSensorRecord: async (id, data) => {
    const response = await fetch(`${baseUrl}/sensor-data/${id}/`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to update record');
    return response.json();
  },

  deleteSensorRecord: async (id) => {
    const response = await fetch(`${baseUrl}/sensor-data/clear/`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete record');
    return response.json();
  },

  exportCSV: async () => {
    const response = await fetch(`${baseUrl}/export-csv/`);
    return response.blob();
  },

  exportModels: async () => {
    const response = await fetch(`${baseUrl}/export-models/`);
    return response.blob();
  },

  getModelDetails: async (modelId) => {
    const response = await fetch(`${baseUrl}/ml/models/${modelId}/`);
    if (!response.ok) throw new Error('Failed to fetch model details');
    return response.json();
  }
});

// Initialize mermaid
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

// RAG Chat Widget Component
const RAGChatWidget = ({ isOpen, onClose }) => {
  const [documentUrl, setDocumentUrl] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your RAG assistant. Please paste a document URL above, then ask me any questions about it. I specialize in legal, insurance, technical, and scientific documents.",
      sender: 'assistant'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [documentStatus, setDocumentStatus] = useState({ text: '', type: '' });
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen) {
      checkHealth();
    }
  }, [isOpen]);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${RAG_API_URL}/health`);
      if (response.ok) {
        console.log('✓ RAG Backend connected');
      }
    } catch (error) {
      console.warn('⚠ RAG Backend not reachable');
      setMessages(prev => [...prev, {
        id: Date.now(),
        text: '⚠ Warning: Cannot connect to RAG backend. Please ensure the server is running at ' + RAG_API_URL,
        sender: 'assistant'
      }]);
    }
  };

  const handleDocumentUrlChange = (e) => {
    const url = e.target.value.trim();
    setDocumentUrl(url);

    if (!url) {
      setDocumentStatus({ text: '', type: '' });
      return;
    }

    try {
      new URL(url);
      setDocumentStatus({ text: '✓ Document URL ready', type: 'success' });
    } catch {
      setDocumentStatus({ text: '✗ Invalid URL format', type: 'error' });
    }
  };

  const isValidUrl = () => {
    if (!documentUrl) return false;
    try {
      new URL(documentUrl);
      return true;
    } catch {
      return false;
    }
  };

  const handleSendMessage = async () => {
    const question = inputValue.trim();

    if (!question || !isValidUrl() || isProcessing) {
      return;
    }

    const userMessage = {
      id: Date.now(),
      text: question,
      sender: 'user'
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    const typingMessage = {
      id: Date.now() + 1,
      text: '',
      sender: 'assistant',
      isTyping: true
    };
    setMessages(prev => [...prev, typingMessage]);

    setIsProcessing(true);

    try {
      const response = await fetch(`${RAG_API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': RAG_AUTH_TOKEN
        },
        body: JSON.stringify({
          documents: documentUrl,
          questions: [question]
        })
      });

      setMessages(prev => prev.filter(msg => !msg.isTyping));

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        id: Date.now() + 2,
        text: data.answers && data.answers.length > 0 
          ? data.answers[0] 
          : "Sorry, I couldn't generate a response.",
        sender: 'assistant'
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      setMessages(prev => prev.filter(msg => !msg.isTyping));

      const errorMessage = {
        id: Date.now() + 3,
        text: `Error: ${error.message}. Please check your document URL and try again.`,
        sender: 'assistant'
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('RAG API Error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-24 right-8 w-96 h-[600px] bg-gray-900 border border-gray-700 rounded-3xl shadow-2xl flex flex-col overflow-hidden z-50 animate-slideIn">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-5 flex justify-between items-center flex-shrink-0">
        <div>
          <h3 className="text-lg font-semibold">RAG Assistant</h3>
          <div className="text-xs opacity-90">Multi-Agent AI System</div>
        </div>
        <button 
          className="text-white hover:opacity-70 transition-opacity p-1" 
          onClick={onClose}
        >
          <X size={24} />
        </button>
      </div>

      {/* Document section */}
      <div className="p-4 bg-gray-800/50 border-b border-gray-700 flex-shrink-0">
        <label className="block text-sm font-semibold text-gray-300 mb-2">Document URL</label>
        <input
          type="url"
          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white text-sm focus:outline-none focus:border-indigo-500 transition-colors"
          placeholder="https://example.com/document.pdf"
          value={documentUrl}
          onChange={handleDocumentUrlChange}
        />
        {documentStatus.text && (
          <div className={`mt-2 text-xs ${documentStatus.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
            {documentStatus.text}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex gap-3 ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`w-9 h-9 rounded-full flex items-center justify-center text-xs font-semibold text-white flex-shrink-0 ${
              message.sender === 'user' 
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600' 
                : 'bg-gradient-to-r from-pink-500 to-red-500'
            }`}>
              {message.sender === 'user' ? 'You' : 'AI'}
            </div>
            <div className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm ${
              message.sender === 'user'
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                : 'bg-gray-800 text-gray-200'
            }`}>
              {message.isTyping ? (
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
              ) : (
                message.text
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-700 bg-gray-900 flex-shrink-0">
        <div className="flex gap-3 items-center">
          <input
            type="text"
            className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-full text-white text-sm focus:outline-none focus:border-indigo-500 transition-colors disabled:opacity-50"
            placeholder="Ask a question..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={!isValidUrl() || isProcessing}
          />
          <button
            className="w-11 h-11 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full flex items-center justify-center text-white hover:scale-110 transition-transform disabled:opacity-50 disabled:hover:scale-100 flex-shrink-0"
            onClick={handleSendMessage}
            disabled={!isValidUrl() || !inputValue.trim() || isProcessing}
          >
            <svg viewBox="0 0 24 24" className="w-5 h-5 fill-current">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

// Rest of the original components remain the same
const MachineSelector = ({ selectedMachine, onSelectMachine, machineStatus }) => {
  const getColorClasses = (color, isSelected) => {
    const colors = {
      blue: isSelected 
        ? 'bg-blue-600 border-blue-500 shadow-blue-500/50' 
        : 'bg-blue-900/20 border-blue-800/30 hover:border-blue-600/50',
      green: isSelected 
        ? 'bg-green-600 border-green-500 shadow-green-500/50' 
        : 'bg-green-900/20 border-green-800/30 hover:border-green-600/50',
      purple: isSelected 
        ? 'bg-purple-600 border-purple-500 shadow-purple-500/50' 
        : 'bg-purple-900/20 border-purple-800/30 hover:border-purple-600/50'
    };
    return colors[color] || colors.blue;
  };

  return (
    <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 mb-6">
      <h3 className="text-white text-xl font-semibold mb-4 flex items-center gap-3">
        <Server className="text-indigo-400" size={24} />
        Select Machine
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.values(MACHINES).map((machine) => {
          const isSelected = selectedMachine === machine.id;
          const status = machineStatus[machine.id];
          
          return (
            <button
              key={machine.id}
              onClick={() => onSelectMachine(machine.id)}
              className={`p-4 border-2 rounded-2xl transition-all duration-300 ${
                getColorClasses(machine.color, isSelected)
              } ${isSelected ? 'scale-105 shadow-2xl' : 'hover:scale-102'}`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <Server size={24} className="text-white" />
                  <div className="text-left">
                    <h4 className="text-white font-semibold">{machine.name}</h4>
                    <p className="text-gray-300 text-xs">{machine.description}</p>
                  </div>
                </div>
                <div className={`w-3 h-3 rounded-full ${
                  status?.online ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                }`}></div>
              </div>
              
              {status?.online && status?.stats && (
                <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-white/10">
                  <div className="text-left">
                    <p className="text-xs text-gray-300">Records</p>
                    <p className="text-white font-semibold text-sm">
                      {status.stats.sensor_data?.total_records || 0}
                    </p>
                  </div>
                  <div className="text-left">
                    <p className="text-xs text-gray-300">Models</p>
                    <p className="text-white font-semibold text-sm">
                      {status.stats.ml_models?.total_models || 0}
                    </p>
                  </div>
                </div>
              )}
              
              {!status?.online && (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <p className="text-red-400 text-xs">Offline</p>
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

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
            <X size={24} />
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
  const [selectedMachine, setSelectedMachine] = useState('machine1');
  const [machineStatus, setMachineStatus] = useState({});
  const [api, setApi] = useState(createApi(MACHINES.machine1.url));
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
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false);

  const [predictionData, setPredictionData] = useState({
    vibration: '',
    temperature: '',
    pressure: '',
    rms_vibration: '',
    mean_temp: ''
  });

  const [trainingForm, setTrainingForm] = useState({
    model_type: 'random_forest',
    name: '',
    test_size: 0.2,
    description: ''
  });

  const handleLogin = (userData) => {
    setUser(userData);
    setIsAuthenticated(true);
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('user');
  };

  const checkMachineStatus = async (machineId) => {
    try {
      const machineApi = createApi(MACHINES[machineId].url);
      const stats = await machineApi.getStats();
      return { online: true, stats };
    } catch (error) {
      return { online: false, stats: null };
    }
  };

  const checkAllMachines = async () => {
    const statuses = {};
    for (const machineId of Object.keys(MACHINES)) {
      statuses[machineId] = await checkMachineStatus(machineId);
    }
    setMachineStatus(statuses);
  };

  const handleSelectMachine = (machineId) => {
    setSelectedMachine(machineId);
    setApi(createApi(MACHINES[machineId].url));
    setSuccess(`Switched to ${MACHINES[machineId].name}`);
    loadData(createApi(MACHINES[machineId].url));
  };

  const loadData = async (apiInstance = api) => {
    try {
      setIsLoading(true);
      
      const [statsData, modelsData] = await Promise.all([
        apiInstance.getStats(),
        apiInstance.getModels()
      ]);
      
      if (statsData) setStats(statsData);
      if (modelsData) setModels(modelsData);
      
    } catch (err) {
      console.error('Error loading data:', err);
      setError(`Failed to connect to ${MACHINES[selectedMachine].name}. Please ensure the backend is running.`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      setIsAuthenticated(true);
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      checkAllMachines();
      loadData();
      
      const interval = setInterval(checkAllMachines, 30000);
      return () => clearInterval(interval);
    }
  }, [isAuthenticated, selectedMachine]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setIsLoading(true);
      setUploadProgress(0);
      setError(null);

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
        setSuccess(`Data uploaded successfully to ${MACHINES[selectedMachine].name}!`);
        loadData();
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
      
      setSuccess(`Model "${trainingForm.name}" trained successfully on ${MACHINES[selectedMachine].name}!`);
      loadData();

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
      setSuccess(`Prediction completed on ${MACHINES[selectedMachine].name}!`);

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

  const handleVisualize = (model) => {
    setSelectedModelForViz(model);
    setShowVisualization(true);
  };

  useEffect(() => {
    if (error || success) {
      const timer = setTimeout(() => {
        setError(null);
        setSuccess(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, success]);

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

  const OverviewTab = () => (
    <div className="space-y-8">
      <div className="relative overflow-hidden bg-gradient-to-br from-indigo-900/50 via-purple-900/30 to-pink-900/20 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <div className="relative z-10">
          <div className="flex items-center gap-4 mb-6">
            <div className="p-4 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-3xl shadow-2xl shadow-indigo-500/25">
              <Shield className="text-white" size={32} />
            </div>
            <div>
              <h1 className="text-white text-3xl font-bold mb-2">Multi-Machine Fault Detection System</h1>
              <p className="text-gray-400">Connected to {MACHINES[selectedMachine].name} - {MACHINES[selectedMachine].description}</p>
            </div>
          </div>
        </div>
      </div>

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard 
            icon={Database}
            title="Sensor Records"
            value={stats.sensor_data?.total_records?.toLocaleString() || '0'}
            change={15}
            color="blue"
          />
          <MetricCard 
            icon={Brain}
            title="Active Models"
            value={stats.ml_models?.total_models || 0}
            change={25}
            color="purple"
          />
          <MetricCard 
            icon={Target}
            title="Total Predictions"
            value={stats.predictions?.total_predictions?.toLocaleString() || '0'}
            change={8}
            color="green"
          />
          <MetricCard 
            icon={Upload}
            title="Upload Success Rate"
            value={stats.uploads?.total_uploads ? 
              `${((stats.uploads.successful_uploads / stats.uploads.total_uploads) * 100).toFixed(1)}%` : 
              '0%'}
            change={3}
            color="orange"
          />
        </div>
      )}

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
          Upload Sensor Data to {MACHINES[selectedMachine].name}
        </h2>
        
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
              <span>Uploading to {MACHINES[selectedMachine].name}...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="bg-gray-700 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-indigo-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      <DatabaseViewer api={api} />
    </div>
  );

  const ModelsTab = () => (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-white text-2xl font-bold flex items-center gap-3">
          <Brain className="text-purple-400" size={28} />
          ML Models on {MACHINES[selectedMachine].name}
        </h2>
        <button 
          onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-purple-500/25 transition-all duration-300"
        >
          <Plus size={20} />
          Train New Model
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <ModelCard key={model.id} model={model} />
        ))}
      </div>

      <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <h3 className="text-white text-xl font-semibold mb-6">Train New Model</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-gray-300 font-medium mb-2">Model Type</label>
            <select 
              value={trainingForm.model_type}
              onChange={(e) => setTrainingForm(prev => ({ ...prev, model_type: e.target.value }))} 
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
            >
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
          disabled={isTraining || !trainingForm.name.trim()}
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
        Fault Prediction on {MACHINES[selectedMachine].name}
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

        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
          <h3 className="text-white text-xl font-semibold mb-6">Prediction Results</h3>
          
          {predictions.length > 0 ? (
            <div className="space-y-4">
              {predictions.slice(0, 1).map((prediction, index) => (
                <div key={index} className={`bg-gradient-to-br ${prediction.predicted_fault_label === 0 
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
                  <div className="grid grid-cols-2 gap-4 text-sm">
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
      </div>
    </div>
  );

  const AnalyticsTab = () => (
    <div className="space-y-8">
      <h2 className="text-white text-2xl font-bold flex items-center gap-3">
        <BarChart3 className="text-blue-400" size={28} />
        Analytics for {MACHINES[selectedMachine].name}
      </h2>

      {stats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
            <h3 className="text-white text-xl font-semibold mb-6">Fault Distribution</h3>
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                  <span className="text-gray-300">Normal Operations</span>
                </div>
                <span className="text-white font-semibold">
                  {stats.sensor_data?.fault_distribution ? 
                    ((stats.sensor_data.fault_distribution.fault_0 / stats.sensor_data.total_records) * 100).toFixed(1) : 
                    '81.1'}%
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full" 
                  style={{ 
                    width: stats.sensor_data?.fault_distribution ? 
                      `${(stats.sensor_data.fault_distribution.fault_0 / stats.sensor_data.total_records) * 100}%` : 
                      '81.1%'
                  }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                  <span className="text-gray-300">Fault Conditions</span>
                </div>
                <span className="text-white font-semibold">
                  {stats.sensor_data?.fault_distribution ? 
                    ((stats.sensor_data.fault_distribution.fault_1 / stats.sensor_data.total_records) * 100).toFixed(1) : 
                    '18.9'}%
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full" 
                  style={{ 
                    width: stats.sensor_data?.fault_distribution ? 
                      `${(stats.sensor_data.fault_distribution.fault_1 / stats.sensor_data.total_records) * 100}%` : 
                      '18.9%'
                  }}
                ></div>
              </div>
            </div>
          </div>

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
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const SettingsTab = () => (
    <div className="space-y-8">
      <h2 className="text-white text-2xl font-bold flex items-center gap-3">
        <Settings className="text-gray-400" size={28} />
        System Settings - {MACHINES[selectedMachine].name}
      </h2>

      <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <h3 className="text-white text-xl font-semibold mb-6 flex items-center gap-3">
          <Database className="text-blue-400" size={24} />
          Database Management
        </h3>
        
        <div className="grid grid-cols-2 gap-4">
          <button 
            onClick={async () => {
              try {
                const blob = await api.exportCSV();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `sensor_data_${selectedMachine}.csv`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setSuccess('CSV exported successfully!');
              } catch (err) {
                setError('Failed to export CSV');
              }
            }}
            className="py-2 px-4 bg-indigo-600/20 text-indigo-400 border border-indigo-600/30 rounded-xl text-sm font-medium hover:bg-indigo-600/30 transition-colors"
          >
            Export CSV
          </button>
          <button 
            onClick={async () => {
              try {
                const blob = await api.exportModels();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `ml_models_${selectedMachine}.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setSuccess('Models exported successfully!');
              } catch (err) {
                setError('Failed to export models');
              }
            }}
            className="py-2 px-4 bg-purple-600/20 text-purple-400 border border-purple-600/30 rounded-xl text-sm font-medium hover:bg-purple-600/30 transition-colors"
          >
            Export Models
          </button>
        </div>
      </div>
    </div>
  );

  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
      <NotificationBar />
      
      <div className="bg-gray-900/50 backdrop-blur-xl border-b border-gray-800 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center gap-4">
            <h1 className="text-white text-xl font-bold">Multi-Machine Fault Detection System</h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-gray-300">
              <div className={`w-3 h-3 rounded-full ${
                user?.role === 'admin' ? 'bg-red-500' :
                user?.role === 'engineer' ? 'bg-blue-500' :
                'bg-green-500'
              }`}></div>
              <span className="capitalize">{user?.role || 'Guest'}</span>
              <span>•</span>
              <span>{user?.username}</span>
            </div>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600/20 text-red-400 border border-red-600/30 rounded-xl text-sm font-medium hover:bg-red-600/30 transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </div>

      <div 
        className="fixed inset-0 opacity-30 pointer-events-none"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234F46E5' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zM36 0V4h-2V0h-4v2h4v4h2V2h4V0h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V2h4V0H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }}
      ></div>
      
      <div className="relative z-10 container mx-auto px-6 py-8">
        <MachineSelector 
          selectedMachine={selectedMachine}
          onSelectMachine={handleSelectMachine}
          machineStatus={machineStatus}
        />

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

        <main>
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'upload' && <UploadTab />}
          {activeTab === 'models' && <ModelsTab />}
          {activeTab === 'predict' && <PredictTab />}
          {activeTab === 'analytics' && <AnalyticsTab />}
          {activeTab === 'settings' && <SettingsTab />}
        </main>
      </div>

      <ModelVisualizationModal 
        model={selectedModelForViz}
        isOpen={showVisualization}
        onClose={() => setShowVisualization(false)}
      />

      {/* RAG Chat Button */}
      <button
        onClick={() => setIsChatOpen(true)}
        className="fixed bottom-8 right-8 w-16 h-16 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full shadow-2xl hover:scale-110 transition-transform duration-300 flex items-center justify-center z-40"
        title="Open RAG Assistant"
      >
        <MessageCircle className="text-white" size={28} />
      </button>

      {/* RAG Chat Widget */}
      <RAGChatWidget isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} />
    </div>
  );
};

export default Dashboard;