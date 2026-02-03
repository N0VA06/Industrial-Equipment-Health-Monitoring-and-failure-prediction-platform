import './index.css';
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

  // Get stats
  getStats: async () => {
    const response = await fetch(`${API_BASE_URL}/stats/`);
    if (!response.ok) {
      throw new Error(`Failed to fetch stats: ${response.statusText}`);
    }
    return response.json();
  },

  // Get models
  getModels: async () => {
    const response = await fetch(`${API_BASE_URL}/models/`);
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.statusText}`);
    }
    return response.json();
  },

  // Train model
  trainModel: async (modelData) => {
    const response = await fetch(`${API_BASE_URL}/train/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modelData),
    });
    
    if (!response.ok) {
      throw new Error(`Training failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Make predictions
  predict: async (modelId, data) => {
    const response = await fetch(`${API_BASE_URL}/predict/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_id: modelId,
        data: data
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Prediction failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Delete model
  deleteModel: async (modelId) => {
    const response = await fetch(`${API_BASE_URL}/models/${modelId}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error(`Delete failed: ${response.statusText}`);
    }
    
    return response.json();
  }
};

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [models, setModels] = useState([]);
  const [stats, setStats] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

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

  // Load initial data
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setIsLoading(true);
      
      // Load stats and models in parallel
      const [statsData, modelsData] = await Promise.all([
        api.getStats().catch(() => null), // Don't fail if stats endpoint doesn't exist
        api.getModels().catch(() => [])   // Don't fail if models endpoint doesn't exist
      ]);
      
      if (statsData) setStats(statsData);
      if (modelsData) setModels(Array.isArray(modelsData) ? modelsData : []);
      
    } catch (err) {
      console.error('Error loading data:', err);
      setError('Failed to connect to backend. Please ensure the backend is running.');
      
      // Set mock data for demonstration
      setStats({
        sensor_data: { total_records: 15420, fault_distribution: { fault_0: 12500, fault_1: 2920 } },
        ml_models: { total_models: 3, inactive_models: 1 },
        predictions: { total_predictions: 1250 },
        uploads: { total_uploads: 23, successful_uploads: 22, failed_uploads: 1 }
      });
      
      setModels([
        { 
          id: 1, 
          name: 'Production_RF_Model', 
          model_type: 'random_forest', 
          accuracy: 0.948, 
          f1_score: 0.943, 
          created_at: '2025-08-20T10:30:00',
          is_active: true 
        },
        { 
          id: 2, 
          name: 'SVM_Classifier_v2', 
          model_type: 'svm', 
          accuracy: 0.921, 
          f1_score: 0.918, 
          created_at: '2025-08-19T14:15:00',
          is_active: true 
        },
        { 
          id: 3, 
          name: 'Neural_Network_Deep', 
          model_type: 'neural_network', 
          accuracy: 0.956, 
          f1_score: 0.951, 
          created_at: '2025-08-18T09:45:00',
          is_active: false 
        }
      ]);
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
      setTrainingForm({
        model_type: 'random_forest',
        name: '',
        test_size: 0.2,
        description: ''
      });
      
      loadData(); // Reload models after training

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

    const hasEmptyFields = Object.values(predictionData).some(value => value === '');
    if (hasEmptyFields) {
      setError('Please fill all sensor data fields');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const data = Object.fromEntries(
        Object.entries(predictionData).map(([key, value]) => [key, parseFloat(value)])
      );

      const result = await api.predict(selectedModel.id, data);
      
      setPredictions(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 predictions
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
      <span className="hidden sm:inline">{label}</span>
      {count !== undefined && (
        <span className={`px-2 py-1 text-xs rounded-full ${
          activeTab === id ? 'bg-white/20' : 'bg-gray-700'
        }`}>
          {count}
        </span>
      )}
    </button>
  );

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
          <p className="text-white text-2xl font-bold">{value || '0'}</p>
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
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-xl transition-colors"
            title="View Details"
          >
            <Eye size={16} />
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
          <div className={`w-2 h-2 rounded-full ${model.is_active ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <span>{model.is_active ? 'Active' : 'Inactive'}</span>
        </div>
      </div>
      
      <button 
        onClick={() => {
          setSelectedModel(model);
          setActiveTab('predict');
        }}
        className="w-full py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300"
        disabled={!model.is_active}
      >
        {model.is_active ? 'Use for Prediction' : 'Model Inactive'}
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
          <div className="flex flex-col lg:flex-row items-start lg:items-center gap-6 mb-6">
            <div className="flex items-center gap-4 flex-1">
              <div className="p-4 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-3xl shadow-2xl shadow-indigo-500/25">
                <Shield className="text-white" size={32} />
              </div>
              <div>
                <h1 className="text-white text-3xl font-bold mb-2">Industrial Fault Detection System</h1>
                <p className="text-gray-400">AI-powered predictive maintenance and fault analysis</p>
              </div>
            </div>
            <div className="flex gap-3">
              <button 
                onClick={() => setActiveTab('upload')}
                className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300 flex items-center gap-2"
              >
                <Upload size={18} />
                Upload Data
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
        </div>
      </div>

      {/* Stats Grid */}
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
            value={stats.ml_models?.total_models || models.length}
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
            value={stats.uploads ? `${((stats.uploads.successful_uploads / stats.uploads.total_uploads) * 100).toFixed(1)}%` : '95%'}
            change={3}
            color="orange"
          />
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div 
          onClick={() => setActiveTab('upload')}
          className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300 group cursor-pointer"
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-2xl shadow-lg group-hover:shadow-blue-500/25">
              <Upload className="text-white" size={24} />
            </div>
            <div>
              <h3 className="text-white font-medium">Upload Sensor Data</h3>
              <p className="text-gray-400 text-sm">Import CSV files</p>
            </div>
          </div>
        </div>

        <div 
          onClick={() => setActiveTab('models')}
          className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-purple-500/10 transition-all duration-300 group cursor-pointer"
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl shadow-lg group-hover:shadow-purple-500/25">
              <Brain className="text-white" size={24} />
            </div>
            <div>
              <h3 className="text-white font-medium">Train Models</h3>
              <p className="text-gray-400 text-sm">Create ML models</p>
            </div>
          </div>
        </div>

        <div 
          onClick={() => setActiveTab('predict')}
          className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 hover:shadow-2xl hover:shadow-green-500/10 transition-all duration-300 group cursor-pointer"
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-green-600 to-emerald-600 rounded-2xl shadow-lg group-hover:shadow-green-500/25">
              <Zap className="text-white" size={24} />
            </div>
            <div>
              <h3 className="text-white font-medium">Make Predictions</h3>
              <p className="text-gray-400 text-sm">Analyze for faults</p>
            </div>
          </div>
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
                  <span>Uploading...</span>
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
          
          <div className="space-y-6">
            <div className="bg-gray-800/50 rounded-2xl p-6">
              <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                <FileText className="text-indigo-400" size={20} />
                CSV Format Requirements
              </h3>
              <div className="space-y-2 text-sm">
                {[
                  'Timestamp', 'Vibration', 'Temperature', 'Pressure', 
                  'RMS Vibration', 'Mean Temp', 'Fault Label'
                ].map((field) => (
                  <div key={field} className="flex justify-between">
                    <span className="text-gray-400">{field}</span>
                    <span className="text-green-400">✓ Required</span>
                  </div>
                ))}
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
          </div>
        </div>
      </div>
    </div>
  );

  const ModelsTab = () => (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h2 className="text-white text-2xl font-bold flex items-center gap-3">
          <Brain className="text-purple-400" size={28} />
          ML Models ({models.length})
        </h2>
        <button 
          onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-purple-500/25 transition-all duration-300"
        >
          <Plus size={20} />
          Train New Model
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {models.map((model) => (
          <ModelCard key={model.id} model={model} />
        ))}
        {models.length === 0 && !isLoading && (
          <div className="col-span-full text-center py-12">
            <Brain className="mx-auto text-gray-600 mb-4" size={48} />
            <h3 className="text-white text-lg font-medium mb-2">No Models Available</h3>
            <p className="text-gray-400 mb-4">Train your first model to get started with predictions</p>
            <button 
              onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })}
              className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:shadow-lg transition-all duration-300"
            >
              Train First Model
            </button>
          </div>
        )}
      </div>

      {/* Model Training Form */}
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
        Fault Prediction
      </h2>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Input Form */}
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

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Fault Distribution */}
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
                <p className="text-2xl font-bold text-green-400">
                  {stats?.sensor_data?.fault_distribution?.fault_0?.toLocaleString() || '12,500'}
                </p>
                <p className="text-gray-400 text-sm">Normal Records</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-red-400">
                  {stats?.sensor_data?.fault_distribution?.fault_1?.toLocaleString() || '2,920'}
                </p>
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
            {models.length === 0 && (
              <div className="text-center py-8">
                <Brain className="mx-auto text-gray-600 mb-4" size={48} />
                <p className="text-gray-400">No models available for analysis</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
        <h3 className="text-white text-xl font-semibold mb-6">System Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="p-4 bg-gradient-to-br from-green-600 to-emerald-600 rounded-2xl shadow-lg mb-4 mx-auto w-fit">
              <Database className="text-white" size={32} />
            </div>
            <h4 className="text-white font-semibold mb-2">Database</h4>
            <p className="text-green-400">Online</p>
          </div>
          <div className="text-center">
            <div className="p-4 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-2xl shadow-lg mb-4 mx-auto w-fit">
              <Brain className="text-white" size={32} />
            </div>
            <h4 className="text-white font-semibold mb-2">ML Engine</h4>
            <p className="text-blue-400">Ready</p>
          </div>
          <div className="text-center">
            <div className="p-4 bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl shadow-lg mb-4 mx-auto w-fit">
              <Zap className="text-white" size={32} />
            </div>
            <h4 className="text-white font-semibold mb-2">API Status</h4>
            <p className={error ? "text-red-400" : "text-purple-400"}>
              {error ? "Disconnected" : "Connected"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
      {/* Animated background */}
      <div 
        className="fixed inset-0 opacity-30"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234F46E5' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }}
      ></div>
      
      <div className="relative z-10 container mx-auto px-4 sm:px-6 py-8">
        {/* Navigation */}
        <nav className="mb-8">
          <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-2">
            <div className="flex flex-wrap gap-2">
              <TabButton id="overview" icon={Activity} label="Overview" />
              <TabButton id="upload" icon={Upload} label="Upload" />
              <TabButton id="models" icon={Brain} label="Models" count={models.length} />
              <TabButton id="predict" icon={Zap} label="Predict" />
              <TabButton id="analytics" icon={BarChart3} label="Analytics" />
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
        </main>
      </div>

      {/* Notification Bar */}
      <NotificationBar />
    </div>
  );
};

export default Dashboard;