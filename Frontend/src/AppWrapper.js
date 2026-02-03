import React, { useState, useEffect } from 'react';
import Dashboard from './App'; // Your existing dashboard component
import LoginPage from './LoginPage';
import MachineSelector from './MachineSelector';
import DatabaseViewer from './DatabaseViewer';

const AppWrapper = () => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check for existing authentication on app start
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('auth_token');
      const userRole = localStorage.getItem('user_role');
      const username = localStorage.getItem('username');
      const userId = localStorage.getItem('user_id');

      if (token && userRole && username && userId) {
        try {
          // Verify token with backend
          const response = await fetch('http://localhost:8000/auth/verify/', {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          });

          if (response.ok) {
            const userData = await response.json();
            setUser(userData.user);
            setIsAuthenticated(true);
          } else {
            // Token invalid, clear storage
            localStorage.removeItem('auth_token');
            localStorage.removeItem('user_role');
            localStorage.removeItem('username');
            localStorage.removeItem('user_id');
          }
        } catch (error) {
          console.error('Auth verification failed:', error);
          // Clear invalid auth data
          localStorage.removeItem('auth_token');
          localStorage.removeItem('user_role');
          localStorage.removeItem('username');
          localStorage.removeItem('user_id');
        }
      }
      setLoading(false);
    };

    checkAuth();
  }, []);

  const handleLogin = (userData) => {
    setUser(userData);
    setIsAuthenticated(true);
    
    // Store auth data
    localStorage.setItem('auth_token', userData.access_token || 'demo_token');
    localStorage.setItem('user_role', userData.role);
    localStorage.setItem('username', userData.username);
    localStorage.setItem('user_id', userData.id || 'demo_id');
  };

  const handleLogout = () => {
    setUser(null);
    setIsAuthenticated(false);
    setSelectedMachine(null);
    
    // Clear auth data
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('username');
    localStorage.removeItem('user_id');
  };

  const handleMachineSelect = (machine) => {
    setSelectedMachine(machine);
    // Store selected machine for persistence
    if (machine) {
      localStorage.setItem('selected_machine', JSON.stringify(machine));
    } else {
      localStorage.removeItem('selected_machine');
    }
  };

  // Enhanced API object with authentication and machine context
  const enhancedApi = {
    // Upload data with machine context
    uploadData: async (file, machineId = selectedMachine?.id) => {
      const formData = new FormData();
      formData.append('file', file);
      if (machineId) formData.append('machine_id', machineId);
      
      const response = await fetch('http://localhost:8000/upload-csv/', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      return response.json();
    },

    // Get database stats for specific machine
    getStats: async (machineId = selectedMachine?.id) => {
      const url = machineId 
        ? `http://localhost:8000/database-stats/?machine_id=${machineId}`
        : 'http://localhost:8000/database-stats/';
      
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch stats: ${response.statusText}`);
      }
      return response.json();
    },

    // Get models for specific machine
    getModels: async (machineId = selectedMachine?.id) => {
      const url = machineId 
        ? `http://localhost:8000/ml/models/trained/?machine_id=${machineId}`
        : 'http://localhost:8000/ml/models/trained/';
      
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }
      return response.json();
    },

    // Train model with machine context
    trainModel: async (modelData, machineId = selectedMachine?.id) => {
      const response = await fetch('http://localhost:8000/ml/train/', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: modelData.model_type,
          model_name: modelData.name,
          test_size: modelData.test_size,
          description: modelData.description,
          machine_id: machineId,
          hyperparameters: {}
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Training failed: ${response.statusText}`);
      }
      return response.json();
    },

    // Make predictions with machine context
    predict: async (modelId, sensorData, machineId = selectedMachine?.id) => {
      const response = await fetch('http://localhost:8000/ml/predict/', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelId,
          machine_id: machineId,
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
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });
      
      if (!response.ok) {
        throw new Error(`Delete failed: ${response.statusText}`);
      }
      
      return response.json();
    },

    clearData: async (machineId = selectedMachine?.id) => {
      const response = await fetch('http://localhost:8000/clear-data/', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ machine_id: machineId })
      });
      
      if (!response.ok) {
        throw new Error(`Clear data failed: ${response.statusText}`);
      }
      
      return response.json();
    },

    // DBMS functions
    getSensorData: async (limit = 100, offset = 0, machineId = selectedMachine?.id) => {
      let url = `http://localhost:8000/sensor-data/?limit=${limit}&offset=${offset}`;
      if (machineId) url += `&machine_id=${machineId}`;
      
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to fetch sensor data');
      return response.json();
    },

    updateSensorRecord: async (id, data) => {
      const response = await fetch(`http://localhost:8000/sensor-data/${id}/`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to update record');
      return response.json();
    },

    deleteSensorRecord: async (id) => {
      const response = await fetch(`http://localhost:8000/sensor-data/${id}/`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to delete record');
      return response.json();
    },
  };

  // Enhanced Dashboard component with role-based features and machine context
  const EnhancedDashboard = () => {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
        {/* Top Navigation Bar */}
        <div className="bg-gray-900/50 backdrop-blur-xl border-b border-gray-800 p-4">
          <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-white text-xl font-bold">Fault Detection System</h1>
              {selectedMachine && (
                <div className="flex items-center gap-2 px-3 py-1 bg-indigo-600/20 text-indigo-400 rounded-full text-sm">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  {selectedMachine.name}
                </div>
              )}
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-gray-300">
                <div className={`w-3 h-3 rounded-full ${
                  user?.role === 'admin' ? 'bg-red-500' :
                  user?.role === 'engineer' ? 'bg-blue-500' :
                  user?.role === 'technician' ? 'bg-green-500' : 'bg-purple-500'
                }`}></div>
                <span className="capitalize">{user?.role}</span>
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

        {/* Machine Selection */}
        <div className="container mx-auto px-6 py-4">
          <MachineSelector
            userRole={user?.role}
            onMachineSelect={handleMachineSelect}
            selectedMachine={selectedMachine}
            api={enhancedApi}
          />
        </div>

        {/* Main Dashboard */}
        {selectedMachine ? (
          <div className="container mx-auto px-6">
            <Dashboard 
              api={enhancedApi}
              userRole={user?.role}
              selectedMachine={selectedMachine}
              DatabaseViewer={DatabaseViewer}
            />
          </div>
        ) : (
          <div className="container mx-auto px-6 py-16 text-center">
            <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-12">
              <div className="text-gray-400 mb-4">
                <svg className="mx-auto w-16 h-16" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
              </div>
              <h3 className="text-white text-xl font-semibold mb-2">Select a Machine</h3>
              <p className="text-gray-400">Choose a machine from the selector above to start monitoring and analysis</p>
            </div>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading application...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return <EnhancedDashboard />;
};

export default AppWrapper;