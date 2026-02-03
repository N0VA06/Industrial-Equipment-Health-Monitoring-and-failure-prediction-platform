import React, { useState, useEffect } from 'react';
import { Settings, Plus, Edit3, Trash2, CheckCircle, AlertTriangle, Activity } from 'lucide-react';

const MachineSelector = ({ userRole, onMachineSelect, selectedMachine, api }) => {
  const [machines, setMachines] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showAddMachine, setShowAddMachine] = useState(false);
  const [newMachine, setNewMachine] = useState({
    name: '',
    location: '',
    type: '',
    description: ''
  });

  useEffect(() => {
    loadMachines();
  }, []);

  const loadMachines = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/machines/', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) throw new Error('Failed to load machines');
      const data = await response.json();
      setMachines(data);
    } catch (error) {
      console.error('Error loading machines:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddMachine = async () => {
    if (!newMachine.name.trim()) return;

    try {
      const response = await fetch('http://localhost:8000/machines/', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newMachine),
      });

      if (!response.ok) throw new Error('Failed to add machine');
      
      await loadMachines();
      setNewMachine({ name: '', location: '', type: '', description: '' });
      setShowAddMachine(false);
    } catch (error) {
      console.error('Error adding machine:', error);
    }
  };

  const handleDeleteMachine = async (machineId) => {
    if (!window.confirm('Are you sure you want to delete this machine?')) return;

    try {
      const response = await fetch(`http://localhost:8000/machines/${machineId}/`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });

      if (!response.ok) throw new Error('Failed to delete machine');
      await loadMachines();
      
      if (selectedMachine?.id === machineId) {
        onMachineSelect(null);
      }
    } catch (error) {
      console.error('Error deleting machine:', error);
    }
  };

  const canManageMachines = ['admin', 'engineer'].includes(userRole);

  const machineTypes = [
    'Pump', 'Motor', 'Compressor', 'Turbine', 'Generator', 'Conveyor', 'Other'
  ];

  return (
    <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-white text-xl font-semibold flex items-center gap-3">
          <Settings className="text-blue-400" size={24} />
          Machine Selection
        </h3>
        {canManageMachines && (
          <button
            onClick={() => setShowAddMachine(true)}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl font-medium hover:shadow-lg transition-all duration-300"
          >
            <Plus size={18} />
            Add Machine
          </button>
        )}
      </div>

      {loading ? (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading machines...</p>
        </div>
      ) : machines.length === 0 ? (
        <div className="text-center py-8">
          <Settings className="mx-auto text-gray-600 mb-4" size={48} />
          <h4 className="text-white text-lg font-medium mb-2">No Machines Available</h4>
          <p className="text-gray-400">Add machines to start monitoring</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {machines.map((machine) => (
            <div
              key={machine.id}
              className={`p-4 rounded-2xl border-2 transition-all duration-300 cursor-pointer ${
                selectedMachine?.id === machine.id
                  ? 'border-indigo-500 bg-indigo-600/20'
                  : 'border-gray-700 hover:border-gray-600 bg-gray-800/50'
              }`}
              onClick={() => onMachineSelect(machine)}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${
                    machine.status === 'active' ? 'bg-green-500 animate-pulse' : 
                    machine.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                  <div>
                    <h4 className="text-white font-semibold">{machine.name}</h4>
                    <p className="text-gray-400 text-sm">{machine.type} • {machine.location}</p>
                  </div>
                </div>
                
                {canManageMachines && (
                  <div className="flex gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle edit machine
                      }}
                      className="p-1 text-gray-400 hover:text-blue-400 rounded"
                    >
                      <Edit3 size={14} />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteMachine(machine.id);
                      }}
                      className="p-1 text-gray-400 hover:text-red-400 rounded"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Last Update:</span>
                  <span className="text-gray-300">
                    {machine.last_updated ? new Date(machine.last_updated).toLocaleDateString() : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Status:</span>
                  <span className={`font-medium ${
                    machine.status === 'active' ? 'text-green-400' :
                    machine.status === 'warning' ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {machine.status?.charAt(0).toUpperCase() + machine.status?.slice(1) || 'Unknown'}
                  </span>
                </div>
              </div>

              {machine.description && (
                <p className="text-gray-400 text-sm mt-3 line-clamp-2">{machine.description}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Add Machine Modal */}
      {showAddMachine && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-3xl max-w-md w-full p-6">
            <h3 className="text-white text-xl font-semibold mb-6">Add New Machine</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-gray-300 font-medium mb-2">Machine Name</label>
                <input
                  type="text"
                  value={newMachine.name}
                  onChange={(e) => setNewMachine(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  placeholder="Enter machine name"
                />
              </div>

              <div>
                <label className="block text-gray-300 font-medium mb-2">Location</label>
                <input
                  type="text"
                  value={newMachine.location}
                  onChange={(e) => setNewMachine(prev => ({ ...prev, location: e.target.value }))}
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  placeholder="Enter location"
                />
              </div>

              <div>
                <label className="block text-gray-300 font-medium mb-2">Machine Type</label>
                <select
                  value={newMachine.type}
                  onChange={(e) => setNewMachine(prev => ({ ...prev, type: e.target.value }))}
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="">Select type...</option>
                  {machineTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-gray-300 font-medium mb-2">Description</label>
                <textarea
                  value={newMachine.description}
                  onChange={(e) => setNewMachine(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500 resize-none"
                  rows={3}
                  placeholder="Enter description (optional)"
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAddMachine(false)}
                className="flex-1 py-3 bg-gray-800 text-gray-300 rounded-xl font-medium hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleAddMachine}
                disabled={!newMachine.name.trim()}
                className="flex-1 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl font-medium hover:shadow-lg transition-all duration-300 disabled:opacity-50"
              >
                Add Machine
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MachineSelector;