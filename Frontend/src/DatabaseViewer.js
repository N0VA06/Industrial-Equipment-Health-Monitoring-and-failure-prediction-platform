import React, { useState, useEffect } from 'react';
import { Database, Edit3, Trash2, Save, X, Plus, Search, Filter } from 'lucide-react';

const DatabaseViewer = ({ api }) => {
  const [sensorData, setSensorData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [editingRecord, setEditingRecord] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const recordsPerPage = 20;

  useEffect(() => {
    loadSensorData();
  }, [currentPage]);

  const loadSensorData = async () => {
    try {
      setLoading(true);
      const offset = (currentPage - 1) * recordsPerPage;
      const response = await api.getSensorData(recordsPerPage, offset);
      setSensorData(response.data || []);
      setTotalRecords(response.total || 0);
    } catch (error) {
      console.error('Failed to load sensor data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (record) => {
    setEditingRecord({ ...record });
  };

  const handleSave = async () => {
    try {
      await api.updateSensorRecord(editingRecord.id, editingRecord);
      await loadSensorData();
      setEditingRecord(null);
    } catch (error) {
      console.error('Failed to update record:', error);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this record?')) return;
    
    try {
      await api.deleteSensorRecord(id);
      await loadSensorData();
    } catch (error) {
      console.error('Failed to delete record:', error);
    }
  };

  const filteredData = sensorData.filter(record =>
    Object.values(record).some(value =>
      value?.toString().toLowerCase().includes(searchTerm.toLowerCase())
    )
  );

  const totalPages = Math.ceil(totalRecords / recordsPerPage);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-white text-xl font-semibold flex items-center gap-3">
          <Database className="text-blue-400" size={24} />
          Database Records
        </h3>
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-3 text-gray-400" size={18} />
            <input
              type="text"
              placeholder="Search records..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-xl pl-10 pr-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          <button
            onClick={loadSensorData}
            className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">ID</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">Timestamp</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">Vibration</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">Temperature</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">Pressure</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">RMS Vibration</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">Mean Temp</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">Fault Label</th>
                <th className="px-6 py-4 text-left text-gray-300 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="9" className="px-6 py-8 text-center text-gray-400">
                    Loading records...
                  </td>
                </tr>
              ) : filteredData.length === 0 ? (
                <tr>
                  <td colSpan="9" className="px-6 py-8 text-center text-gray-400">
                    No records found
                  </td>
                </tr>
              ) : (
                filteredData.map((record) => (
                  <tr key={record.id} className="border-t border-gray-800 hover:bg-gray-800/30">
                    {editingRecord?.id === record.id ? (
                      // Edit mode
                      <>
                        <td className="px-6 py-4 text-white">{record.id}</td>
                        <td className="px-6 py-4">
                          <input
                            type="datetime-local"
                            value={editingRecord.timestamp?.slice(0, 16)}
                            onChange={(e) => setEditingRecord({
                              ...editingRecord,
                              timestamp: e.target.value
                            })}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <input
                            type="number"
                            step="0.01"
                            value={editingRecord.vibration}
                            onChange={(e) => setEditingRecord({
                              ...editingRecord,
                              vibration: parseFloat(e.target.value)
                            })}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm w-20"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <input
                            type="number"
                            step="0.1"
                            value={editingRecord.temperature}
                            onChange={(e) => setEditingRecord({
                              ...editingRecord,
                              temperature: parseFloat(e.target.value)
                            })}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm w-20"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <input
                            type="number"
                            step="0.1"
                            value={editingRecord.pressure}
                            onChange={(e) => setEditingRecord({
                              ...editingRecord,
                              pressure: parseFloat(e.target.value)
                            })}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm w-20"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <input
                            type="number"
                            step="0.01"
                            value={editingRecord.rms_vibration}
                            onChange={(e) => setEditingRecord({
                              ...editingRecord,
                              rms_vibration: parseFloat(e.target.value)
                            })}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm w-20"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <input
                            type="number"
                            step="0.1"
                            value={editingRecord.mean_temp}
                            onChange={(e) => setEditingRecord({
                              ...editingRecord,
                              mean_temp: parseFloat(e.target.value)
                            })}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm w-20"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <select
                            value={editingRecord.fault_label}
                            onChange={(e) => setEditingRecord({
                              ...editingRecord,
                              fault_label: parseInt(e.target.value)
                            })}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm"
                          >
                            <option value={0}>0 (Normal)</option>
                            <option value={1}>1 (Fault)</option>
                          </select>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex gap-2">
                            <button
                              onClick={handleSave}
                              className="p-1 text-green-400 hover:bg-green-400/20 rounded"
                            >
                              <Save size={16} />
                            </button>
                            <button
                              onClick={() => setEditingRecord(null)}
                              className="p-1 text-gray-400 hover:bg-gray-400/20 rounded"
                            >
                              <X size={16} />
                            </button>
                          </div>
                        </td>
                      </>
                    ) : (
                      // View mode
                      <>
                        <td className="px-6 py-4 text-white">{record.id}</td>
                        <td className="px-6 py-4 text-gray-300 text-sm">
                          {new Date(record.timestamp).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-gray-300">{record.vibration}</td>
                        <td className="px-6 py-4 text-gray-300">{record.temperature}</td>
                        <td className="px-6 py-4 text-gray-300">{record.pressure}</td>
                        <td className="px-6 py-4 text-gray-300">{record.rms_vibration}</td>
                        <td className="px-6 py-4 text-gray-300">{record.mean_temp}</td>
                        <td className="px-6 py-4">
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            record.fault_label === 0 
                              ? 'bg-green-900/30 text-green-400' 
                              : 'bg-red-900/30 text-red-400'
                          }`}>
                            {record.fault_label === 0 ? 'Normal' : 'Fault'}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleEdit(record)}
                              className="p-1 text-blue-400 hover:bg-blue-400/20 rounded"
                            >
                              <Edit3 size={16} />
                            </button>
                            <button
                              onClick={() => handleDelete(record.id)}
                              className="p-1 text-red-400 hover:bg-red-400/20 rounded"
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        </td>
                      </>
                    )}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-800">
          <div className="text-gray-400 text-sm">
            Showing {((currentPage - 1) * recordsPerPage) + 1} to {Math.min(currentPage * recordsPerPage, totalRecords)} of {totalRecords} records
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-gray-800 text-gray-400 rounded disabled:opacity-50"
            >
              Previous
            </button>
            <span className="px-3 py-1 bg-blue-600 text-white rounded">
              {currentPage}
            </span>
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-gray-800 text-gray-400 rounded disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatabaseViewer;