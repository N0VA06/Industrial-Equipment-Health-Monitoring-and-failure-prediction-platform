import React, { useState } from 'react';
import { Shield, User, Lock, Eye, EyeOff, AlertTriangle } from 'lucide-react';

const AUTH_BASE_URL = 'https://52.90.145.182/backend1';
const LoginPage = ({ onLogin }) => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    role: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!credentials.username || !credentials.password || !credentials.role) {
      setError('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch(`${AUTH_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({
          username: credentials.username,
          password: credentials.password,
          role: credentials.role
        }),
      });

      if (!response.ok) {
        throw new Error('Invalid credentials');
      }

      const data = await response.json();
      
      // Call parent login handler
      onLogin(data.user);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const roles = [
    { value: 'admin', label: 'Administrator', color: 'from-red-600 to-orange-600' },
    { value: 'engineer', label: 'Engineer', color: 'from-blue-600 to-cyan-600' },
    { value: 'technician', label: 'Technician', color: 'from-green-600 to-emerald-600' },
    { value: 'operator', label: 'Operator', color: 'from-purple-600 to-pink-600' },
  ];

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 flex items-center justify-center p-6">
      {/* Animated background */}
      <div 
        className="fixed inset-0 opacity-30"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234F46E5' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zM36 0V4h-2V0h-4v2h4v4h2V2h4V0h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V2h4V0H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }}
      />

      <div className="relative z-10 w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-3xl shadow-2xl shadow-indigo-500/25 mb-6">
            <Shield className="text-white" size={40} />
          </div>
          <h1 className="text-white text-3xl font-bold mb-2">Industrial Fault Detection</h1>
          <p className="text-gray-400">Secure access to monitoring systems</p>
        </div>

        {/* Login Form */}
        <div className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-3xl p-8">
          <div className="space-y-6">
            {/* Username */}
            <div>
              <label className="block text-gray-300 font-medium mb-2">Username</label>
              <div className="relative">
                <User className="absolute left-4 top-4 text-gray-400" size={20} />
                <input
                  type="text"
                  value={credentials.username}
                  onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl pl-12 pr-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
                  placeholder="Enter your username"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-gray-300 font-medium mb-2">Password</label>
              <div className="relative">
                <Lock className="absolute left-4 top-4 text-gray-400" size={20} />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={credentials.password}
                  onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                  onKeyPress={handleKeyPress}
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl pl-12 pr-12 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-4 text-gray-400 hover:text-white"
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            {/* Role Selection */}
            <div>
              <label className="block text-gray-300 font-medium mb-2">Role</label>
              <div className="grid grid-cols-2 gap-3">
                {roles.map((role) => (
                  <button
                    key={role.value}
                    type="button"
                    onClick={() => setCredentials(prev => ({ ...prev, role: role.value }))}
                    className={`p-3 rounded-xl border-2 transition-all duration-300 ${
                      credentials.role === role.value
                        ? `bg-gradient-to-r ${role.color} border-transparent text-white shadow-lg`
                        : 'border-gray-700 text-gray-400 hover:border-gray-600 hover:text-white'
                    }`}
                  >
                    <div className="text-sm font-medium">{role.label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-900/50 border border-red-800 rounded-2xl p-4 flex items-center gap-3">
                <AlertTriangle className="text-red-400 flex-shrink-0" size={20} />
                <span className="text-red-200">{error}</span>
              </div>
            )}

            {/* Login Button */}
            <button
              onClick={handleSubmit}
              disabled={isLoading || !credentials.username || !credentials.password || !credentials.role}
              className="w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Signing In...
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </div>

          {/* Demo Credentials */}
          <div className="mt-6 p-4 bg-gradient-to-br from-blue-900/20 to-purple-900/20 backdrop-blur-xl border border-blue-800/30 rounded-2xl">
            <h4 className="text-blue-400 font-medium mb-2">Demo Credentials</h4>
            <div className="text-sm text-gray-300 space-y-1">
              <div>Admin: admin / admin123</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;