-- Initialize the database
CREATE DATABASE IF NOT EXISTS sensor_data;
USE sensor_data;

-- Create app user with necessary privileges
CREATE USER IF NOT EXISTS 'app_user'@'%' IDENTIFIED BY 'app_password';
GRANT ALL PRIVILEGES ON sensor_data.* TO 'app_user'@'%';
FLUSH PRIVILEGES;

-- Set timezone
SET time_zone = '+00:00';

-- Optimize MySQL settings for better performance and larger BLOB support
SET GLOBAL innodb_buffer_pool_size = 268435456;  -- 256MB
SET GLOBAL max_connections = 200;
SET GLOBAL wait_timeout = 28800;
SET GLOBAL interactive_timeout = 28800;
SET GLOBAL max_allowed_packet = 67108864;  -- 64MB for large model storage