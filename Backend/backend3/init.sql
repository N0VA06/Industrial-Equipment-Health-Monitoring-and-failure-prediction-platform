-- Initialize the database
CREATE DATABASE IF NOT EXISTS maintenance_data CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE maintenance_data;

-- Verify database creation
SELECT 'Creating maintenance_data database...' as status;

-- Create app user with necessary privileges
CREATE USER IF NOT EXISTS 'app_user'@'%' IDENTIFIED BY 'app_password';
GRANT ALL PRIVILEGES ON maintenance_data.* TO 'app_user'@'%';

-- Grant additional privileges for the app user
GRANT CREATE, ALTER, DROP, INSERT, UPDATE, DELETE, SELECT, REFERENCES, RELOAD on *.* TO 'app_user'@'%' WITH GRANT OPTION;

-- Also ensure root can connect from any host (for debugging)
CREATE USER IF NOT EXISTS 'root'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;

FLUSH PRIVILEGES;

-- Set timezone
SET time_zone = '+00:00';

-- Optimize MySQL settings for better performance and larger BLOB support
SET GLOBAL innodb_buffer_pool_size = 268435456; -- 256MB
SET GLOBAL max_connections = 200;
SET GLOBAL wait_timeout = 28800;
SET GLOBAL interactive_timeout = 28800;
SET GLOBAL max_allowed_packet = 67108864; -- 64MB for large model storage

-- Verify the database and user setup
SELECT 'Database maintenance_data created successfully' as status;
SELECT User, Host FROM mysql.user WHERE User IN ('app_user', 'root');
SHOW DATABASES;