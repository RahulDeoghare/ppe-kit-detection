-- Simple PPE Violations Database - Single Table Approach
-- This replaces the complex multi-table schema with ONE table for ALL violations

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables if they exist (clean slate)
DROP TABLE IF EXISTS ppe_violations CASCADE;
DROP TABLE IF EXISTS violation_summary CASCADE;
DROP TABLE IF EXISTS object_detections CASCADE;
DROP TABLE IF EXISTS person_detections CASCADE;
DROP TABLE IF EXISTS detection_sessions CASCADE;

-- Drop existing types and recreate
DROP TYPE IF EXISTS violation_type CASCADE;
DROP TYPE IF EXISTS detection_class CASCADE;

-- Create enum for violation types
CREATE TYPE violation_type AS ENUM (
    'NO-Hardhat',
    'NO-Mask', 
    'NO-Safety Vest'
);

-- Single table to store ALL PPE violations with screenshots
CREATE TABLE ppe_violations (
    violation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Basic violation info
    violation_type violation_type NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    
    -- Person and location info
    person_id INTEGER,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    
    -- Source info
    source_type VARCHAR(50) NOT NULL DEFAULT 'webcam' CHECK (source_type IN ('webcam', 'video_file', 'image_file', 'rtsp_stream')),
    session_name VARCHAR(255),
    frame_number INTEGER,
    
    -- Screenshot path
    screenshot_path TEXT NOT NULL,
    
    -- Timing
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Acknowledgment
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    notes TEXT,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_violations_timestamp ON ppe_violations(timestamp DESC);
CREATE INDEX idx_violations_type ON ppe_violations(violation_type);
CREATE INDEX idx_violations_acknowledged ON ppe_violations(acknowledged);
CREATE INDEX idx_violations_session ON ppe_violations(session_name);
CREATE INDEX idx_violations_severity ON ppe_violations(severity);

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE ppe_violations TO ppe_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ppe_user;
GRANT USAGE ON SCHEMA public TO ppe_user;

-- Create screenshots directory if it doesn't exist (for reference)
-- mkdir -p static/screenshots (to be created by application)
