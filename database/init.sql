-- Initialize PPE Detection Database Schema

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types for violation categories
CREATE TYPE violation_type AS ENUM (
    'NO-Hardhat',
    'NO-Safety Vest'
);

CREATE TYPE detection_class AS ENUM (
    'Hardhat',
    'NO-Hardhat',
    'Safety Vest',
    'NO-Safety Vest'
);

-- Table to store detection sessions
CREATE TABLE detection_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name VARCHAR(255),
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('webcam', 'video_file', 'image_file', 'rtsp_stream')),
    source_path TEXT,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    total_frames INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store person detections
CREATE TABLE person_detections (
    detection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES detection_sessions(session_id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL,
    frame_number INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    confidence DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store all object detections
CREATE TABLE object_detections (
    detection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES detection_sessions(session_id) ON DELETE CASCADE,
    person_detection_id UUID REFERENCES person_detections(detection_id) ON DELETE SET NULL,
    frame_number INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    class_name detection_class NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store PPE violations
CREATE TABLE ppe_violations (
    violation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES detection_sessions(session_id) ON DELETE CASCADE,
    person_detection_id UUID REFERENCES person_detections(detection_id) ON DELETE CASCADE,
    violation_type violation_type NOT NULL,
    frame_number INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    confidence DECIMAL(5,4) NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store violation summary statistics
CREATE TABLE violation_summary (
    summary_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES detection_sessions(session_id) ON DELETE CASCADE,
    violation_type violation_type NOT NULL,
    total_violations INTEGER DEFAULT 0,
    unique_persons INTEGER DEFAULT 0,
    first_occurrence TIMESTAMP WITH TIME ZONE,
    last_occurrence TIMESTAMP WITH TIME ZONE,
    avg_confidence DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better query performance
CREATE INDEX idx_person_detections_session_id ON person_detections(session_id);
CREATE INDEX idx_person_detections_timestamp ON person_detections(timestamp);
CREATE INDEX idx_person_detections_person_id ON person_detections(person_id);

CREATE INDEX idx_object_detections_session_id ON object_detections(session_id);
CREATE INDEX idx_object_detections_timestamp ON object_detections(timestamp);
CREATE INDEX idx_object_detections_class_name ON object_detections(class_name);

CREATE INDEX idx_ppe_violations_session_id ON ppe_violations(session_id);
CREATE INDEX idx_ppe_violations_timestamp ON ppe_violations(timestamp);
CREATE INDEX idx_ppe_violations_violation_type ON ppe_violations(violation_type);
CREATE INDEX idx_ppe_violations_acknowledged ON ppe_violations(acknowledged);

CREATE INDEX idx_violation_summary_session_id ON violation_summary(session_id);
CREATE INDEX idx_violation_summary_violation_type ON violation_summary(violation_type);

-- Function to update violation summary
CREATE OR REPLACE FUNCTION update_violation_summary()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert or update violation summary
    INSERT INTO violation_summary (
        session_id,
        violation_type,
        total_violations,
        unique_persons,
        first_occurrence,
        last_occurrence,
        avg_confidence
    )
    SELECT 
        NEW.session_id,
        NEW.violation_type,
        COUNT(*),
        COUNT(DISTINCT pd.person_id),
        MIN(v.timestamp),
        MAX(v.timestamp),
        AVG(v.confidence)
    FROM ppe_violations v
    JOIN person_detections pd ON v.person_detection_id = pd.detection_id
    WHERE v.session_id = NEW.session_id 
    AND v.violation_type = NEW.violation_type
    GROUP BY v.session_id, v.violation_type
    ON CONFLICT (session_id, violation_type) 
    DO UPDATE SET
        total_violations = EXCLUDED.total_violations,
        unique_persons = EXCLUDED.unique_persons,
        first_occurrence = EXCLUDED.first_occurrence,
        last_occurrence = EXCLUDED.last_occurrence,
        avg_confidence = EXCLUDED.avg_confidence,
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add unique constraint for violation summary
ALTER TABLE violation_summary ADD CONSTRAINT unique_session_violation_type 
UNIQUE (session_id, violation_type);

-- Trigger to automatically update violation summary
CREATE TRIGGER trigger_update_violation_summary
    AFTER INSERT ON ppe_violations
    FOR EACH ROW
    EXECUTE FUNCTION update_violation_summary();

-- Function to update session frame count
CREATE OR REPLACE FUNCTION update_session_frame_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE detection_sessions 
    SET total_frames = (
        SELECT MAX(frame_number) 
        FROM person_detections 
        WHERE session_id = NEW.session_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE session_id = NEW.session_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update session frame count
CREATE TRIGGER trigger_update_session_frame_count
    AFTER INSERT ON person_detections
    FOR EACH ROW
    EXECUTE FUNCTION update_session_frame_count();

-- Insert some initial data for testing
INSERT INTO detection_sessions (session_name, source_type, source_path) 
VALUES ('Test Session', 'webcam', '/dev/video0');

-- Create views for easy data access
CREATE VIEW violation_details AS
SELECT 
    v.violation_id,
    v.violation_type,
    v.timestamp,
    v.confidence,
    v.severity,
    v.acknowledged,
    pd.person_id,
    pd.frame_number,
    ds.session_name,
    ds.source_type
FROM ppe_violations v
JOIN person_detections pd ON v.person_detection_id = pd.detection_id
JOIN detection_sessions ds ON v.session_id = ds.session_id;

CREATE VIEW session_statistics AS
SELECT 
    ds.session_id,
    ds.session_name,
    ds.source_type,
    ds.start_time,
    ds.end_time,
    ds.total_frames,
    COUNT(DISTINCT pd.person_id) as total_persons,
    COUNT(v.violation_id) as total_violations,
    COUNT(CASE WHEN v.acknowledged = false THEN 1 END) as unacknowledged_violations
FROM detection_sessions ds
LEFT JOIN person_detections pd ON ds.session_id = pd.session_id
LEFT JOIN ppe_violations v ON ds.session_id = v.session_id
GROUP BY ds.session_id, ds.session_name, ds.source_type, ds.start_time, ds.end_time, ds.total_frames;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ppe_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ppe_user;
GRANT USAGE ON SCHEMA public TO ppe_user;