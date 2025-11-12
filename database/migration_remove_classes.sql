-- Migration script to remove unwanted classes from PPE Detection system
-- This script updates the existing enums to only include the required 4 classes:
-- Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest

-- First, remove any existing data with the old classes to avoid conflicts
DELETE FROM violations WHERE violation_type IN ('NO-Mask');
DELETE FROM object_detections WHERE class_name IN ('Mask', 'NO-Mask', 'Person', 'Safety Cone', 'machinery', 'vehicle');

-- Update the violation_type enum
ALTER TYPE violation_type RENAME TO violation_type_old;
CREATE TYPE violation_type AS ENUM (
    'NO-Hardhat',
    'NO-Safety Vest'
);

-- Update any remaining data to use the new enum
ALTER TABLE violations ALTER COLUMN violation_type TYPE violation_type USING violation_type::text::violation_type;
ALTER TABLE violation_summary ALTER COLUMN violation_type TYPE violation_type USING violation_type::text::violation_type;

-- Drop the old enum type
DROP TYPE violation_type_old;

-- Update the detection_class enum
ALTER TYPE detection_class RENAME TO detection_class_old;
CREATE TYPE detection_class AS ENUM (
    'Hardhat',
    'NO-Hardhat',
    'Safety Vest',
    'NO-Safety Vest'
);

-- Update any remaining data to use the new enum
ALTER TABLE object_detections ALTER COLUMN class_name TYPE detection_class USING class_name::text::detection_class;

-- Drop the old enum type
DROP TYPE detection_class_old;