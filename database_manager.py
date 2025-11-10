"""
Database connection and management module for PPE Detection System
Single table approach - stores only violations with screenshots
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import time
from dotenv import load_dotenv
import cv2
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and operations for PPE violations only"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "vms_staging",
                 user: str = "postgres", 
                 password: str = "fostgres",
                 min_connections: int = 1,
                 max_connections: int = 10):
        
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user
        }
        
        # Only add password if it's provided (for trust authentication)
        if password:
            self.connection_params['password'] = password
        
        # Initialize connection pool
        try:
            self.pool = SimpleConnectionPool(
                min_connections, 
                max_connections, 
                **self.connection_params
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if string is a valid UUID format"""
        import re
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_pattern.match(uuid_string))
    
    def save_violation_with_image(self,
                                session_name: str,
                                violation_type: str,
                                person_id: int,
                                frame_number: int,
                                bbox: tuple,
                                confidence: float,
                                image_frame,
                                whole_frame=None,
                                severity: str = "high",
                                source_type: str = "webcam",
                                source_path: str = None,
                                camera_id: str = None,
                                device_id: str = None,
                                office_id: str = None,
                                status: str = "pending"):
        """Save violation with both bounding box screenshot and whole frame to database"""
        
        violation_id = str(uuid.uuid4())
        x1, y1, x2, y2 = bbox
        timestamp = datetime.now(timezone.utc)
        
        # Convert image to binary data for database storage
        violation_image_binary = None
        if image_frame is not None:
            try:
                # Encode image as JPEG binary data
                success, buffer = cv2.imencode('.jpg', image_frame)
                if success:
                    violation_image_binary = buffer.tobytes()
                    logger.info(f"Encoded violation image to binary: {len(violation_image_binary)} bytes")
            except Exception as e:
                logger.error(f"Failed to encode violation image to binary: {e}")
                violation_image_binary = None
        
        # Ensure severity is lowercase to match database constraint
        severity = severity.lower()
        
        # Set camera_id if not provided (use UUID format)
        if camera_id is None:
            camera_id = "b48ff955-d517-44ba-939a-97d7c76c17b6"
        elif camera_id and not self._is_valid_uuid(camera_id):
            # If provided but not a valid UUID, use default
            logger.warning(f"Invalid camera_id format: {camera_id}, using default")
            camera_id = "b48ff955-d517-44ba-939a-97d7c76c17b6"
        
        # Set device_id if not provided (use UUID format)
        if device_id is None:
            device_id = "01a0ea94-6d15-4740-b97e-95cb7c65e112"
        elif device_id and not self._is_valid_uuid(device_id):
            # If provided but not a valid UUID, use default
            logger.warning(f"Invalid device_id format: {device_id}, using default")
            device_id = "01a0ea94-6d15-4740-b97e-95cb7c65e112"
        
        # Set office_id if not provided (use UUID format)
        if office_id is None:
            office_id = "48ee8982-56bd-4d49-bd24-4b148d73d8f3"
        elif office_id and not self._is_valid_uuid(office_id):
            # If provided but not a valid UUID, use default
            logger.warning(f"Invalid office_id format: {office_id}, using default")
            office_id = "48ee8982-56bd-4d49-bd24-4b148d73d8f3"
        
        # Save violation screenshot (bounding box area)
        screenshot_path = ""  # Default empty path
        
        if image_frame is not None:
            try:
                # Create violations directory if it doesn't exist
                violations_dir = "violations"
                os.makedirs(violations_dir, exist_ok=True)
                
                # Save bounding box image file
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_filename = f"violation_bbox_{timestamp_str}.jpg"
                screenshot_path = os.path.join(violations_dir, image_filename)
                
                cv2.imwrite(screenshot_path, image_frame)
                
                logger.info(f"Saved violation bounding box screenshot: {screenshot_path}")
                
            except Exception as e:
                logger.error(f"Failed to save violation bounding box screenshot: {e}")
                screenshot_path = ""  # Set empty path if saving fails
        
        # Save whole frame screenshot
        whole_frame_path = ""  # Default empty path
        
        if whole_frame is not None:
            try:
                # Create violations directory if it doesn't exist
                violations_dir = "violations"
                os.makedirs(violations_dir, exist_ok=True)
                
                # Save whole frame image file
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                frame_filename = f"violation_frame_{timestamp_str}.jpg"
                whole_frame_path = os.path.join(violations_dir, frame_filename)
                
                cv2.imwrite(whole_frame_path, whole_frame)
                
                logger.info(f"Saved violation whole frame screenshot: {whole_frame_path}")
                
            except Exception as e:
                logger.error(f"Failed to save violation whole frame screenshot: {e}")
                whole_frame_path = ""  # Set empty path if saving fails
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO ppe_violations 
                        (violation_id, session_name, source_type, camera_id, device_id, office_id, violation_type, 
                         person_id, frame_number, status, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                         confidence, severity, screenshot_path, whole_frame_path, violation_image_data, timestamp, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (violation_id, session_name, source_type, camera_id, device_id, office_id, violation_type,
                         person_id, frame_number, status, x1, y1, x2, y2, 
                         confidence, severity, screenshot_path, whole_frame_path, violation_image_binary, timestamp, timestamp))
                    conn.commit()
                    
                    logger.info(f"Saved violation: {violation_type} for person {person_id} in session {session_name} with binary image data")
                    return violation_id
                    
        except Exception as e:
            logger.error(f"Failed to save violation: {e}")
            raise
    
    def get_violations_paginated(self, page: int = 1, per_page: int = 50) -> tuple:
        """Get violations with pagination - returns (violations, total_count)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Get total count
                    cursor.execute("SELECT COUNT(*) as total FROM ppe_violations")
                    count_result = cursor.fetchone()
                    total_count = count_result['total'] if count_result else 0
                    
                    # Get paginated results
                    offset = (page - 1) * per_page
                    cursor.execute("""
                        SELECT 
                            violation_id,
                            session_name,
                            source_type,
                            camera_id,
                            device_id,
                            office_id,
                            violation_type,
                            person_id,
                            frame_number,
                            status,
                            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                            confidence,
                            severity,
                            screenshot_path,
                            whole_frame_path,
                            acknowledged,
                            acknowledged_by,
                            acknowledged_at,
                            notes,
                            timestamp,
                            created_at,
                            updated_at
                        FROM ppe_violations 
                        ORDER BY timestamp DESC
                        LIMIT %s OFFSET %s
                    """, (per_page, offset))
                    
                    violations = cursor.fetchall()
                    return violations, total_count
        except Exception as e:
            logger.error(f"Failed to get paginated violations: {e}")
            # Return empty results instead of raising
            return [], 0

    def get_all_violations(self, limit: int = 1000) -> List[Dict]:
        """Get ALL violations from the database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            violation_id,
                            session_name,
                            source_type,
                            camera_id,
                            device_id,
                            office_id,
                            violation_type,
                            person_id,
                            frame_number,
                            status,
                            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                            confidence,
                            severity,
                            screenshot_path,
                            whole_frame_path,
                            acknowledged,
                            acknowledged_by,
                            acknowledged_at,
                            notes,
                            timestamp,
                            created_at,
                            updated_at
                        FROM ppe_violations 
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get violations: {e}")
            raise
    
    def get_violation_image_path(self, violation_id: str) -> Optional[str]:
        """Get violation image path from database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT screenshot_path FROM ppe_violations 
                        WHERE violation_id = %s
                    """, (violation_id,))
                    result = cursor.fetchone()
                    return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Failed to get violation image path: {e}")
            raise

    def get_whole_frame_image_path(self, violation_id: str) -> Optional[str]:
        """Get whole frame image path from database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT whole_frame_path FROM ppe_violations 
                        WHERE violation_id = %s
                    """, (violation_id,))
                    result = cursor.fetchone()
                    return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Failed to get whole frame image path: {e}")
            raise

    def get_violation_image(self, violation_id: str) -> Optional[bytes]:
        """Get violation image binary data (fallback for missing files) - not stored in DB"""
        # Binary image data is not stored in the database, only file paths
        return None

    def get_whole_frame_image(self, violation_id: str) -> Optional[bytes]:
        """Get whole frame image binary data (fallback for missing files) - not stored in DB"""
        # Binary image data is not stored in the database, only file paths
        return None
    
    def acknowledge_violation(self, violation_id: str, acknowledged_by: str = "web_user", notes: str = None):
        """Mark violation as acknowledged"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE ppe_violations 
                        SET acknowledged = true,
                            acknowledged_at = CURRENT_TIMESTAMP,
                            acknowledged_by = %s,
                            notes = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE violation_id = %s
                    """, (acknowledged_by, notes, violation_id))
                    conn.commit()
                    logger.info(f"Acknowledged violation: {violation_id}")
        except Exception as e:
            logger.error(f"Failed to acknowledge violation: {e}")
            raise
    
    def get_session_statistics(self) -> List[Dict]:
        """Get comprehensive session statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            session_name,
                            source_type,
                            MIN(timestamp) as first_violation,
                            MAX(timestamp) as last_violation,
                            COUNT(*) as total_violations,
                            COUNT(CASE WHEN acknowledged = false THEN 1 END) as unacknowledged_violations,
                            COUNT(DISTINCT person_id) as unique_persons,
                            COUNT(CASE WHEN violation_type = 'NO-Hardhat' THEN 1 END) as no_hardhat_count,
                            COUNT(CASE WHEN violation_type = 'NO-Mask' THEN 1 END) as no_mask_count,
                            COUNT(CASE WHEN violation_type = 'NO-Safety Vest' THEN 1 END) as no_vest_count,
                            AVG(confidence) as avg_confidence
                        FROM ppe_violations 
                        GROUP BY session_name, source_type
                        ORDER BY last_violation DESC
                    """)
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            raise
    
    def close_pool(self):
        """Close database connection pool"""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

# Global database manager instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        # Load from environment variables with defaults matching your setup
        host = os.getenv('DB_HOST', 'localhost')
        # Convert localhost to 127.0.0.1 to avoid IPv6 issues on Windows
        if host == 'localhost':
            host = '127.0.0.1'
        
        password = os.getenv('DB_PASSWORD', 'postgres')
        # If password is empty, don't pass it (for trust authentication)
        if not password:
            password = None
        
        db_manager = DatabaseManager(
            host=host,
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'vms_staging'),
            user=os.getenv('DB_USER', 'postgres'),
            password=password
        )
    return db_manager

def wait_for_db(max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for database to be ready"""
    db = get_db_manager()
    
    for attempt in range(max_retries):
        try:
            if db.test_connection():
                logger.info("Database is ready!")
                return True
        except Exception as e:
            logger.info(f"Waiting for database... Attempt {attempt + 1}/{max_retries}")
            time.sleep(delay)
    
    logger.error("Database is not ready after maximum retries")
    return False

if __name__ == "__main__":
    # Test database connection
    db = get_db_manager()
    if db.test_connection():
        print("✅ Database connection successful!")
        
        # Test getting violations
        violations = db.get_all_violations(limit=5)
        print(f"✅ Found {len(violations)} existing violations")
        
        # Get session statistics
        stats = db.get_session_statistics()
        print(f"✅ Session statistics: {len(stats)} sessions")
        
        # Test inserting a sample violation (for testing purposes)
        import cv2
        import numpy as np
        try:
            # Create a dummy test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            test_image[:] = [255, 0, 0]  # Blue image
            
            test_violation_id = db.save_violation_with_image(
                session_name="Test_Session_DB_Integration",
                violation_type="NO-Hardhat",
                person_id=1,
                frame_number=1,
                bbox=(10, 10, 50, 50),
                confidence=0.95,
                image_frame=test_image,
                whole_frame=test_image,
                severity="high",
                source_type="test",
                source_path=None
            )
            print(f"✅ Test violation saved successfully! ID: {test_violation_id}")
            
        except Exception as e:
            print(f"❌ Test violation save failed: {e}")
            
        print("✅ Database integration test completed!")
    else:
        print("❌ Database connection failed!")
