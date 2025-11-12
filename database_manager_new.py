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
                 database: str = "ppe_detection",
                 user: str = "ppe_user", 
                 password: str = None,
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
    
    def save_violation_with_image(self,
                                session_name: str,
                                violation_type: str,
                                person_id: int,
                                frame_number: int,
                                bbox: tuple,
                                confidence: float,
                                image_frame,
                                whole_frame=None,
                                severity: str = "HIGH",
                                source_type: str = "webcam",
                                source_path: str = None):
        """Save violation with screenshot to database - SINGLE TABLE APPROACH"""
        
        violation_id = str(uuid.uuid4())
        x1, y1, x2, y2 = bbox
        timestamp = datetime.now(timezone.utc)
        
        # Save cropped violation image
        image_data = None
        image_path = None
        
        if image_frame is not None:
            try:
                # Create violations directory if it doesn't exist
                violations_dir = "violations"
                os.makedirs(violations_dir, exist_ok=True)
                
                # Save image file
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_filename = f"violation_{timestamp_str}.jpg"
                image_path = os.path.join(violations_dir, image_filename)
                
                cv2.imwrite(image_path, image_frame)
                
                # Also encode as binary data for database storage (backup)
                _, buffer = cv2.imencode('.jpg', image_frame)
                image_data = buffer.tobytes()
                
                logger.info(f"Saved violation image: {image_path}")
                
            except Exception as e:
                logger.error(f"Failed to save violation image: {e}")
        
        # Save whole frame image
        whole_frame_data = None
        whole_frame_path = None
        
        if whole_frame is not None:
            try:
                # Save whole frame file
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                whole_frame_filename = f"whole_frame_{timestamp_str}.jpg"
                whole_frame_path = os.path.join(violations_dir, whole_frame_filename)
                
                cv2.imwrite(whole_frame_path, whole_frame)
                
                # Also encode as binary data for database storage (backup)
                _, buffer = cv2.imencode('.jpg', whole_frame)
                whole_frame_data = buffer.tobytes()
                
                logger.info(f"Saved whole frame image: {whole_frame_path}")
                
            except Exception as e:
                logger.error(f"Failed to save whole frame image: {e}")
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO ppe_violations 
                        (violation_id, session_name, source_type, source_path, violation_type, 
                         person_id, frame_number, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                         confidence, severity, image_path, image_data, whole_frame_path, 
                         whole_frame_data, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (violation_id, session_name, source_type, source_path, violation_type,
                         person_id, frame_number, x1, y1, x2, y2, 
                         confidence, severity, image_path, image_data, whole_frame_path,
                         whole_frame_data, timestamp))
                    conn.commit()
                    
                    logger.info(f"Saved violation: {violation_type} for person {person_id} in session {session_name}")
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
                    cursor.execute("SELECT COUNT(*) FROM ppe_violations")
                    total_count = cursor.fetchone()[0]
                    
                    # Get paginated results
                    offset = (page - 1) * per_page
                    cursor.execute("""
                        SELECT 
                            violation_id,
                            session_name,
                            source_type,
                            source_path,
                            violation_type,
                            person_id,
                            frame_number,
                            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                            confidence,
                            severity,
                            image_path,
                            whole_frame_path,
                            acknowledged,
                            acknowledged_by,
                            acknowledged_at,
                            notes,
                            timestamp,
                            created_at
                        FROM ppe_violations 
                        ORDER BY timestamp DESC
                        LIMIT %s OFFSET %s
                    """, (per_page, offset))
                    
                    violations = cursor.fetchall()
                    return violations, total_count
        except Exception as e:
            logger.error(f"Failed to get paginated violations: {e}")
            raise

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
                            source_path,
                            violation_type,
                            person_id,
                            frame_number,
                            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                            confidence,
                            severity,
                            image_path,
                            whole_frame_path,
                            acknowledged,
                            acknowledged_by,
                            acknowledged_at,
                            notes,
                            timestamp,
                            created_at
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
                        SELECT image_path FROM ppe_violations 
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
        """Get violation image binary data (fallback for missing files)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT image_data FROM ppe_violations 
                        WHERE violation_id = %s
                    """, (violation_id,))
                    result = cursor.fetchone()
                    return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Failed to get violation image: {e}")
            raise

    def get_whole_frame_image(self, violation_id: str) -> Optional[bytes]:
        """Get whole frame image binary data (fallback for missing files)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT whole_frame_data FROM ppe_violations 
                        WHERE violation_id = %s
                    """, (violation_id,))
                    result = cursor.fetchone()
                    return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Failed to get whole frame image: {e}")
            raise
    
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
        # Load from environment variables
        host = os.getenv('DB_HOST', 'localhost')
        # Convert localhost to 127.0.0.1 to avoid IPv6 issues on Windows
        if host == 'localhost':
            host = '127.0.0.1'
        
        password = os.getenv('DB_PASSWORD', 'ppe_password')
        # If password is empty, don't pass it (for trust authentication)
        if not password:
            password = None
        
        db_manager = DatabaseManager(
            host=host,
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'ppe_detection'),
            user=os.getenv('DB_USER', 'ppe_user'),
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
        print(f"✅ Found {len(violations)} violations")
        
        # Get session statistics
        stats = db.get_session_statistics()
        print(f"✅ Session statistics: {len(stats)} sessions")
        print("✅ Test completed successfully!")
    else:
        print("❌ Database connection failed!")
