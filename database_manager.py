"""
Database connection and management module for PPE Detection System
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and operations for PPE detection system"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "ppe_detection",
                 user: str = "ppe_user", 
                 password: str = "ppe_password",
                 min_connections: int = 1,
                 max_connections: int = 10):
        
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
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
    
    def create_detection_session(self, 
                                session_name: str = None,
                                source_type: str = "webcam",
                                source_path: str = None) -> str:
        """Create a new detection session and return session ID"""
        
        session_id = str(uuid.uuid4())
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO detection_sessions 
                        (session_id, session_name, source_type, source_path)
                        VALUES (%s, %s, %s, %s)
                    """, (session_id, session_name, source_type, source_path))
                    conn.commit()
                    logger.info(f"Created detection session: {session_id}")
                    return session_id
        except Exception as e:
            logger.error(f"Failed to create detection session: {e}")
            raise
    
    def end_detection_session(self, session_id: str):
        """Mark detection session as ended"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE detection_sessions 
                        SET end_time = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE session_id = %s
                    """, (session_id,))
                    conn.commit()
                    logger.info(f"Ended detection session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to end detection session: {e}")
            raise
    
    def save_person_detection(self, 
                            session_id: str,
                            person_id: int,
                            frame_number: int,
                            bbox: Tuple[int, int, int, int],
                            confidence: float,
                            timestamp: datetime = None) -> str:
        """Save person detection to database"""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        detection_id = str(uuid.uuid4())
        x1, y1, x2, y2 = bbox
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO person_detections 
                        (detection_id, session_id, person_id, frame_number, 
                         timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (detection_id, session_id, person_id, frame_number, 
                         timestamp, x1, y1, x2, y2, confidence))
                    conn.commit()
                    return detection_id
        except Exception as e:
            logger.error(f"Failed to save person detection: {e}")
            raise
    
    def save_object_detection(self,
                            session_id: str,
                            person_detection_id: str = None,
                            frame_number: int = None,
                            class_name: str = "",
                            confidence: float = 0.0,
                            bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
                            timestamp: datetime = None) -> str:
        """Save object detection to database"""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        detection_id = str(uuid.uuid4())
        x1, y1, x2, y2 = bbox
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO object_detections 
                        (detection_id, session_id, person_detection_id, frame_number,
                         timestamp, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (detection_id, session_id, person_detection_id, frame_number,
                         timestamp, class_name, confidence, x1, y1, x2, y2))
                    conn.commit()
                    return detection_id
        except Exception as e:
            logger.error(f"Failed to save object detection: {e}")
            raise
    
    def save_violation(self,
                      session_id: str,
                      person_detection_id: str,
                      violation_type: str,
                      frame_number: int,
                      confidence: float,
                      bbox: Tuple[int, int, int, int],
                      severity: str = "medium",
                      timestamp: datetime = None) -> str:
        """Save PPE violation to database"""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        violation_id = str(uuid.uuid4())
        x1, y1, x2, y2 = bbox
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO ppe_violations 
                        (violation_id, session_id, person_detection_id, violation_type,
                         frame_number, timestamp, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, severity)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (violation_id, session_id, person_detection_id, violation_type,
                         frame_number, timestamp, confidence, x1, y1, x2, y2, severity))
                    conn.commit()
                    logger.info(f"Saved violation: {violation_type} for person {person_detection_id}")
                    return violation_id
        except Exception as e:
            logger.error(f"Failed to save violation: {e}")
            raise
    
    def get_session_violations(self, session_id: str) -> List[Dict]:
        """Get all violations for a session"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM violation_details 
                        WHERE session_id = %s 
                        ORDER BY timestamp DESC
                    """, (session_id,))
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get session violations: {e}")
            raise
    
    def get_unacknowledged_violations(self, session_id: str = None) -> List[Dict]:
        """Get unacknowledged violations"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if session_id:
                        cursor.execute("""
                            SELECT * FROM violation_details 
                            WHERE acknowledged = false AND session_id = %s
                            ORDER BY timestamp DESC
                        """, (session_id,))
                    else:
                        cursor.execute("""
                            SELECT * FROM violation_details 
                            WHERE acknowledged = false
                            ORDER BY timestamp DESC
                        """)
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get unacknowledged violations: {e}")
            raise
    
    def acknowledge_violation(self, violation_id: str, acknowledged_by: str = None):
        """Mark violation as acknowledged"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE ppe_violations 
                        SET acknowledged = true,
                            acknowledged_at = CURRENT_TIMESTAMP,
                            acknowledged_by = %s
                        WHERE violation_id = %s
                    """, (acknowledged_by, violation_id))
                    conn.commit()
                    logger.info(f"Acknowledged violation: {violation_id}")
        except Exception as e:
            logger.error(f"Failed to acknowledge violation: {e}")
            raise
    
    def get_session_statistics(self, session_id: str = None) -> List[Dict]:
        """Get session statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if session_id:
                        cursor.execute("""
                            SELECT * FROM session_statistics 
                            WHERE session_id = %s
                        """, (session_id,))
                    else:
                        cursor.execute("SELECT * FROM session_statistics ORDER BY start_time DESC")
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            raise
    
    def get_violation_summary(self, session_id: str) -> List[Dict]:
        """Get violation summary for a session"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM violation_summary 
                        WHERE session_id = %s
                        ORDER BY total_violations DESC
                    """, (session_id,))
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get violation summary: {e}")
            raise
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up sessions older than specified days"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        DELETE FROM detection_sessions 
                        WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    """, (days_old,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    logger.info(f"Cleaned up {deleted_count} old sessions")
                    return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
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
        db_manager = DatabaseManager(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'ppe_detection'),
            user=os.getenv('DB_USER', 'ppe_user'),
            password=os.getenv('DB_PASSWORD', 'ppe_password')
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
        
        # Create a test session
        session_id = db.create_detection_session("Test Session", "webcam", "/dev/video0")
        print(f"✅ Created test session: {session_id}")
        
        # Get session statistics
        stats = db.get_session_statistics(session_id)
        print(f"✅ Session statistics: {stats}")
        
        # End session
        db.end_detection_session(session_id)
        print("✅ Test completed successfully!")
    else:
        print("❌ Database connection failed!")