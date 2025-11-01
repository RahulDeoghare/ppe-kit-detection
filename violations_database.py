"""
Simple Database Manager for PPE Detection System
Single table approach - stores only violations with screenshots
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViolationsDatabase:
    """Simple database manager for PPE violations - single table only"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "ppe_detection",
                 user: str = "ppe_user", 
                 password: str = None):
        
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user
        }
        
        # Only add password if it's provided (for trust authentication)
        if password:
            self.connection_params['password'] = password
        
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            logger.info("Connected to PPE violations database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from database")
    
    def create_screenshots_directory(self):
        """Create screenshots directory if it doesn't exist"""
        screenshots_dir = "static/screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
            logger.info(f"Created screenshots directory: {screenshots_dir}")
        return screenshots_dir
    
    def save_violation(self, 
                      violation_type: str,
                      confidence: float,
                      bbox_x1: int, bbox_y1: int, bbox_x2: int, bbox_y2: int,
                      screenshot_path: str,
                      person_id: int = None,
                      session_name: str = "Live Detection",
                      frame_number: int = None,
                      source_type: str = "webcam",
                      severity: str = "medium") -> str:
        """
        Save a PPE violation to the database
        Returns the violation_id
        """
        try:
            with self.connection.cursor() as cursor:
                violation_id = str(uuid.uuid4())
                
                query = """
                INSERT INTO ppe_violations (
                    violation_id, violation_type, confidence, severity,
                    person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    source_type, session_name, frame_number, screenshot_path
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                cursor.execute(query, (
                    violation_id, violation_type, confidence, severity,
                    person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    source_type, session_name, frame_number, screenshot_path
                ))
                
                self.connection.commit()
                logger.info(f"Saved violation {violation_id}: {violation_type}")
                return violation_id
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to save violation: {e}")
            raise
    
    def get_all_violations(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all violations from the database"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                SELECT * FROM ppe_violations 
                ORDER BY timestamp DESC 
                LIMIT %s
                """
                cursor.execute(query, (limit,))
                violations = cursor.fetchall()
                
                # Convert to list of dictionaries
                return [dict(row) for row in violations]
                
        except Exception as e:
            logger.error(f"Failed to get violations: {e}")
            return []
    
    def get_unacknowledged_violations(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get unacknowledged violations"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                SELECT * FROM ppe_violations 
                WHERE acknowledged = FALSE
                ORDER BY timestamp DESC 
                LIMIT %s
                """
                cursor.execute(query, (limit,))
                violations = cursor.fetchall()
                
                return [dict(row) for row in violations]
                
        except Exception as e:
            logger.error(f"Failed to get unacknowledged violations: {e}")
            return []
    
    def acknowledge_violation(self, violation_id: str, acknowledged_by: str = "user"):
        """Mark a violation as acknowledged"""
        try:
            with self.connection.cursor() as cursor:
                query = """
                UPDATE ppe_violations 
                SET acknowledged = TRUE, 
                    acknowledged_at = CURRENT_TIMESTAMP,
                    acknowledged_by = %s
                WHERE violation_id = %s
                """
                cursor.execute(query, (acknowledged_by, violation_id))
                self.connection.commit()
                logger.info(f"Acknowledged violation {violation_id} by {acknowledged_by}")
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to acknowledge violation: {e}")
            raise
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all violations"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Total violations
                cursor.execute("SELECT COUNT(*) as total FROM ppe_violations")
                total = cursor.fetchone()['total']
                
                # Unacknowledged violations
                cursor.execute("SELECT COUNT(*) as unack FROM ppe_violations WHERE acknowledged = FALSE")
                unacknowledged = cursor.fetchone()['unack']
                
                # By violation type
                cursor.execute("""
                    SELECT violation_type, COUNT(*) as count 
                    FROM ppe_violations 
                    GROUP BY violation_type
                """)
                by_type = {row['violation_type']: row['count'] for row in cursor.fetchall()}
                
                # By severity
                cursor.execute("""
                    SELECT severity, COUNT(*) as count 
                    FROM ppe_violations 
                    GROUP BY severity
                """)
                by_severity = {row['severity']: row['count'] for row in cursor.fetchall()}
                
                # Recent violations (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) as recent 
                    FROM ppe_violations 
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                """)
                recent_24h = cursor.fetchone()['recent']
                
                return {
                    'total_violations': total,
                    'unacknowledged_violations': unacknowledged,
                    'acknowledged_violations': total - unacknowledged,
                    'by_violation_type': by_type,
                    'by_severity': by_severity,
                    'recent_24h': recent_24h
                }
                
        except Exception as e:
            logger.error(f"Failed to get violation summary: {e}")
            return {}
    
    def search_violations(self, 
                         violation_type: str = None,
                         severity: str = None,
                         acknowledged: bool = None,
                         session_name: str = None,
                         limit: int = 500) -> List[Dict[str, Any]]:
        """Search violations with filters"""
        try:
            conditions = []
            params = []
            
            if violation_type:
                conditions.append("violation_type = %s")
                params.append(violation_type)
            
            if severity:
                conditions.append("severity = %s")
                params.append(severity)
            
            if acknowledged is not None:
                conditions.append("acknowledged = %s")
                params.append(acknowledged)
            
            if session_name:
                conditions.append("session_name ILIKE %s")
                params.append(f"%{session_name}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                query = f"""
                SELECT * FROM ppe_violations 
                WHERE {where_clause}
                ORDER BY timestamp DESC 
                LIMIT %s
                """
                cursor.execute(query, params)
                violations = cursor.fetchall()
                
                return [dict(row) for row in violations]
                
        except Exception as e:
            logger.error(f"Failed to search violations: {e}")
            return []
    
    def delete_violation(self, violation_id: str):
        """Delete a specific violation"""
        try:
            with self.connection.cursor() as cursor:
                # First get the screenshot path to delete the file
                cursor.execute("SELECT screenshot_path FROM ppe_violations WHERE violation_id = %s", (violation_id,))
                result = cursor.fetchone()
                
                if result:
                    screenshot_path = result[0]
                    
                    # Delete from database
                    cursor.execute("DELETE FROM ppe_violations WHERE violation_id = %s", (violation_id,))
                    self.connection.commit()
                    
                    # Delete screenshot file if it exists
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                        logger.info(f"Deleted screenshot: {screenshot_path}")
                    
                    logger.info(f"Deleted violation {violation_id}")
                else:
                    logger.warning(f"Violation {violation_id} not found")
                    
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to delete violation: {e}")
            raise

# Global database instance
_db_instance = None

def get_violations_db():
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ViolationsDatabase()
    return _db_instance

def wait_for_database(max_retries: int = 10, delay: int = 2) -> bool:
    """Wait for database to be available"""
    for attempt in range(max_retries):
        try:
            db = ViolationsDatabase()
            db.disconnect()
            logger.info("Database is available")
            return True
        except Exception as e:
            logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    
    logger.error("Database not available after maximum retries")
    return False
