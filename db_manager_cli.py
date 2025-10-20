#!/usr/bin/env python3
"""
Database management script for PPE Detection System
This script provides utilities to manage the PostgreSQL database
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from database_manager import get_db_manager, wait_for_db
import json

def setup_database():
    """Initialize and test database connection"""
    print("🔄 Setting up database connection...")
    
    if not wait_for_db(max_retries=10):
        print("❌ Failed to connect to database. Please ensure PostgreSQL is running.")
        return False
    
    db = get_db_manager()
    
    if db.test_connection():
        print("✅ Database connection successful!")
        return True
    else:
        print("❌ Database connection failed!")
        return False

def create_test_session():
    """Create a test detection session"""
    print("🔄 Creating test detection session...")
    
    db = get_db_manager()
    session_id = db.create_detection_session(
        session_name="Test Session - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source_type="webcam",
        source_path="/dev/video0"
    )
    
    print(f"✅ Created test session: {session_id}")
    return session_id

def list_sessions():
    """List all detection sessions"""
    print("📋 Listing all detection sessions...")
    
    db = get_db_manager()
    stats = db.get_session_statistics()
    
    if not stats:
        print("ℹ️  No sessions found.")
        return
    
    print(f"\n{'Session ID':<38} {'Name':<20} {'Type':<12} {'Start Time':<20} {'Violations':<10}")
    print("-" * 110)
    
    for session in stats:
        session_id = str(session['session_id'])[:36]
        name = (session['session_name'] or 'Unnamed')[:18]
        source_type = session['source_type'][:10]
        start_time = session['start_time'].strftime("%Y-%m-%d %H:%M") if session['start_time'] else 'N/A'
        violations = session['total_violations'] or 0
        
        print(f"{session_id:<38} {name:<20} {source_type:<12} {start_time:<20} {violations:<10}")

def show_violations(session_id=None, unacknowledged_only=False):
    """Show violations for a session or all violations"""
    db = get_db_manager()
    
    if unacknowledged_only:
        print("⚠️  Listing unacknowledged violations...")
        violations = db.get_unacknowledged_violations(session_id)
    elif session_id:
        print(f"📋 Listing violations for session: {session_id}")
        violations = db.get_session_violations(session_id)
    else:
        print("📋 Listing all recent violations...")
        # Get violations from all recent sessions
        sessions = db.get_session_statistics()
        if sessions:
            latest_session_id = str(sessions[0]['session_id'])
            violations = db.get_session_violations(latest_session_id)
        else:
            violations = []
    
    if not violations:
        print("ℹ️  No violations found.")
        return
    
    print(f"\n{'Violation ID':<38} {'Type':<15} {'Person':<6} {'Confidence':<10} {'Time':<16} {'Acked':<6}")
    print("-" * 100)
    
    for violation in violations:
        violation_id = str(violation['violation_id'])[:36]
        v_type = violation['violation_type'][:13]
        person_id = violation['person_id']
        confidence = f"{violation['confidence']:.2f}"
        timestamp = violation['timestamp'].strftime("%m-%d %H:%M") if violation['timestamp'] else 'N/A'
        acknowledged = "Yes" if violation['acknowledged'] else "No"
        
        print(f"{violation_id:<38} {v_type:<15} {person_id:<6} {confidence:<10} {timestamp:<16} {acknowledged:<6}")

def acknowledge_violation(violation_id, acknowledged_by="admin"):
    """Acknowledge a violation"""
    print(f"🔄 Acknowledging violation: {violation_id}")
    
    db = get_db_manager()
    db.acknowledge_violation(violation_id, acknowledged_by)
    
    print("✅ Violation acknowledged successfully!")

def show_session_summary(session_id):
    """Show detailed summary for a session"""
    print(f"📊 Session Summary for: {session_id}")
    
    db = get_db_manager()
    
    # Get session statistics
    stats = db.get_session_statistics(session_id)
    if not stats:
        print("❌ Session not found!")
        return
    
    session = stats[0]
    print(f"\n📋 Session Details:")
    print(f"   Name: {session['session_name'] or 'Unnamed'}")
    print(f"   Type: {session['source_type']}")
    print(f"   Start: {session['start_time']}")
    print(f"   End: {session['end_time'] or 'Ongoing'}")
    print(f"   Total Frames: {session['total_frames'] or 0}")
    print(f"   Persons Detected: {session['total_persons'] or 0}")
    print(f"   Total Violations: {session['total_violations'] or 0}")
    print(f"   Unacknowledged: {session['unacknowledged_violations'] or 0}")
    
    # Get violation breakdown
    summary = db.get_violation_summary(session_id)
    if summary:
        print(f"\n⚠️  Violation Breakdown:")
        for item in summary:
            print(f"   {item['violation_type']}: {item['total_violations']} violations "
                  f"({item['unique_persons']} persons, avg confidence: {item['avg_confidence']:.2f})")

def cleanup_old_sessions(days=30):
    """Clean up old sessions"""
    print(f"🗑️  Cleaning up sessions older than {days} days...")
    
    db = get_db_manager()
    deleted_count = db.cleanup_old_sessions(days)
    
    print(f"✅ Cleaned up {deleted_count} old sessions")

def export_session_data(session_id, output_file=None):
    """Export session data to JSON"""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"session_export_{session_id[:8]}_{timestamp}.json"
    
    print(f"📤 Exporting session data to: {output_file}")
    
    db = get_db_manager()
    
    # Get session info
    stats = db.get_session_statistics(session_id)
    violations = db.get_session_violations(session_id)
    summary = db.get_violation_summary(session_id)
    
    export_data = {
        'session_info': stats[0] if stats else None,
        'violations': violations,
        'summary': summary,
        'export_timestamp': datetime.now().isoformat()
    }
    
    # Convert datetime objects to strings for JSON serialization
    def serialize_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=serialize_datetime)
    
    print(f"✅ Session data exported successfully!")

def main():
    parser = argparse.ArgumentParser(description="PPE Detection Database Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    subparsers.add_parser('setup', help='Initialize and test database connection')
    
    # Sessions commands
    subparsers.add_parser('list-sessions', help='List all detection sessions')
    
    test_parser = subparsers.add_parser('test-session', help='Create a test session')
    
    summary_parser = subparsers.add_parser('session-summary', help='Show session summary')
    summary_parser.add_argument('session_id', help='Session ID')
    
    # Violations commands
    violations_parser = subparsers.add_parser('list-violations', help='List violations')
    violations_parser.add_argument('--session-id', help='Filter by session ID')
    violations_parser.add_argument('--unacknowledged', action='store_true', help='Show only unacknowledged violations')
    
    ack_parser = subparsers.add_parser('acknowledge', help='Acknowledge a violation')
    ack_parser.add_argument('violation_id', help='Violation ID to acknowledge')
    ack_parser.add_argument('--by', default='admin', help='Who is acknowledging')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old sessions')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days old to clean up (default: 30)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export session data')
    export_parser.add_argument('session_id', help='Session ID to export')
    export_parser.add_argument('--output', help='Output file name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'setup':
            setup_database()
        elif args.command == 'list-sessions':
            list_sessions()
        elif args.command == 'test-session':
            create_test_session()
        elif args.command == 'session-summary':
            show_session_summary(args.session_id)
        elif args.command == 'list-violations':
            show_violations(args.session_id, args.unacknowledged)
        elif args.command == 'acknowledge':
            acknowledge_violation(args.violation_id, args.by)
        elif args.command == 'cleanup':
            cleanup_old_sessions(args.days)
        elif args.command == 'export':
            export_session_data(args.session_id, args.output)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()