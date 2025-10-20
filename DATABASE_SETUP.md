# PPE Detection System with PostgreSQL Database

A comprehensive Personal Protective Equipment (PPE) detection system using YOLO v8, Flask web interface, and PostgreSQL database for storing detection data locally using Docker.

## Features

- 🎯 Real-time PPE detection (Hardhat, Mask, Safety Vest)
- 📊 PostgreSQL database for persistent data storage
- 🐳 Docker containerization for easy deployment
- 🌐 Web interface for monitoring and management
- 📝 Violation tracking and acknowledgment system
- 📈 Session statistics and reporting
- 🔍 Comprehensive violation analysis
- 🚨 Real-time alerts for safety violations

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   YOLO Model    │───▶│  Flask Web App  │───▶│   PostgreSQL    │
│  (Detection)    │    │   (Interface)   │    │   (Database)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Video/Webcam   │    │ Web Dashboard   │    │   PgAdmin UI    │
│   (Input)       │    │   (Control)     │    │  (Management)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Database Schema

### Core Tables

1. **detection_sessions** - Tracks detection sessions
2. **person_detections** - Stores detected persons with bounding boxes
3. **object_detections** - Stores all detected objects
4. **ppe_violations** - Records PPE safety violations
5. **violation_summary** - Aggregated violation statistics

### Key Features

- UUID primary keys for all records
- Automatic violation summary updates via triggers
- Indexed columns for optimal query performance
- Comprehensive violation tracking with severity levels
- Session-based organization of detections

## Installation & Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- YOLO model weights file (`YOLO-Weights/ppe.pt`)

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd ppe-kit-detection
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Copy and customize the `.env` file:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ppe_detection
DB_USER=ppe_user
DB_PASSWORD=ppe_password

# Flask Configuration
FLASK_SECRET_KEY=konsberg
FLASK_DEBUG=True
FLASK_UPLOAD_FOLDER=static/files
```

### Step 4: Start Database with Docker

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database on port 5432
- PgAdmin interface on port 8080

### Step 5: Verify Database Setup

```bash
python db_manager_cli.py setup
```

### Step 6: Start the Application

```bash
python app.py
```

The web interface will be available at `http://localhost:5000`

## Usage

### Web Interface

1. **Home Page** (`/`) - Main navigation
2. **Live Feed** (`/live_feed`) - Real-time webcam detection
3. **Video Upload** (`/FrontPage`) - Upload and analyze video files
4. **Violations Dashboard** (`/violations`) - View and manage violations

### Command Line Management

The `db_manager_cli.py` script provides comprehensive database management:

```bash
# Setup and test database
python db_manager_cli.py setup

# List all detection sessions
python db_manager_cli.py list-sessions

# Show violations for a session
python db_manager_cli.py list-violations --session-id <session_id>

# Show only unacknowledged violations
python db_manager_cli.py list-violations --unacknowledged

# Acknowledge a violation
python db_manager_cli.py acknowledge <violation_id> --by admin

# Show session summary
python db_manager_cli.py session-summary <session_id>

# Export session data
python db_manager_cli.py export <session_id> --output session_data.json

# Clean up old sessions (30+ days)
python db_manager_cli.py cleanup --days 30

# Create test session
python db_manager_cli.py test-session
```

### API Endpoints

- `GET /api/violations` - Get violations data
- `POST /api/acknowledge_violation` - Acknowledge a violation

## Database Management

### Accessing PgAdmin

1. Open `http://localhost:8080`
2. Login with:
   - Email: `admin@ppe.com`
   - Password: `admin123`
3. Connect to PostgreSQL server:
   - Host: `postgres`
   - Port: `5432`
   - Database: `ppe_detection`
   - Username: `ppe_user`
   - Password: `ppe_password`

### Direct Database Access

```bash
# Connect to PostgreSQL container
docker exec -it ppe_detection_db psql -U ppe_user -d ppe_detection

# Example queries
SELECT * FROM violation_details LIMIT 10;
SELECT * FROM session_statistics;
SELECT violation_type, COUNT(*) FROM ppe_violations GROUP BY violation_type;
```

### Database Backup and Restore

```bash
# Backup database
docker exec ppe_detection_db pg_dump -U ppe_user ppe_detection > backup.sql

# Restore database
docker exec -i ppe_detection_db psql -U ppe_user -d ppe_detection < backup.sql
```

## Configuration Options

### Detection Parameters

- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.5)
- `GPU_ENABLED`: Enable GPU acceleration if available
- `YOLO_MODEL_PATH`: Path to YOLO model weights

### Database Settings

- `AUTO_CLEANUP_DAYS`: Automatically clean sessions older than X days
- `SAVE_VIOLATIONS_TO_JSON`: Keep JSON file backups
- `SAVE_VIOLATIONS_TO_DB`: Enable database storage

## Monitoring and Alerts

### Real-time Monitoring

The system provides real-time monitoring of:
- Active detection sessions
- Violation counts by type
- Person detection statistics
- System performance metrics

### Violation Severity Levels

- **Low**: Confidence < 0.6
- **Medium**: Confidence 0.6-0.8
- **High**: Confidence > 0.8
- **Critical**: Manual assignment for severe cases

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check if containers are running
   docker-compose ps
   
   # Check database logs
   docker-compose logs postgres
   
   # Test connection
   python db_manager_cli.py setup
   ```

2. **YOLO Model Not Found**
   - Ensure `YOLO-Weights/ppe.pt` exists
   - Check model path in configuration

3. **GPU Issues**
   - Verify CUDA installation
   - Check GPU availability with `python test_gpu.py`

4. **Permission Errors**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   chmod +x db_manager_cli.py
   ```

### Performance Optimization

1. **Database Performance**
   - Regular VACUUM operations
   - Index maintenance
   - Cleanup old sessions

2. **Detection Performance**
   - Use GPU acceleration
   - Optimize input resolution
   - Adjust confidence thresholds

## Development

### Adding New Detection Classes

1. Update `classNames` in `YOLO_Video.py`
2. Add new enum values in `database/init.sql`
3. Update violation logic as needed

### Extending the Database

1. Add migration scripts in `database/migrations/`
2. Update schema in `database/init.sql`
3. Modify `database_manager.py` methods

### Custom Alerts

Implement custom alerting by extending the violation detection logic in `YOLO_Video.py`.

## Security Considerations

- Change default passwords in `.env`
- Use SSL/TLS for production deployments
- Implement proper authentication for web interface
- Regular security updates for dependencies

## License

This project is licensed under the MIT License.

## Support

For issues and support, please check the troubleshooting section or create an issue in the repository.