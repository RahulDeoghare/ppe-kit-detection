#!/bin/bash

# PPE Detection System Setup Script
# This script automates the setup process for the PPE detection system with PostgreSQL

set -e  # Exit on any error

echo "🚀 Starting PPE Detection System Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $python_version is installed"
}

# Create .env file if it doesn't exist
create_env_file() {
    print_status "Checking .env file..."
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating default .env file..."
        cp .env.example .env 2>/dev/null || echo "# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ppe_detection
DB_USER=ppe_user
DB_PASSWORD=ppe_password

# Flask Configuration
FLASK_SECRET_KEY=konsberg
FLASK_DEBUG=True
FLASK_UPLOAD_FOLDER=static/files

# PostgreSQL Admin
PGADMIN_EMAIL=admin@ppe.com
PGADMIN_PASSWORD=admin123

# Application Settings
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=50MB
SUPPORTED_EXTENSIONS=mp4,avi,mov,mkv,jpg,jpeg,png,bmp

# Model Configuration
YOLO_MODEL_PATH=YOLO-Weights/ppe.pt
CONFIDENCE_THRESHOLD=0.5
GPU_ENABLED=true

# Detection Settings
SAVE_VIOLATIONS_TO_JSON=true
SAVE_VIOLATIONS_TO_DB=true
AUTO_CLEANUP_DAYS=30" > .env
        print_success "Created .env file with default settings"
    else
        print_success ".env file already exists"
    fi
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Check YOLO model weights
check_yolo_model() {
    print_status "Checking YOLO model weights..."
    
    if [ ! -d "YOLO-Weights" ]; then
        mkdir -p YOLO-Weights
        print_warning "YOLO-Weights directory created"
    fi
    
    if [ ! -f "YOLO-Weights/ppe.pt" ]; then
        print_warning "YOLO model weights (ppe.pt) not found in YOLO-Weights/"
        print_warning "Please ensure you have the trained YOLO model file"
        print_warning "You can download or train your own model and place it in YOLO-Weights/ppe.pt"
    else
        print_success "YOLO model weights found"
    fi
}

# Start Docker containers
start_docker_services() {
    print_status "Starting Docker services..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Start containers
    docker-compose up -d
    
    print_success "Docker containers started"
    print_status "PostgreSQL is available on port 5432"
    print_status "PgAdmin is available on port 8080"
}

# Wait for database to be ready
wait_for_database() {
    print_status "Waiting for database to be ready..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Wait for database
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if python3 -c "from database_manager import wait_for_db; exit(0 if wait_for_db(max_retries=1) else 1)" 2>/dev/null; then
            print_success "Database is ready!"
            break
        fi
        
        print_status "Waiting for database... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_error "Database failed to start after $max_attempts attempts"
        exit 1
    fi
}

# Test database connection
test_database() {
    print_status "Testing database connection..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    if python3 db_manager_cli.py setup; then
        print_success "Database connection successful!"
    else
        print_error "Database connection failed!"
        exit 1
    fi
}

# Create test session
create_test_data() {
    print_status "Creating test session..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    if python3 db_manager_cli.py test-session; then
        print_success "Test session created!"
    else
        print_warning "Failed to create test session"
    fi
}

# Make scripts executable
make_executable() {
    print_status "Making scripts executable..."
    chmod +x db_manager_cli.py
    chmod +x setup.sh
    print_success "Scripts are now executable"
}

# Print final instructions
print_instructions() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Start the Flask application:"
    echo "   source venv/bin/activate"
    echo "   python app.py"
    echo ""
    echo "2. Access the web interface:"
    echo "   http://localhost:5000"
    echo ""
    echo "3. Access PgAdmin (database management):"
    echo "   http://localhost:8080"
    echo "   Email: admin@ppe.com"
    echo "   Password: admin123"
    echo ""
    echo "4. Manage database via CLI:"
    echo "   python db_manager_cli.py --help"
    echo ""
    echo "📁 Important files:"
    echo "   - .env: Configuration settings"
    echo "   - DATABASE_SETUP.md: Detailed documentation"
    echo "   - docker-compose.yml: Docker services configuration"
    echo ""
    echo "🔧 Useful commands:"
    echo "   - View violations: python db_manager_cli.py list-violations"
    echo "   - View sessions: python db_manager_cli.py list-sessions"
    echo "   - Stop services: docker-compose down"
    echo ""
}

# Main execution
main() {
    echo "🔍 System Requirements Check"
    check_docker
    check_python
    
    echo ""
    echo "⚙️  Environment Setup"
    create_env_file
    make_executable
    
    echo ""
    echo "📦 Dependencies Installation"
    install_dependencies
    check_yolo_model
    
    echo ""
    echo "🐳 Docker Services"
    start_docker_services
    wait_for_database
    
    echo ""
    echo "🔬 Database Testing"
    test_database
    create_test_data
    
    echo ""
    print_instructions
}

# Run main function
main