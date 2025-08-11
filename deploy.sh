#!/bin/bash

# Hate Speech Detection API Deployment Script
set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE_NAME="hate-speech-detector"
DOCKER_TAG="latest"
COMPOSE_FILE="docker-compose.yml"
# MODEL_PATH tidak perlu diset di sini karena sudah diatur di docker-compose.yml dan app.py
# ENV_FILE=".env" # Tidak digunakan langsung oleh deploy.sh, tapi bisa dipertimbangkan untuk env vars

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (minimum 2GB)
    if command -v df &> /dev/null; then
        available_space=$(df . | tail -1 | awk '{print $4}')
        # 2GB = 2 * 1024 * 1024 KB = 2097152 KB
        if [ "$available_space" -lt 2097152 ]; then
            log_warning "Low disk space. At least 2GB is recommended."
        fi
    fi
    
    log_success "System requirements check completed."
}

check_model_files() {
    log_info "Checking model files in ./app/model/optimized..."
    MODEL_DIR="./app/model/optimized"
    if [ ! -d "$MODEL_DIR" ]; then
        log_warning "Model directory not found at $MODEL_DIR"
        log_info "Please ensure your model files (e.g., config.json, pytorch_model.bin, tokenizer files) are placed in this directory."
        log_info "The API will start, but model functionality will be limited or fail if files are missing."
        mkdir -p "$MODEL_DIR" || true # Create if not exists, ignore error if it fails
    else
        log_success "Model directory found at $MODEL_DIR"
    fi
}

create_persistent_directories() {
    log_info "Creating persistent directories for logs, results, uploads, plots..."
    mkdir -p logs results uploads plots
    log_success "Persistent directories created."
}

build_and_deploy_services() {
    log_info "Building and deploying services with Docker Compose..."
    
    # Use docker compose or docker-compose
    if docker compose version &> /dev/null; then
        docker compose up -d --build --remove-orphans
    elif docker-compose --version &> /dev/null; then
        docker-compose up -d --build --remove-orphans
    else
        log_error "Neither 'docker compose' nor 'docker-compose' is available!"
        exit 1
    fi
    
    log_success "Services built and deployed successfully."
}

wait_for_services() {
    log_info "Waiting for API service to be ready (max 120 seconds)..."
    
    # Wait for main API health check via Nginx (port 80)
    for i in {1..60}; do # 60 retries * 2 seconds = 120 seconds
        if curl -sf http://localhost/health > /dev/null 2>&1; then
            log_success "API service is ready!"
            break
        fi
        
        if [ $i -eq 60 ]; then
            log_error "API service failed to start within timeout."
            show_logs
            exit 1
        fi
        
        echo -n "."
        sleep 2
    done
}

run_health_checks() {
    log_info "Running comprehensive health checks..."
    
    # Test API endpoints via Nginx (port 80)
    endpoints=(
        "/"
        "/health"
        "/model/info"
        "/docs"
        "/redoc"
    )
    
    all_ok=true
    for endpoint in "${endpoints[@]}"; do
        if curl -sf "http://localhost${endpoint}" > /dev/null 2>&1; then
            log_success "✓ ${endpoint} (via Nginx)"
        else
            log_error "✗ ${endpoint} (via Nginx) - Failed or not ready."
            all_ok=false
        fi
    done

    if $all_ok; then
        log_success "All basic health checks passed."
    else
        log_warning "Some health checks failed. Check logs for details."
    fi
}

show_status() {
    log_info "Service Status:"
    
    if docker compose version &> /dev/null; then
        docker compose ps
    else
        docker-compose ps
    fi
    
    echo
    log_info "Access URLs:"
    echo "🌐 Main Application (Frontend): http://localhost"
    echo "📚 API Documentation (Swagger UI): http://localhost/docs"
    echo "❤️ Health Check: http://localhost/health"
    echo "🔍 Model Info: http://localhost/model/info"
    echo "Direct API (for debugging, if exposed): http://localhost:8000"
}

show_logs() {
    log_info "Recent application logs (hate-speech-api service):"
    if docker compose version &> /dev/null; then
        docker compose logs --tail=50 hate-speech-api
    else
        docker-compose logs --tail=50 hate-speech-api
    fi
    echo
    log_info "Recent Nginx logs (hate-speech-nginx service):"
    if docker compose version &> /dev/null; then
        docker compose logs --tail=20 hate-speech-nginx
    else
        docker-compose logs --tail=20 hate-speech-nginx
    fi
}

stop_services() {
    log_info "Stopping services..."
    if docker compose version &> /dev/null; then
        docker compose down
    else
        docker-compose down
    fi
    log_success "Services stopped."
}

cleanup() {
    log_info "Cleaning up containers and networks..."
    
    if docker compose version &> /dev/null; then
        docker compose down --volumes --remove-orphans
    else
        docker-compose down --volumes --remove-orphans
    fi
    
    log_success "Cleanup completed. Persistent volumes (logs, results, uploads, plots) are retained."
    log_info "To remove persistent data, manually delete 'logs', 'results', 'uploads', 'plots' folders."
}

show_help() {
    echo "Hate Speech Detection API Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  deploy     - Full deployment (default): build, start, wait, test, status"
    echo "  build      - Build Docker images only"
    echo "  start      - Start services (assumes images are built)"
    echo "  stop       - Stop services"
    echo "  restart    - Restart services"
    echo "  status     - Show service status"
    echo "  logs       - Show application and Nginx logs"
    echo "  test       - Run health checks"
    echo "  cleanup    - Clean up containers, networks, and anonymous volumes"
    echo "  help       - Show this help message"
    echo
    echo "Examples:"
    echo "  $0                    # Full deployment"
    echo "  $0 deploy            # Full deployment"
    echo "  $0 build             # Build images only"
    echo "  $0 logs              # Show logs"
    echo "  $0 stop              # Stop running services"
}

# Main execution
main() {
    case "${1:-deploy}" in
        "deploy")
            log_info "Starting full deployment..."
            check_requirements
            check_model_files
            create_persistent_directories
            build_and_deploy_services
            wait_for_services
            run_health_checks
            show_status
            log_success "🎉 Deployment completed successfully!"
            echo
            log_info "You can now access the application at: http://localhost"
            log_info "API documentation: http://localhost/docs"
            ;;
        "build")
            check_requirements
            check_model_files
            build_and_deploy_services # --build flag handles building
            log_success "Docker images built successfully."
            ;;
        "start")
            check_requirements
            create_persistent_directories
            build_and_deploy_services # This will also build if images are outdated
            wait_for_services
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            log_info "Restarting services..."
            stop_services
            start_services # Re-use start logic
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "test")
            run_health_checks
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Helper for restart command
start_services() {
    log_info "Starting services..."
    if docker compose version &> /dev/null; then
        docker compose up -d --remove-orphans
    elif docker-compose --version &> /dev/null; then
        docker-compose up -d --remove-orphans
    else
        log_error "Neither 'docker compose' nor 'docker-compose' is available!"
        exit 1
    fi
    wait_for_services
    show_status
}

# Handle script interruption
trap 'echo -e "\n${YELLOW}[INFO]${NC} Script interrupted. Run '\''./deploy.sh cleanup'\'' to clean up if needed."; exit 130' INT

# Run main function
main "$@"