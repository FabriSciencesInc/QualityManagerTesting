# # requirements/base.txt
# # Core dependencies that are common across all environments
# numpy==1.24.3
# pandas==2.0.0
# requests==2.31.0
# python-dotenv==1.0.0

# # requirements/development.txt
# # Development-specific dependencies
# -r base.txt
# pytest==7.3.1
# black==23.3.0
# flake8==6.0.0
# ipython==8.12.0
# jupyter==1.0.0

# # requirements/production.txt
# # Production-specific dependencies
# -r base.txt
# gunicorn==20.1.0
# sentry-sdk==1.25.1

# # requirements/staging.txt
# # Staging-specific dependencies
# -r production.txt
# pytest==7.3.1

# scripts/generate_requirements.py
import subprocess
import sys
from pathlib import Path

def generate_requirements():
    """Generate requirements.txt file based on the current environment."""
    # Get the environment from command line argument or default to 'development'
    env = sys.argv[1] if len(sys.argv) > 1 else 'development'
    
    # Create requirements directory if it doesn't exist
    Path('requirements').mkdir(exist_ok=True)
    
    # Generate requirements file using pip freeze
    result = subprocess.run(['python3-intel64', '-m', 'pip', 'freeze'], capture_output=True, text=True)
    
    # Write to appropriate requirements file
    with open(f'requirements/{env}.txt', 'w') as f:
        f.write(result.stdout)
    
    print(f"Requirements generated for {env} environment")

if __name__ == "__main__":
    generate_requirements()