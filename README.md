# Verification Playground

![Verification Playground](https://img.shields.io/badge/Hardware-Verification-blue)
![Django](https://img.shields.io/badge/Django-5.1.4-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)

## Overview

Verification Playground is a Django-based web application that provides essential tools for hardware verification engineers:

1. **UVM RAL Generator**: Automatically creates Universal Verification Methodology (UVM) Register Abstraction Layer models from Excel specifications
2. **System Block Diagram Generator**: Creates visual representations of hardware systems with customizable inputs and outputs

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [AWS EC2 Deployment](#aws-ec2-deployment)
- [Usage](#usage)
  - [UVM RAL Generator](#uvm-ral-generator)
  - [System Block Diagram Generator](#system-block-diagram-generator)
- [API Documentation](#api-documentation)
- [API Testing](#api-testing)
- [Excel File Format for UVM RAL Generation](#excel-file-format-for-uvm-ral-generation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

### UVM RAL Generator

The UVM RAL Generator automates the creation of Universal Verification Methodology (UVM) Register Abstraction Layer models from Excel specifications. This tool:

- Parses Excel files containing register definitions
- Extracts register names, fields, access types, and other properties
- Generates SystemVerilog (.sv) files with UVM register models
- Supports field bit ranges (e.g., `BUFFER_SIZE [15:0]`)
- Creates a complete register block with proper UVM structure

### System Block Diagram Generator

The System Block Diagram Generator creates visual representations of hardware systems with:

- Customizable number of inputs and outputs
- Automatically adjusted block dimensions based on I/O count
- Clean, professional-looking diagrams with rounded corners
- Proper spacing and alignment of components
- PNG output format for easy integration into documentation

## Technology Stack

- **Backend**: Django 5.1.4, Django REST Framework 3.15.2
- **Data Processing**: Pandas 2.2.3, NumPy 2.2.1
- **Excel Handling**: OpenPyXL 3.1.5
- **Visualization**: Matplotlib 3.10.1
- **Frontend**: HTML, CSS
- **Development**: Python 3.12

## Project Structure

```
verif_play_ground/
├── manage.py                  # Django management script
├── requirements.txt           # Project dependencies
├── run.sh                     # Convenience script to run the server
├── README.md                  # Project documentation
├── verif_play_ground/         # Main Django project
│   ├── __init__.py
│   ├── asgi.py                # ASGI configuration
│   ├── settings.py            # Project settings
│   ├── urls.py                # URL routing
│   ├── wsgi.py                # WSGI configuration
│   ├── static/                # Static files (CSS, JS, etc.)
│   │   └── css/
│   │       └── style.css      # Main stylesheet
│   ├── templates/             # HTML templates
│   │   └── index.html         # Home page template
│   └── media/                 # User-uploaded files
└── verif_play_ground_app/     # Django application
    ├── __init__.py
    ├── admin.py               # Admin interface configuration
    ├── apps.py                # App configuration
    ├── migrations/            # Database migrations
    ├── models.py              # Data models
    ├── tests.py               # Unit tests
    ├── utils.py               # Utility functions for UVM RAL generation
    └── views.py               # API endpoints and view functions
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/verif_play_ground.git
   cd verif_play_ground
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run database migrations:
   ```bash
   python manage.py migrate
   ```

6. Start the development server:
   ```bash
   python manage.py runserver
   ```
   Or use the convenience script:
   ```bash
   ./run.sh
   ```

7. Access the application at http://127.0.0.1:8000/

## AWS EC2 Deployment Guide

This section provides step-by-step instructions for deploying the application on AWS EC2.

### Step 1: Launch an EC2 Instance

1. **Log in to AWS Console**
   - Go to https://aws.amazon.com/console/

2. **Create a new EC2 instance**
   - Navigate to EC2 → Launch Instance
   - Name your instance (e.g., "verification-playground")

3. **Select an Amazon Machine Image (AMI)**
   - Recommended: Amazon Linux 2023 or Ubuntu Server 22.04 LTS

4. **Choose Instance Type**
   - For testing: t2.micro (free tier eligible)
   - For production: t2.small or better

5. **Configure Security Group**
   - Allow SSH (port 22) from your IP
   - Allow HTTP (port 80) from anywhere
   - Allow HTTPS (port 443) from anywhere
   - Allow Custom TCP (port 8000) from anywhere (for development)

6. **Launch the instance**
   - Create or select an existing key pair
   - Download the key pair file (.pem)

### Step 2: Connect to Your EC2 Instance

**For Amazon Linux:**
```bash
chmod 400 /path/to/your-key-pair.pem
ssh -i /path/to/your-key-pair.pem ec2-user@your-instance-public-dns
```

**For Ubuntu:**
```bash
chmod 400 /path/to/your-key-pair.pem
ssh -i /path/to/your-key-pair.pem ubuntu@your-instance-public-dns
```

### Step 3: Install Required Packages

#### For Amazon Linux 2023:

```bash
# Update system packages
sudo dnf update -y

# Install Python and development tools
sudo dnf install -y python3 python3-pip python3-devel git

# Install additional dependencies for matplotlib
sudo dnf install -y gcc freetype-devel libpng-devel
```

#### For Ubuntu:

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install Python and development tools
sudo apt install -y python3 python3-pip python3-venv python3-dev git

# Install additional dependencies for matplotlib
sudo apt install -y build-essential libfreetype6-dev libpng-dev
```

### Step 4: Clone and Set Up the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/verif_play_ground.git
cd verif_play_ground

# Create and activate virtual environment
sudo python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
sudo pip3 install -r requirements.txt

# Run migrations
sudo python3 manage.py migrate

# Collect static files
sudo python3 manage.py collectstatic --noinput
```

### Step 5: Run the Application

#### Option A: Development Server (Quick Start)

```bash
# Run the development server on all interfaces
sudo python3 manage.py runserver 0.0.0.0:8000
```

Access your application at `http://your-instance-public-dns:8000`

#### Option B: Production Deployment with Gunicorn and Nginx

1. **Install Gunicorn and Nginx**

   ```bash
   # Inside your virtual environment
   sudo pip3 install gunicorn

   # Install Nginx
   # For Amazon Linux:
   sudo dnf install -y nginx
   # For Ubuntu:
   sudo apt install -y nginx
   ```

2. **Create a systemd service file for Gunicorn**

   ```bash
   sudo nano /etc/systemd/system/verif_play_ground.service
   ```

   Add the following content (adjust paths as needed):

   ```ini
   [Unit]
   Description=Gunicorn daemon for Verification Playground
   After=network.target

   [Service]
   User=ec2-user  # Use 'ubuntu' for Ubuntu
   Group=ec2-user  # Use 'ubuntu' for Ubuntu
   WorkingDirectory=/home/ec2-user/verif_play_ground
   ExecStart=/home/ec2-user/verif_play_ground/.venv/bin/gunicorn \
             --access-logfile - \
             --workers 3 \
             --bind unix:/home/ec2-user/verif_play_ground/verif_play_ground.sock \
             verif_play_ground.wsgi:application

   [Install]
   WantedBy=multi-user.target
   ```

3. **Configure Nginx**

   ```bash
   sudo nano /etc/nginx/sites-available/verif_play_ground
   ```

   Add the following content (adjust paths as needed):

   ```nginx
   server {
       listen 80;
       server_name your-instance-public-dns;

       location = /favicon.ico { access_log off; log_not_found off; }

       location /static/ {
           root /home/ec2-user/verif_play_ground;
       }

       location /media/ {
           root /home/ec2-user/verif_play_ground;
       }

       location / {
           include proxy_params;
           proxy_pass http://unix:/home/ec2-user/verif_play_ground/verif_play_ground.sock;
       }
   }
   ```

4. **Create a symbolic link and start services**

   ```bash
   # For Ubuntu:
   sudo ln -s /etc/nginx/sites-available/verif_play_ground /etc/nginx/sites-enabled

   # For Amazon Linux (create the directory if it doesn't exist):
   sudo mkdir -p /etc/nginx/sites-enabled
   sudo ln -s /etc/nginx/sites-available/verif_play_ground /etc/nginx/sites-enabled

   # Update Nginx configuration to include sites-enabled
   sudo nano /etc/nginx/nginx.conf
   # Add this line inside the http block: include /etc/nginx/sites-enabled/*;

   # Start and enable services
   sudo systemctl start verif_play_ground
   sudo systemctl enable verif_play_ground
   sudo systemctl restart nginx
   sudo systemctl enable nginx
   ```

5. **Configure firewall (if needed)**

   ```bash
   # For Ubuntu:
   sudo ufw allow 'Nginx Full'
   sudo ufw allow ssh
   sudo ufw enable
   ```

Access your application at `http://your-instance-public-dns`

## Usage

### UVM RAL Generator

#### Using the Web Interface

1. Navigate to http://127.0.0.1:8000/ in your browser
2. Click on the UVM RAL Generator endpoint
3. Upload an Excel file with register definitions (see [Excel File Format](#excel-file-format-for-uvm-ral-generation))
4. Submit the form to generate and download the UVM RAL model (.sv file)

#### Using the API Directly

```bash
# Generate and download a .sv file
curl -X POST -F "file=@your_registers.xlsx" http://127.0.0.1:8000/generate-uvm-ral/ --output uvm_ral_model.sv

# Generate and get Base64 encoded content
curl -X POST -F "file=@your_registers.xlsx" http://127.0.0.1:8000/generate-uvm-ral-base/
```

### System Block Diagram Generator

#### Using the Web Interface

1. Navigate to http://127.0.0.1:8000/ in your browser
2. Click on the System Block Diagram Generator endpoint
3. Enter the number of inputs and outputs
4. Submit the form to generate and download the diagram (PNG file)

#### Using the API Directly

```bash
curl -X POST -H "Content-Type: application/json" -d '{"input_count": 3, "output_count": 2}' http://127.0.0.1:8000/drawSystemBlockAPIView/ --output diagram.png
```

## API Documentation

### UVM RAL Generator

#### `POST /generate-uvm-ral/`

Generates a UVM RAL model from an Excel file and returns it as a downloadable .sv file.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Excel file with register definitions

**Response:**
- Content-Type: `application/octet-stream`
- Body: SystemVerilog (.sv) file

**Error Responses:**
- 400 Bad Request: No file provided
- 500 Internal Server Error: Error generating .sv file

#### `POST /generate-uvm-ral-base/`

Generates a UVM RAL model from an Excel file and returns it as Base64 encoded content.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Excel file with register definitions

**Response:**
- Content-Type: `application/json`
- Body:
  ```json
  {
    "file": "<base64_encoded_content>"
  }
  ```

**Error Responses:**
- 400 Bad Request: No file provided
- 500 Internal Server Error: Error generating .sv file or encoding to Base64

### System Block Diagram Generator

#### `POST /drawSystemBlockAPIView/`

Generates a system block diagram based on input and output counts.

**Request:**
- Content-Type: `application/json`
- Body:
  ```json
  {
    "input_count": 3,
    "output_count": 2
  }
  ```

**Response:**
- Content-Type: `image/png`
- Body: PNG image file

**Error Responses:**
- 400 Bad Request: Missing or invalid input_count or output_count

## Excel File Format for UVM RAL Generation

The UVM RAL Generator expects an Excel file with the following columns:

| Column Name    | Description                                                  | Example        |
|----------------|--------------------------------------------------------------|----------------|
| Register Name  | Name of the register                                         | CTRL_REG       |
| Offset         | Register address offset                                      | 0x00           |
| Read/Write     | Access type (RW, RO, WO, etc.)                               | RW             |
| Fields         | Field name with bit range                                    | ENABLE [0:0]   |
| Default value  | Default/reset value for the field                            | 0              |
| Reset value    | Alternative reset value (if different from default)          | 0              |
| Description    | Description of the register or field                         | Control Register |

**Notes:**
- Field bit ranges should be specified in the format `NAME [MSB:LSB]`
- Each row can define a new register or add fields to the current register
- For rows adding fields to a register, leave the Register Name column empty

**Example Excel Format:**

| Register Name | Offset | Read/Write | Fields           | Default value | Reset value | Description       |
|---------------|--------|------------|------------------|---------------|-------------|-------------------|
| CTRL_REG      | 0x00   | RW         | ENABLE [0:0]     | 0             | 0           | Control Register  |
|               |        |            | MODE [2:1]       | 0             | 0           | Operating Mode    |
|               |        |            | INTR_EN [3:3]    | 0             | 0           | Interrupt Enable  |
| STATUS_REG    | 0x04   | RO         | BUSY [0:0]       | 0             | 0           | Busy Status       |
|               |        |            | ERROR [1:1]      | 0             | 0           | Error Status      |
| DATA_REG      | 0x08   | RW         | DATA [31:0]      | 0             | 0           | Data Register     |

## API Testing Guide

Follow these steps to test the APIs on your EC2 instance or local environment.

### Step 1: Create a Test Excel File

1. **Create a Python script** named `create_test_excel.py`:

   ```bash
   # Create the script file
   nano create_test_excel.py
   ```

2. **Add the following code** to the script:

   ```python
   import openpyxl
   from openpyxl import Workbook

   # Create a new workbook
   wb = Workbook()
   ws = wb.active

   # Add headers
   headers = ['Register Name', 'Offset', 'Read/Write', 'Fields', 'Default value', 'Reset value', 'Description']
   for col_num, header in enumerate(headers, 1):
       ws.cell(row=1, column=col_num, value=header)

   # Add data
   data = [
       ['CTRL_REG', '0x00', 'RW', 'ENABLE [0:0]', 0, 0, 'Control Register'],
       ['', '', '', 'MODE [2:1]', 0, 0, 'Operating Mode'],
       ['', '', '', 'INTR_EN [3:3]', 0, 0, 'Interrupt Enable'],
       ['STATUS_REG', '0x04', 'RO', 'BUSY [0:0]', 0, 0, 'Busy Status'],
       ['', '', '', 'ERROR [1:1]', 0, 0, 'Error Status'],
       ['DATA_REG', '0x08', 'RW', 'DATA [31:0]', 0, 0, 'Data Register']
   ]

   for row_num, row_data in enumerate(data, 2):
       for col_num, cell_value in enumerate(row_data, 1):
           ws.cell(row=row_num, column=col_num, value=cell_value)

   # Save the workbook
   wb.save('test_registers.xlsx')
   print("Test Excel file created: test_registers.xlsx")
   ```

3. **Run the script** to create the test file:

   ```bash
   # Make sure openpyxl is installed
   sudo pip3 install openpyxl

   # Run the script
   python3 create_test_excel.py
   ```

### Step 2: Test the UVM RAL Generator API

1. **Generate a UVM RAL model** (download as .sv file):

   ```bash
   curl -X POST -F "file=@test_registers.xlsx" http://127.0.0.1:8000/generate-uvm-ral/ -o uvm_ral_model.sv
   ```

2. **Check the generated file**:

   ```bash
   head -20 uvm_ral_model.sv
   ```

3. **Test the Base64 version** of the UVM RAL Generator API:

   ```bash
   curl -X POST -F "file=@test_registers.xlsx" http://127.0.0.1:8000/generate-uvm-ral-base/ | head -20
   ```

### Step 3: Test the System Block Diagram Generator API

1. **Generate a system block diagram**:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"input_count": 3, "output_count": 2}' http://127.0.0.1:8000/drawSystemBlockAPIView/ -o system_block_diagram.png
   ```

2. **Verify the PNG file was created**:

   ```bash
   ls -la system_block_diagram.png
   ```

### Step 4: Test Error Handling

1. **Test UVM RAL Generator API without a file** (should return an error):

   ```bash
   curl -X POST http://127.0.0.1:8000/generate-uvm-ral/
   ```

2. **Test System Block Diagram Generator API with invalid parameters** (should return an error):

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"output_count": "invalid"}' http://127.0.0.1:8000/drawSystemBlockAPIView/ -v
   ```

### Testing on EC2

When testing on EC2, replace `127.0.0.1:8000` with your EC2 instance's public DNS or IP address:

```bash
# Example for EC2
curl -X POST -F "file=@test_registers.xlsx" http://your-ec2-public-dns:8000/generate-uvm-ral/ -o uvm_ral_model.sv
```

## Development

### Setting Up a Development Environment

1. Follow the installation steps above
2. Make your changes to the codebase
3. Run tests (when available): `python manage.py test`

### Project Structure Details

- **utils.py**: Contains the core functionality for parsing Excel files and generating UVM RAL models
- **views.py**: Implements the API endpoints and handles file uploads/downloads
- **settings.py**: Django configuration including static files, templates, and media storage

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Port Already in Use

If you see an error like "That port is already in use" when starting the server:

```bash
# Find the process using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or start the server on a different port
sudo python3 manage.py runserver 0.0.0.0:8001
```

#### Issue 2: Missing Dependencies

If you encounter errors related to missing dependencies:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Reinstall all dependencies
sudo pip3 install -r requirements.txt

# If a specific package is missing
sudo pip3 install <package_name>
```

#### Issue 3: Permission Issues on AWS EC2

If you encounter permission issues on AWS EC2:

```bash
# Check ownership of files
ls -la

# Change ownership if needed
sudo chown -R ec2-user:ec2-user /home/ec2-user/verif_play_ground

# Check permissions for the socket file
sudo chmod 664 /home/ec2-user/verif_play_ground/verif_play_ground.sock

# Make sure the .venv directory has correct permissions
sudo chmod -R 755 .venv
```

#### Issue 4: Static Files Not Loading

If static files are not loading in production:

```bash
# Make sure you've collected static files
sudo python3 manage.py collectstatic --noinput

# Check Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Check permissions on static files directory
sudo chmod -R 755 staticfiles/
```

#### Issue 5: Database Errors

If you encounter database errors:

```bash
# Reset migrations (if needed)
rm -rf verif_play_ground_app/migrations/
mkdir verif_play_ground_app/migrations/
touch verif_play_ground_app/migrations/__init__.py

# Recreate database
sudo python3 manage.py makemigrations
sudo python3 manage.py migrate
```

### Checking Logs for Debugging

```bash
# Check Django development server logs
# They appear in the terminal where you run the server

# Check Gunicorn logs
sudo journalctl -u verif_play_ground

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# Check system logs for other issues
sudo tail -f /var/log/syslog  # Ubuntu
sudo tail -f /var/log/messages  # Amazon Linux
```

### Getting Help

If you continue to experience issues:

1. Check the Django documentation: https://docs.djangoproject.com/
2. Search for specific error messages online
3. Open an issue on the GitHub repository with detailed information about your problem

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Conclusion

The Verification Playground provides a powerful set of tools for hardware verification engineers. With its UVM RAL Generator and System Block Diagram Generator, it simplifies common verification tasks and improves productivity. The application can be deployed locally for development or on AWS EC2 for production use.

For questions, issues, or feature requests, please open an issue on the GitHub repository.

---

Developed with ❤️ for hardware verification engineers
"# verifPlay_Django" 
"# verifPlay_Django" 
#   V e r i f P l a y _ A I  
 