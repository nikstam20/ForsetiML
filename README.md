# FοrsetiML

## Overview

FοrsetiML is a web application built with React, designed to enhance machine learning fairness. It provides users with suggested bias mitigation algorithms and fairness indexes. This tool leverages a knowledge graph stored in Neo4j to power its analysis and recommendations.

## Prerequisites

Before you begin the setup, ensure you have the following installed:
- Python 3.8 or higher
- pip and virtualenv
- Node.js and npm
- Neo4j

## Setup Instructions

### Neo4j Database Setup

First, you need to set up the Neo4j database:
- Load the provided `neo4j.dump` file into your Neo4j instance. This can be done using the Neo4j Admin tool as follows:
  `neo4j-admin load --from=neo4j.dump --database=neo4j --force`
- Ensure your Neo4j instance is configured with the following credentials and URI:
  `NEO4J_URI="bolt://localhost:7687"`
  `NEO4J_USER="neo4j"`
  `NEO4J_PASSWORD="password"`

### Backend Setup

Set up the Python environment and install the necessary dependencies:
- Navigate to the main folder of the application.
- Set up a virtual environment:
  `python -m venv venv`
- Activate the virtual environment:
  - On Windows:
    `.\venv\Scripts\activate`
  - On Unix or MacOS:
    `source venv/bin/activate`
- Install the required Python packages:
  `pip install -r requirements.txt`

### Frontend Setup

Set up the React application:
- Navigate to the `frontend` folder.
- Install the necessary npm packages:
  `npm install`

## Running the Application

To run FοrsetiML, follow these steps:

1. Start the backend server from the main folder:
   - Ensure your virtual environment is activated.
   - Run the command:
     `python ./API/app.py`
2. In a new terminal, navigate to the `frontend` folder and start the React application:
   - Run the command:
     `npm start`

The React application will be available at `http://localhost:3000`, and the backend API will interface with it on the specified routes.
