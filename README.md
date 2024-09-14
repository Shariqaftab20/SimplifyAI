# AI-Powered Customer Support System

Welcome to the AI-Powered Customer Support System repository! This project demonstrates the integration of advanced NLP techniques with a Django backend to create a sophisticated customer support system. Leveraging LangChain, a vector database for RAG (Retrieval-Augmented Generation), and various AI models, this system provides intelligent and context-aware responses to customer queries.

## Overview

This application uses LangChain and FAISS or Chroma DB to set up a question-answering system with Retrieval-Augmented Generation (RAG). The backend is built with Django, and the application is designed to handle customer queries efficiently, providing accurate responses based on a pre-defined knowledge base.

## Features

- **AI-Driven Responses**: Utilizes LangChain and advanced NLP techniques to generate accurate answers to customer queries.
- **Knowledge Base Integration**: Implements a vector store using FAISS or Chroma DB to retrieve relevant information.
- **Django API**: Provides a RESTful API endpoint to handle customer queries and return AI-generated responses.
- **Error Handling and Logging**: Includes basic error handling and logging to ensure reliable operation.

## Getting Started

Follow these steps to set up and run the application locally:

### Prerequisites

- **Python 3.10+**: Ensure Python is installed on your machine.
- **Docker**: (Optional) For containerized deployment.

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/ai-customer-support-system.git
   cd ai-customer-support-system
   
2. **Set Up a Python Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt

4. **Set Up Environment Variables**

   Create a .env file in the root directory and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key


5. **Run Migrations**

   ```bash
   python manage.py migrate

6. **Start the Django Development Server**

   ```bash
   python manage.py runserver
  The API will be available at http://127.0.0.1:8000/api/query/.

### Docker Deployment

  To run the application in a Docker container, follow these steps:
  
1. **Build the Docker Image**

   ```bash
   docker build -t ai-customer-support-system.
   
2. **Run the Docker Container**

   ```bash
   docker run -d -p 8000:8000 --name ai-customer-support-system ai-customer-support-system
   
## API Endpoints

### `POST /api/query/`

- **Description**: Submits a customer query and retrieves an AI-generated response.

- **Request Body**:

  ```json
  {
    "query": "Your customer query here."
  }
- **Response Body**:

  ```json
  {
  "response": "AI-generated response to the query."
  }




