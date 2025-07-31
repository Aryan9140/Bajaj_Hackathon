HackRx 6.0 - Complete Project Implementation
🏗️ Final Project Structure
hackrx-intelligent-retrieval/
├── 📁 app/
│   ├── __init__.py
│   ├── main.py                      # ✅ FastAPI application with webhook endpoint
│   ├── 📁 api/
│   │   ├── __init__.py
│   │   └── endpoints.py             # ✅ /hackrx/run endpoint implementation
│   ├── 📁 core/
│   │   ├── __init__.py
│   │   ├── config.py                # ✅ Environment configuration
│   │   ├── security.py              # ✅ API key authentication
│   │   └── response_models.py       # ✅ Pydantic models
│   ├── 📁 services/
│   │   ├── __init__.py
│   │   ├── document_processor.py    # ✅ PDF processing & chunking
│   │   └── clause_service.py     
│   │   └── document_Analizer.py       # ✅ implementation
         ---explainable_ai.py
         ---system_architecture.py
│   │   ├── vector_store.py          # ✅ AstraDB integration
│   │   ├── llm_handler.py           # ✅ Groq/Claude LLM calls
│   │   ├── embedding_service.py     # ✅ HuggingFace embeddings
│   │   └── query_processor.py       # ✅ Main RAG pipeline
│   └── 📁 utils/
│       ├── __init__.py
│       ├── logger.py                # ✅ Structured logging
│       └── validation.py       
│       └── cache.py                 # ✅ Performance caching
├── 📁 logs/                         # ✅ Application logs
├── requirements.txt                 # ✅ Optimized dependencies
├── .env                             # ✅ Environment variables
├── run.py                           # ✅ Application entry point
└── README.md                        # ✅ Documentation

🎯 Key Features Implemented
⚡ Performance Optimized (Latency <5s)

Async Architecture: Non-blocking I/O throughout
Batch Processing: Multiple questions processed in parallel
Connection Pooling: Reuse database connections
Intelligent Caching: Avoid redundant processing
Optimized Chunking: 512-character chunks for speed

🎯 Accuracy Focused

Multi-stage Retrieval: Semantic search + ranking
Context Optimization: Intelligent text truncation
Dual LLM Support: openai + Groq primary + Claude fallback
Precise Extraction: PDF text processing with page tracking

💰 Token Efficient

Smart Prompting: Minimal token usage
Context Truncation: Remove redundant content
Caching Strategy: Avoid repeated LLM calls
Batch Embeddings: Process multiple texts together

🔧 Production Ready

Error Handling: Graceful failures with logging
Health Checks: System monitoring endpoints
Security: Bearer token authentication
Monitoring: Detailed request tracking


📊 Technology Stack
ComponentTechnologyPurposeBackendFastAPIHigh-performance async APIVector DBAstraDBScalable vector storageLLMGroq (Gemma2-9B)Fast inferenceFallback LLMClaudeHigh accuracy backupEmbeddingsHuggingFaceSemantic searchCachingIn-memoryResponse optimizationLoggingPython loggingDebugging & monitoring

🚀 Quick Start Guide
1. Setup (5 minutes)
bash# Clone and setup
git clone <your-repo>
cd hackrx-intelligent-retrieval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Configure (.env file)
envAPI_KEY=6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193
ASTRA_DB_API_ENDPOINT=https://your-db.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:your-token
GROQ_API_KEY=your-groq-key
3. Run Locally
bashpython run.py
# API available at: http://localhost:8000/hackrx/run
4. Test with Postman

URL: http://localhost:8000/hackrx/run
Method: POST
Headers: Authorization: Bearer 6d2683f8...
Body: Use the exact format from problem statement

5. Deploy & Submit
bash# Deploy to Railway/Render/Vercel
# Submit webhook URL to HackRx platform
# https://your-app.com/hackrx/run

🎯 Evaluation Criteria Alignment
✅ Accuracy (Precision of query understanding)

Semantic search with cosine similarity
Multi-chunk retrieval for comprehensive context
Precise PDF text extraction with page tracking
Dual LLM system for reliability

✅ Token Efficiency (Optimized LLM usage)

Context truncation to optimal length
Minimal prompt engineering
Batch processing for embeddings
Response caching to avoid redundant calls

✅ Latency (Response speed <5s target)

Async processing throughout pipeline
Connection pooling for database operations
Parallel question processing
Optimized chunk sizes for fast retrieval

✅ Reusability (Code modularity)

Service-oriented architecture
Configurable parameters via environment
Clean separation of concerns
Easy to extend and modify

✅ Explainability (Clear decision reasoning)

Detailed logging at each step
Source attribution in responses
Clear error messages
Processing time tracking


📈 Performance Metrics
Target Benchmarks

⏱️ Response Time: <5 seconds (target: 2-3s)
🎯 Accuracy: >90% on policy questions
💰 Token Usage: <2000 tokens per request
📊 Throughput: >10 requests/minute
🔄 Cache Hit Rate: >50% for repeated queries

Optimization Features

Parallel document processing
Batch embedding generation
Intelligent context ranking
Automatic error recovery
Resource cleanup


🔧 API Specification
Endpoint: /hackrx/run
Method: POST
Authentication: Bearer 6d2683f8...
Request Format:
json{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
Response Format:
json{
    "answers": [
        "A grace period of thirty days is provided...",
        "There is a waiting period of thirty-six months..."
    ],
    "processing_time": 2.45,
    "request_id": "req_1690123456789",
    "cached": false
}

🎉 Success Checklist
Before Submission:

 ✅ API responds to /hackrx/run with correct format
 ✅ HTTPS deployment working
 ✅ Bearer token authentication functional
 ✅ Sample policy questions answered accurately
 ✅ Response times under 10 seconds
 ✅ Error handling graceful
 ✅ Logs showing successful processing
 ✅ All environment variables configured
 ✅ Health checks passing

Deployment Options:

Railway (Recommended) - Easy Python deployment
Render - Free tier with auto-scaling
Vercel - Serverless functions
Heroku - Traditional platform
Docker - Containerized deployment


🏆 Competitive Advantages
Speed Optimizations

Async architecture throughout
Parallel processing pipeline
Intelligent caching layer
Optimized vector search

Accuracy Features

Multi-stage document retrieval
Context ranking algorithms
Dual LLM fallback system
Precise text extraction

Reliability Measures

Comprehensive error handling
Graceful degradation
Health monitoring
Resource management

Scalability Design

Stateless architecture
Connection pooling
Horizontal scaling ready
Resource optimization


🚀 Your HackRx 6.0 solution is now complete and optimized for winning performance!
Submit your webhook URL and compete for the top leaderboard position! 🏆









HackRx 6.0 - Testing & Deployment Guide
📋 Quick Setup Checklist
1. Environment Setup
bash# Clone/create project structure
mkdir hackrx-intelligent-retrieval
cd hackrx-intelligent-retrieval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
2. Environment Variables (.env)
bash# Required Configuration
API_KEY=6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193
ASTRA_DB_API_ENDPOINT=https://your-db-id-region.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:your-token-here
ASTRA_DB_KEYSPACE=hackrx
ASTRA_DB_COLLECTION=document_embeddings
GROQ_API_KEY=your-groq-api-key-here

# Optional
CLAUDE_API_KEY=your-claude-api-key-here
DEBUG=true
HOST=0.0.0.0
PORT=8000

🔧 Local Development Testing
1. Start the Application
bash# Method 1: Direct run
python run.py

# Method 2: Using uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Using gunicorn (production-like)
gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
2. Verify API is Running
bash# Check health endpoint
curl http://localhost:8000/health

# Expected Response:
{
  "status": "healthy",
  "message": "HackRx Intelligent Retrieval System is running",
  "services": {
    "vector_store": true,
    "embedding": true
  }
}

📨 Postman Testing
1. Local Testing Setup
URL: http://localhost:8000/hackrx/run
Method: POST
Headers:
json{
  "Content-Type": "application/json",
  "Accept": "application/json",
  "Authorization": "Bearer 6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193"
}
Request Body:
json{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
2. Expected Response Format
json{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date...",
        "There is a waiting period of thirty-six (36) months of continuous coverage...",
        "Yes, the policy covers maternity expenses, including childbirth..."
    ],
    "processing_time": 2.45,
    "request_id": "req_1690123456789",
    "cached": false
}
3. cURL Testing Command
bashcurl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer 6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
  }'

🚀 Deployment Options
1. Railway (Recommended for Hackathon)
bash# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Set environment variables in Railway dashboard
2. Render
bash# Connect GitHub repo to Render
# Build Command: pip install -r requirements.txt
# Start Command: python run.py
3. Vercel (Serverless)
bash# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
4. Heroku
bash# Create Procfile
echo "web: python run.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
5. Docker Deployment
dockerfile# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run.py"]

🌐 Production Testing (After Deployment)
URL: https://your-deployed-app.com/hackrx/run
Method: POST
Headers: Same as local testing
Body: Same as local testing
cURL Command for Production:
bashcurl -X POST "https://your-deployed-app.com/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer 6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
  }'

⚡ Performance Optimization Tips
1. Latency Optimization (<5s target)

✅ Async Processing: All I/O operations are async
✅ Batch Embeddings: Process multiple questions together
✅ Connection Pooling: Reuse database connections
✅ Caching: Cache document embeddings and responses
✅ Optimized Chunking: Smaller chunks (512 chars) for faster processing

2. Token Efficiency

✅ Context Optimization: Truncate long contexts intelligently
✅ Prompt Engineering: Minimal, focused prompts
✅ Model Selection: Use Groq (fast) with Claude fallback
✅ Batch Processing: Process multiple questions in one call

3. Accuracy Improvements

✅ Semantic Search: Use cosine similarity with normalized embeddings
✅ Multi-stage Retrieval: Retrieve more chunks, rank by relevance
✅ Context Ranking: Keep most relevant chunks for LLM
✅ Query Enhancement: Clean and optimize question text


📊 Monitoring & Debugging
1. Check Logs
bash# View logs
tail -f logs/hackrx.log

# Check for errors
grep "ERROR" logs/hackrx.log
2. Performance Metrics
bash# Get API status
curl http://localhost:8000/status

# Cache statistics
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/api/v1/status
3. Health Checks
bash# Basic health
curl http://localhost:8000/health

# Detailed diagnostics
curl http://localhost:8000/

🎯 HackRx Submission Process
1. Prepare Your Webhook URL
After deployment, your webhook URL will be:
https://your-app-domain.com/hackrx/run
2. Submit to HackRx Platform

Go to: HackRx Dashboard → Submissions
Enter your webhook URL: https://your-app-domain.com/hackrx/run
Add description: FastAPI + AstraDB + Groq LLM + HuggingFace Embeddings
Click "Run" to start evaluation

3. Pre-Submission Checklist

 ✅ API responds to /hackrx/run endpoint
 ✅ HTTPS enabled (required for submission)
 ✅ Bearer token authentication working
 ✅ Returns JSON with "answers" array
 ✅ Response time < 30 seconds
 ✅ Handles the exact request format from problem statement
 ✅ All environment variables configured
 ✅ AstraDB connection working
 ✅ Groq API key valid


🔧 Troubleshooting Guide
Common Issues & Solutions
1. "AstraDB connection failed"
bash# Check environment variables
echo $ASTRA_DB_API_ENDPOINT
echo $ASTRA_DB_APPLICATION_TOKEN

# Verify AstraDB credentials in dashboard
2. "Groq API error"
bash# Test Groq API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     https://api.groq.com/openai/v1/models
3. "No embeddings generated"
bash# Check if sentence-transformers downloaded
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
4. "Document processing failed"
bash# Test document URL access
curl -I "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=..."
5. "Response timeout"

Reduce MAX_RETRIEVE_DOCS to 5-8
Optimize chunk size to 256-512
Enable caching for repeated queries


📈 Scoring Optimization
Focus Areas for High Scores
1. Accuracy (40% weight)

Extract exact text from policy documents
Match questions to specific clauses
Provide precise answers with correct details

2. Token Efficiency (20% weight)

Use context truncation intelligently
Minimize prompt length while maintaining accuracy
Cache embeddings to avoid recomputation

3. Latency (20% weight)

Target <5 seconds total response time
Use async processing throughout
Optimize vector search parameters

4. Reusability (10% weight)

Modular service architecture
Clean separation of concerns
Configurable parameters

5. Explainability (10% weight)

Log processing steps
Provide source attribution
Clear error messages


📝 Final Deployment Command
bash# 1. Setup environment
cp .env.example .env
# Edit .env with your credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test locally
python run.py

# 4. Deploy to Railway/Render/Vercel
# Follow platform-specific instructions above

# 5. Submit webhook URL to HackRx platform
# https://your-app.com/hackrx/run

🎉 Success Indicators
Your API is ready when:

✅ Health check returns 200 OK
✅ Sample request returns structured answers
✅ Response time < 10 seconds (target: <5s)
✅ HTTPS certificate valid
✅ Bearer token authentication working
✅ Logs show successful document processing
✅ Vector search returning relevant chunks
✅ LLM generating accurate answers

Good luck with your HackRx 6.0 submission! 🚀