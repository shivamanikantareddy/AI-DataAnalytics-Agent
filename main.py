from fastapi import FastAPI,UploadFile,File,HTTPException
# from langchain_core.messages import HumanMessage,AIMessage
from fastapi.responses import JSONResponse
from typing import List
# from embeddings.vectorstore import split_store_documents,namespace_deletion
# from generation.response import response_generation
# from embeddings.process_and_load import load_document
# from schema.schema_models import ChatRequest
# from generation.response import chat_history,response_generation
from pathlib import Path
import pandas as pd
from graph import run_agent

app = FastAPI(title="AI-assisted Data Analysis and Visualisation System API")

MODEL_VERSION="1.0.0"

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def home():
    return JSONResponse(status_code=200, content={"message": "Welcome to the Analytics and Visualisation Assistant!","version": MODEL_VERSION,})

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded",
        "vector_store_status": "connected",
        "version": MODEL_VERSION
    }
    
@app.post("/load_dataset")
async def knowledge_assistant(file : UploadFile = File(..., description="Dataset file to be processed and loaded into the knowledge base. Supported formats: CSV, Excel, JSON.")):

        MAX_SIZE= 500 * 1024 ** 2
        
        file_size = file.size
        
        content_type = file.content_type
        
        file_headers=file.headers
        print(f"Received file: {file.filename}, Size: {file_size} bytes, Content-Type: {content_type}, Headers: {file_headers}")
        
        if (UPLOAD_DIR / file.filename).exists():
            raise HTTPException(status_code=400, detail=f"File {file.filename} already exists!")
        
        file_path = UPLOAD_DIR / file.filename
        print(file_path)
        if file_size > MAX_SIZE:
            raise ValueError("File too large")
        
        async def process_file(file,file_path):
            with open (file_path,"wb") as out_file:
                
                out_file.write(await file.read())
                print ('processfile')
                    
                    
            return run_agent(file_path)


        try:
            
            report = await process_file(file,file_path)
        
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail="Failed to process and load the file.")
    
        return JSONResponse(status_code=200, content={"message": f"{file.filename} stored and profiled successfully. Report : {report}"})
    

# @app.post('/chat_assistant')
# async def chat_interface(request: ChatRequest):
#     """
#     Endpoint for chatting with the knowledge assistant.
#     Takes a user query and returns a response based on the loaded documents.
#     """
#     query=request.query.strip()
    
#     if not query:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Query cannot be empty"}
#         )
    
#     try:
        
#         if query.lower() in {"quit", "exit", "end"}:
#             await namespace_deletion()
#             if chat_history:
#                 chat_history.clear()
#             return JSONResponse(status_code=200,content={"message":"Conversation ended. Session cleared."})
        
#         chat_history.append(HumanMessage(content=query))
        
#         # Generate response using the query
#         response = await response_generation(query,chat_history)
        
#         chat_history.append(AIMessage(content=response))
        
#         print(chat_history)
        
#         # Return the response in a structured format
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "query": query,
#                 "response": response
#             }
#         )
        
#     except Exception as e:
#         # Handle any errors that occur during response generation
#         print(str(e))
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "error": "Internal server error",
#                 "query": query,
#                 "status": "Error",
#                 "message": "Failed to generate response. Please try again."
#             }
#         )
    