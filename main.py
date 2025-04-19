from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

# main.py - FastAPI Backend for Process Mining with OpenAI O3

import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import openai
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Process Mining API with OpenAI O3")

# Configure CORS - updated to include the Lovable.dev URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://future-proof-workforce-insights.lovable.app",  # Production Lovable.dev app
        "http://localhost:3000",  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query_type: str  # "process_mining", "knowledge_graph", or "causal_graph"
    filters: Optional[Dict[str, Any]] = None

class ReasoningRequest(BaseModel):
    data: Dict[str, Any]
    question: str

# Helper function to fetch data from Supabase
def fetch_data_from_supabase(table_name: str, filters: Optional[Dict[str, Any]] = None):
    query = supabase.table(table_name).select("*")
    
    if filters:
        for key, value in filters.items():
            # Handle special filter types
            if key == "date_range":
                if value == "last_30":
                    query = query.gte("timestamp", "now() - interval '30 days'")
                elif value == "last_90":
                    query = query.gte("timestamp", "now() - interval '90 days'")
                elif value == "last_365":
                    query = query.gte("timestamp", "now() - interval '365 days'")
            elif key == "risk_level":
                if value == "high":
                    query = query.gte("automation_probability", 0.66)
                elif value == "medium":
                    query = query.gte("automation_probability", 0.33).lt("automation_probability", 0.66)
                elif value == "low":
                    query = query.lt("automation_probability", 0.33)
            elif key == "outcome":
                if value == "success":
                    query = query.eq("certification_earned", True)
                elif value == "failure":
                    query = query.eq("certification_earned", False)
            elif key == "skill_category":
                # query = query.eq("training_program", value)
                query = query.eq("skill_category", value)
                print(f"Filtering by skill_category: {value}")
            else:
                query = query.eq(key, value)
    
    result = query.execute()
    print(f"Supabase query result: {result}")
    if result.error:
        raise HTTPException(status_code=500, detail=f"Supabase error: {result.error}")
    if result.status_code != 200:
        raise HTTPException(status_code=result.status_code, detail=f"Supabase error: {result.error}")
    if result.data:
        return result.data
    else:
        return []

# Function to convert SQL results to formatted JSON for OpenAI
def format_data_for_openai(data: List[Dict[str, Any]], query_type: str) -> Dict[str, Any]:
    if query_type == "process_mining":
        # Format specifically for process mining analysis
        events_df = pd.DataFrame(data)
        print(f"Events DataFrame: {events_df.head()}")
        # Ensure timestamp is properly formatted
        if 'timestamp' in events_df.columns:
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            events_df = events_df.sort_values('timestamp')
        print(f"Sorted Events DataFrame: {events_df.head()}")
        # Group by case_id to create process instances
        process_instances = {}
        for case_id, group in events_df.groupby('case_id'):
            process_instances[str(case_id)] = group.to_dict(orient='records')
        
        # Create a summary of the process instances
        process_summary = {
            "process_instances": process_instances,
            "event_count": len(events_df),
            "case_count": len(process_instances)
        }
        print(f"Process Summary: {process_summary}")
        return process_summary
        
    
    elif query_type == "knowledge_graph":
        # Format for knowledge graph analysis
        entities = {}
        relationships = []
        
        # Extract entities and relationships
        for record in data:
            if 'employee_id' in record:
                entity_id = f"employee_{record['employee_id']}"
                if entity_id not in entities:
                    entities[entity_id] = {"type": "employee", **record}
            
            if 'case_id' in record:
                case_id = f"case_{record['case_id']}"
                if case_id not in entities:
                    entities[case_id] = {"type": "case", **record}
                
                if 'employee_id' in record:
                    relationships.append({
                        "source": f"employee_{record['employee_id']}",
                        "target": case_id,
                        "type": "participates_in"
                    })
        
        return {
            "knowledge_graph": {
                "entities": list(entities.values()),
                "relationships": relationships
            }
        }
    
    elif query_type == "causal_graph":
        # Format for causal analysis
        df = pd.DataFrame(data)
        
        # Calculate potential causal factors
        causal_factors = {}
        
        if 'certification_earned' in df.columns and 'training_program' in df.columns:
            program_success = df.groupby('training_program')['certification_earned'].mean().to_dict()
            causal_factors['program_success_rate'] = program_success
        
        if 'automation_probability' in df.columns and 'certification_earned' in df.columns:
            # Check correlation between automation risk and certification success
            risk_success = {}
            for risk_range in ['low', 'medium', 'high']:
                if risk_range == 'low':
                    subset = df[df['automation_probability'] <= 0.33]
                elif risk_range == 'medium':
                    subset = df[(df['automation_probability'] > 0.33) & (df['automation_probability'] <= 0.66)]
                else:
                    subset = df[df['automation_probability'] > 0.66]
                
                if not subset.empty:
                    risk_success[risk_range] = subset['certification_earned'].mean()
            
            causal_factors['risk_success_correlation'] = risk_success
        
        return {
            "causal_data": {
                "factors": causal_factors,
                "record_count": len(df)
            }
        }
    
    else:
        # Default formatting as simple JSON
        return {"data": data}

# Function to send reasoning queries to OpenAI
async def query_openai_reasoning(formatted_data: Dict[str, Any], question: str) -> Dict[str, Any]:
    try:
        messages = [
            {"role": "system", "content": "You are a process mining and data analysis expert. Analyze the provided data and answer the question with detailed insights."},
            {"role": "user", "content": f"I have the following data:\n{json.dumps(formatted_data, indent=2)}\n\nQuestion: {question}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",  # Using O3 model
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        
        return {
            "analysis": response.choices[0].message.content,
            "reasoning_path": "OpenAI reasoning process applied to structured data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# Function to generate process graph
def generate_process_graph(data: List[Dict[str, Any]]) -> str:
    # Create a directed graph
    G = nx.DiGraph()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Group by case_id
    for case_id, case_group in df.groupby('case_id'):
        # Sort by timestamp
        case_group = case_group.sort_values('timestamp')
        activities = case_group['activity'].tolist()
        
        # Add edges between consecutive activities
        for i in range(len(activities) - 1):
            source = activities[i]
            target = activities[i + 1]
            
            # Add nodes if they don't exist
            if not G.has_node(source):
                G.add_node(source)
            if not G.has_node(target):
                G.add_node(target)
            
            # Add edge or increment weight if it exists
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)
    
    # Draw the graph with improved styling
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Consistent layout with seed
    
    # Calculate edge widths based on weights
    max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
    edge_width = [G[u][v]['weight'] / max_weight * 5 for u, v in G.edges()]
    
    # Draw with better colors and styling
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800, alpha=0.8, edgecolors='navy')
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7, edge_color='navy', arrowstyle='->', arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add edge labels showing frequency count
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Process Flow Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Convert plot to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.getvalue()).decode()

# Endpoint for fetching process mining data
@app.post("/api/process-mining")
async def process_mining(request: QueryRequest):
    try:
        # Step 1: Load SQL results from Supabase
        if request.query_type == "process_mining":
            data = fetch_data_from_supabase("workforce_reskilling_events", request.filters)
        elif request.query_type == "knowledge_graph":
            # Join tables for knowledge graph
            employee_data = fetch_data_from_supabase("employee_profile")
            job_risk_data = fetch_data_from_supabase("job_risk")
            cases_data = fetch_data_from_supabase("workforce_reskilling_cases", request.filters)
            
            # Combine the data
            df_employees = pd.DataFrame(employee_data)
            df_jobs = pd.DataFrame(job_risk_data)
            df_cases = pd.DataFrame(cases_data)
            
            # Merge dataframes
            df_merged = df_employees.merge(df_jobs, on='soc_code', how='left')
            df_merged = df_merged.merge(df_cases, on='employee_id', how='left')
            
            data = df_merged.to_dict(orient='records')
        elif request.query_type == "causal_graph":
            # Fetch all tables for causal analysis
            employee_data = fetch_data_from_supabase("employee_profile")
            job_risk_data = fetch_data_from_supabase("job_risk")
            cases_data = fetch_data_from_supabase("workforce_reskilling_cases", request.filters)
            events_data = fetch_data_from_supabase("workforce_reskilling_events")
            
            # Combine all data
            df_employees = pd.DataFrame(employee_data)
            df_jobs = pd.DataFrame(job_risk_data)
            df_cases = pd.DataFrame(cases_data)
            df_events = pd.DataFrame(events_data)
            
            # Merge dataframes
            df_merged = df_employees.merge(df_jobs, on='soc_code', how='left')
            df_merged = df_merged.merge(df_cases, on='employee_id', how='left')
            df_merged = df_merged.merge(df_events, on='case_id', how='left')
            
            data = df_merged.to_dict(orient='records')
        else:
            raise HTTPException(status_code=400, detail="Invalid query type")
        
        # Step 2: Format the structured data as JSON for OpenAI
        formatted_data = format_data_for_openai(data, request.query_type)
        
        # Step 3: Generate process graph if applicable
        graph_image = None
        if request.query_type == "process_mining" and data:
            graph_image = generate_process_graph(data)
        
        return {
            "data": formatted_data,
            "graph_image": graph_image
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Endpoint for OpenAI reasoning queries
@app.post("/api/reasoning")
async def reasoning_query(request: ReasoningRequest):
    try:
        # Step 3: Send reasoning query to OpenAI O3
        reasoning_result = await query_openai_reasoning(request.data, request.question)
        
        # Step 4: Return AI-driven analysis & recommendations
        return reasoning_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)