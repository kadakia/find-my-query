import os
import streamlit as st

# -------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="Find My Query", page_icon="🧠", layout="wide")

required_secrets = ["OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST", "PINECONE_API_KEY", "APP_USERS", "DEVELOPER_PASSWORD"]
for secret in required_secrets:
    if secret not in st.secrets:
        st.error(f"Missing required secret: '{secret}'. Please add it to `.streamlit/secrets.toml`.")
        st.stop()

# Set OS environment variables BEFORE importing Langfuse/OpenAI/Pinecone
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGFUSE_SECRET_KEY"] = st.secrets["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_PUBLIC_KEY"] = st.secrets["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_HOST"] = st.secrets["LANGFUSE_HOST"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

import re
from pinecone import Pinecone
from openai import OpenAI
import pandas as pd
import json
import time
import hashlib
import sqlparse
from langfuse import Langfuse

# This variable now safely routes BOTH your Langfuse traces and your Pinecone database namespace!
env_name = st.secrets.get("LANGFUSE_TRACING_ENVIRONMENT", "local-dev")

# Initialize clients
llm_client = OpenAI()
langfuse = Langfuse()

# Initialize Pinecone and connect to the index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "analyst-queries"

# Note: Pinecone indexes take a moment to provision on first creation. 
# Ensure the index 'analyst-queries' exists in your Pinecone dashboard with 1536 dimensions.
if index_name not in [idx.name for idx in pc.list_indexes()]:
    st.error(f"Pinecone index '{index_name}' not found. Please create it in the Pinecone dashboard with 1536 dimensions.")
    st.stop()

index = pc.Index(index_name)

def get_embedding(text):
    """Calls OpenAI to convert text into a 1536-dimensional vector."""
    response = llm_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# -------------------------------------------------------------------
# 2. APP USER AUTHORIZATION
# -------------------------------------------------------------------
st.sidebar.header("🔐 Access Control")
app_users = st.secrets["APP_USERS"]
current_app_user = st.sidebar.selectbox("App User (Required)", [""] + app_users)

if not current_app_user:
    st.warning("Please select an App User from the sidebar to use the application.")
    st.stop()

# -------------------------------------------------------------------
# 3. METADATA & ENRICHMENT LOGIC
# -------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("⚙️ Developer Settings")
dev_pass = st.sidebar.text_input("Developer Password", type="password")

if dev_pass == st.secrets["DEVELOPER_PASSWORD"]:
    uploaded_csv = st.sidebar.file_uploader("Upload Redshift Metadata (CSV)", type="csv")
    if uploaded_csv:
        start_time = time.time()
        st.session_state['metadata_df'] = pd.read_csv(uploaded_csv)
        exec_time = time.time() - start_time
        st.sidebar.success(f"Metadata loaded in {exec_time:.2f} seconds!")
elif dev_pass:
    st.sidebar.error("Incorrect developer password.")

def get_relevant_metadata(sql_query):
    if 'metadata_df' not in st.session_state:
        return "No metadata provided."
    df = st.session_state['metadata_df']
    sql_lower = sql_query.lower()
    
    if 'schema_name' in df.columns and 'table_name' in df.columns:
        df['full_table_name'] = df['schema_name'].astype(str) + '.' + df['table_name'].astype(str)
        unique_full_tables = df['full_table_name'].unique()
        found_tables = [t for t in unique_full_tables if str(t).lower() in sql_lower]
        
        if found_tables:
            filtered_df = df[df['full_table_name'].isin(found_tables)].copy()
            filtered_df = filtered_df.drop(columns=['full_table_name'])
            return filtered_df.to_csv(index=False)
            
    return "No matching schema.table references found in metadata."

def generate_enrichment_data(sql_query):
    metadata_context = get_relevant_metadata(sql_query)
    prompt = f"""
    Analyze the following SQL query. 
    SQL Query: {sql_query}
    Database Metadata (Context): {metadata_context}
    Task:
    1. Write a brief, 1-2 sentence description of what the query calculates or retrieves. 
    2. Identify the primary query type (e.g., SELECT, INSERT, UPDATE, DELETE, CREATE, DROP).
    3. List the database tables involved in the query using the exact `schema_name.table_name` format.
    """
    
    # Guarantee structured output using OpenAI's JSON Schema implementation
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "sql_enrichment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "query_type": {"type": "string"},
                    "tables_involved": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["summary", "query_type", "tables_involved"],
                "additionalProperties": False
            }
        }
    }
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format=schema
    )
    return json.loads(response.choices[0].message.content.strip())

# -------------------------------------------------------------------
# 4. PARSING & DEDUPLICATION LOGIC
# -------------------------------------------------------------------
def normalize_sql(raw_sql):
    """Removes comments, unifies whitespace, and lowercases for deduplication."""
    cleaned = sqlparse.format(raw_sql, strip_comments=True, reindent=False)
    return " ".join(cleaned.split()).lower()

def clean_extraneous_lines(raw_sql):
    """Removes standalone -- comments, /* */ comments, and blank lines."""
    cleaned = re.sub(r'/\*.*?\*/', '', raw_sql, flags=re.DOTALL)
    
    final_lines = []
    for line in cleaned.splitlines():
        s = line.strip()
        if not s: continue 
        if s.startswith('--'): continue 
        final_lines.append(line)
        
    return '\n'.join(final_lines)

def generate_doc_id(author, tenant, normalized_sql):
    """Generates a unique MD5 hash for a query based on its core identity."""
    unique_string = f"{author}_{tenant}_{normalized_sql}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def get_all_pinecone_records():
    """Fetches all records from Pinecone to populate dropdowns and the Vault."""
    all_matches = []
    # Pinecone list() yields lists of IDs in the namespace
    for ids in index.list(namespace=env_name):
        if ids:
            fetch_res = index.fetch(ids=ids, namespace=env_name)
            all_matches.extend(fetch_res.vectors.values())
    return all_matches

def get_unique_metadata_values(all_records):
    """Extracts unique tables, authors, and tenants from fetched records."""
    tables, authors, tenants = set(), set(), set()
    
    for record in all_records:
        meta = record.metadata
        if not meta: continue
        
        if 'tables_involved' in meta and isinstance(meta['tables_involved'], list):
            tables.update(meta['tables_involved'])
        if 'author' in meta: authors.add(meta['author'])
        if 'tenant' in meta: tenants.add(meta['tenant'])
            
    return sorted(list(tables)), sorted(list(authors)), sorted(list(tenants))

# -------------------------------------------------------------------
# 5. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------
# Pre-fetch all records once per load for UI population
all_db_records = get_all_pinecone_records()
db_count = len(all_db_records)

st.title(f"🧠 Find My Query (User: {current_app_user})")
st.write(f"*Currently tracking **{db_count}** queries in the **{env_name}** environment.*")

unique_tables, unique_authors, unique_tenants = get_unique_metadata_values(all_db_records)

tab_search, tab_batch, tab_add, tab_vault = st.tabs([
    "🔍 Search Queries", "📂 Batch Import", "➕ Add New Query", "📚 Vault Overview"
])

# --- TAB 1: HYBRID SEARCH ---
with tab_search:
    col_text, col_voice = st.columns([3, 1])
    
    with col_voice:
        audio_val = st.audio_input("Speak your search:")
        if audio_val and audio_val != st.session_state.get("last_audio"):
            with st.spinner("Transcribing..."):
                transcript = llm_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", audio_val)
                )
                st.session_state["voice_term"] = transcript.text
                st.session_state["last_audio"] = audio_val
                st.rerun() 

    with col_text:
        search_term = st.text_input("Natural Language Intent:", value=st.session_state.get("voice_term", ""))
    
    # Advanced Filters row
    col_t, col_a, col_ten, col_tab = st.columns(4)
    with col_t:
        query_types = ["All", "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"]
        selected_type = st.selectbox("Query Type", query_types)
    with col_a:
        selected_authors = st.multiselect("Author(s)", unique_authors)
    with col_ten:
        selected_tenants = st.multiselect("Tenant(s)", unique_tenants)
    with col_tab:
        selected_tables = st.multiselect("Table(s) Involved", unique_tables, placeholder="e.g., public.user")
    
    if st.button("Run Search", type="primary"):
        if search_term:
            start_time = time.time()
            
            # Build Pinecone Filters natively
            and_clauses = []
            if selected_type != "All":
                and_clauses.append({"query_type": {"$eq": selected_type}})
            
            # For each table selected, ensure it is in the metadata array
            for table in selected_tables:
                and_clauses.append({"tables_involved": table})
            
            if len(selected_authors) == 1:
                and_clauses.append({"author": {"$eq": selected_authors[0]}})
            elif len(selected_authors) > 1:
                and_clauses.append({"author": {"$in": selected_authors}})
                
            if len(selected_tenants) == 1:
                and_clauses.append({"tenant": {"$eq": selected_tenants[0]}})
            elif len(selected_tenants) > 1:
                and_clauses.append({"tenant": {"$in": selected_tenants}})
            
            filter_dict = None
            if len(and_clauses) == 1:
                filter_dict = and_clauses[0]
            elif len(and_clauses) > 1:
                filter_dict = {"$and": and_clauses}

            with langfuse.start_as_current_observation(as_type="span", name="Search Execution") as span:
                langfuse.update_current_trace(tags=[env_name], user_id=current_app_user)
                span.update(input={"search_term": search_term, "filters": filter_dict})

                # Embed the search term and query Pinecone
                query_vector = get_embedding(search_term)
                
                results = index.query(
                    vector=query_vector,
                    top_k=5, 
                    include_metadata=True,
                    filter=filter_dict,
                    namespace=env_name
                )
                
                exec_time = time.time() - start_time
                st.success(f"Search completed in {exec_time:.2f} seconds.")
                st.divider()
                
                logged_results = []
                if results.matches:
                    for match in results.matches:
                        r_meta = match.metadata
                        r_summary = r_meta.get('summary', 'No summary available.')
                        logged_results.append(r_summary)
                        
                        st.info(f"**Intent Summary:** {r_summary}")
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.caption(f"**Author:** {r_meta.get('author', 'Unknown')}")
                        m_col2.caption(f"**Tenant:** {r_meta.get('tenant', 'Unknown')}")
                        m_col3.caption(f"**Type:** {r_meta.get('query_type', 'N/A')}")
                        
                        st.code(r_meta.get('raw_sql', ''), language="sql")
                        st.divider()
                    
                    span.update(output=logged_results, metadata={"execution_time_sec": exec_time, "results_count": len(logged_results)})
                else:
                    st.warning("No matching queries found.")
                    span.update(output="No results", metadata={"execution_time_sec": exec_time})
            
            langfuse.flush()

# --- TAB 2: BATCH IMPORT ---
with tab_batch:
    st.subheader("Batch Ingest `.sql` Files")
    st.info("**Naming Convention:** Files must be named `[author]-[tenant].sql` (e.g., `rkadakia-csi.sql`). Only valid SQL queries separated by semicolons will be extracted. Comments will be ignored.")
    
    uploaded_sql_files = st.file_uploader("Upload `.sql` files", type="sql", accept_multiple_files=True)
    
    if st.button("Run Batch Ingestion"):
        if uploaded_sql_files:
            start_time = time.time()
            
            with langfuse.start_as_current_observation(as_type="span", name="Batch Query Upload") as span:
                langfuse.update_current_trace(tags=[env_name], user_id=current_app_user)
                span.update(input={"files": [f.name for f in uploaded_sql_files]})
                
                total_queries_found = 0
                total_ingested = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_sql_files):
                    filename = file.name
                    base_name = filename.replace(".sql", "")
                    parts = base_name.split("-")
                    
                    if len(parts) < 2:
                        st.error(f"File `{filename}` does not match `author-tenant.sql` format. Skipping.")
                        continue
                        
                    author = parts[0].strip()
                    tenant = parts[1].strip()
                    raw_text = file.read().decode("utf-8")
                    parsed_statements = sqlparse.split(raw_text)
                    
                    for stmt in parsed_statements:
                        cleaned_stmt = clean_extraneous_lines(stmt)
                        norm_sql = normalize_sql(cleaned_stmt)
                        if not norm_sql: continue 
                        
                        total_queries_found += 1
                        doc_id = generate_doc_id(author, tenant, norm_sql)
                        
                        # Deduplication check in Pinecone namespace
                        fetch_res = index.fetch(ids=[doc_id], namespace=env_name)
                        if doc_id not in fetch_res.vectors:
                            enrichment = generate_enrichment_data(cleaned_stmt)
                            tables_list = enrichment.get("tables_involved", [])
                            summary_text = enrichment.get("summary", "")
                            
                            # Embed the generated summary
                            vector = get_embedding(summary_text)
                            
                            metadata = {
                                "summary": summary_text,
                                "raw_sql": cleaned_stmt.strip(), 
                                "normalized_sql": norm_sql,
                                "query_type": enrichment.get("query_type", "UNKNOWN"),
                                "tables_involved": tables_list,
                                "author": author,
                                "tenant": tenant
                            }
                            
                            index.upsert(vectors=[{"id": doc_id, "values": vector, "metadata": metadata}], namespace=env_name)
                            total_ingested += 1
                            
                    progress_bar.progress((i + 1) / len(uploaded_sql_files))
                    status_text.text(f"Processed {filename}...")
                
                exec_time = time.time() - start_time
                span.update(output={"total_found": total_queries_found, "ingested": total_ingested}, metadata={"execution_time_sec": exec_time})
            
            langfuse.flush()
            
            st.success(f"Batch complete in {exec_time:.2f} seconds!")
            st.metric(label="Total Queries Found", value=total_queries_found)
            st.metric(label="Non-Redundant Queries Ingested", value=total_ingested)
            time.sleep(2) # Give Pinecone a moment to index before user switches tabs
            st.rerun() # Refresh to update the global count
            
        else:
            st.error("Please upload at least one `.sql` file.")

# --- TAB 3: SEAMLESS ADDITION ---
with tab_add:
    st.subheader("Add a New Query to the Vault")
    st.write("Ensure you select the Author and Tenant. Redundant queries will be rejected.")
    
    col_auth, col_ten = st.columns(2)
    with col_auth:
        new_author = st.selectbox("Author", unique_authors if unique_authors else ["rkadakia"])
    with col_ten:
        new_tenant = st.selectbox("Tenant", unique_tenants if unique_tenants else ["csi", "qns"])
        
    new_sql = st.text_area("Paste Raw SQL here:", height=200)
    
    if st.button("Enrich & Add"):
        if new_sql and new_author and new_tenant:
            start_time = time.time()
            
            with langfuse.start_as_current_observation(as_type="span", name="Individual Query Upload") as span:
                langfuse.update_current_trace(tags=[env_name], user_id=current_app_user)
                span.update(input={"author": new_author, "tenant": new_tenant, "sql_length": len(new_sql)})
                
                cleaned_sql = clean_extraneous_lines(new_sql)
                norm_sql = normalize_sql(cleaned_sql)
                
                if not norm_sql:
                    st.error("No valid SQL detected.")
                    st.stop()
                    
                doc_id = generate_doc_id(new_author, new_tenant, norm_sql)
                
                # Deduplication check in Pinecone namespace
                fetch_res = index.fetch(ids=[doc_id], namespace=env_name)
                if doc_id in fetch_res.vectors:
                    st.warning(f"This exact query is already present in the database for author '{new_author}' and tenant '{new_tenant}'.")
                    span.update(output="Redundant query rejected.")
                else:
                    with st.spinner("Analyzing with LLM..."):
                        enrichment = generate_enrichment_data(cleaned_sql)
                        tables_list = enrichment.get("tables_involved", [])
                        summary_text = enrichment.get("summary", "")
                        
                        vector = get_embedding(summary_text)
                        
                        metadata = {
                            "summary": summary_text,
                            "raw_sql": cleaned_sql.strip(), 
                            "normalized_sql": norm_sql,
                            "query_type": enrichment.get("query_type", "UNKNOWN"),
                            "tables_involved": tables_list,
                            "author": new_author,
                            "tenant": new_tenant
                        }
                        
                        index.upsert(vectors=[{"id": doc_id, "values": vector, "metadata": metadata}], namespace=env_name)
                        
                    exec_time = time.time() - start_time
                    st.success(f"Added successfully in {exec_time:.2f} seconds!")
                    span.update(output="Query ingested.", metadata={"execution_time_sec": exec_time})
            
            langfuse.flush()
            time.sleep(2)
            st.rerun()
        else:
            st.error("Please provide Author, Tenant, and SQL.")

# --- TAB 4: VAULT OVERVIEW ---
with tab_vault:
    st.subheader("Your Query Vault")
    st.write("Review all generated intent summaries and metadata across the entire database.")
    
    if all_db_records:
        vault_rows = []
        for record in all_db_records:
            meta = record.metadata
            if not meta: continue
            vault_rows.append({
                "Author": meta.get('author', ''),
                "Tenant": meta.get('tenant', ''),
                "Summary": meta.get('summary', ''),
                "Type": meta.get('query_type', ''),
                "Table(s)": ", ".join(meta.get('tables_involved', []))
            })
        st.dataframe(pd.DataFrame(vault_rows), width='stretch')
    else:
        st.info("Your vault is currently empty.")