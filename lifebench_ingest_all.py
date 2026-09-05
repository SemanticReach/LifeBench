# lifebench_ingest_all.py - Ingest ALL users 1-10 with retry logic
import json
import io
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import time
import sys
import tempfile
import argparse

load_dotenv()

SERVER_URL = os.getenv("HB_SERVER_URL", "http://18.220.128.24:8000")
API_KEY = os.getenv("HB_API_KEY", "")
DB_NAME = "fractal_db"  # Using fractal_db since it works
QA_NS_PREFIX = "lifebench_qa_"
DOC_NS_PREFIX = "lifebench_doc_"
EMBED_DIM = 2048  # Increased to support phase_dim=512

# ── Document Schema ──────────────────────────────────────────────────────────
# Note: phase_dim is removed - server uses default (512)
LIFEBENCH_DOC_SCHEMA = json.dumps({
    "molecule": "Row",
    "primary_key": {"name": "cell_id", "encoding": "exact"},
    "fields": {
        "cell_id": {"name": "cell_id", "encoding": "exact"},
        "user_id": {"name": "user_id", "encoding": "exact"},
        "session": {"name": "session", "encoding": "exact"},
        "cell_type": {"name": "cell_type", "encoding": "exact"},
        "content": {"name": "content", "encoding": "semantic"},
    },
    "field_order": ["cell_id", "user_id", "session", "cell_type", "content"]
})

DOC_ROW_FIELDS = ["cell_id", "user_id", "session", "cell_type", "content"]

# ── QA Schema ──────────────────────────────────────────────────────────────────

LIFEBENCH_QA_SCHEMA = json.dumps({
    "molecule": "Row",
    "primary_key": {"name": "qa_id", "encoding": "exact"},
    "fields": {
        "qa_id": {"name": "qa_id", "encoding": "exact"},
        "user_id": {"name": "user_id", "encoding": "exact"},
        "category": {"name": "category", "encoding": "exact"},
        "question": {"name": "question", "encoding": "semantic"},
        "answer": {"name": "answer", "encoding": "exact"},
        "evidence": {"name": "evidence", "encoding": "exact"},
        "sample_id": {"name": "sample_id", "encoding": "exact"},
    },
    "field_order": ["qa_id", "user_id", "category", "question", "answer", "evidence", "sample_id"]
})

QA_ROW_FIELDS = ["qa_id", "user_id", "category", "question", "answer", "evidence", "sample_id"]

def load_existing_users() -> set:
    ns_file = "lifebench_doc_namespaces.json"
    if os.path.exists(ns_file):
        with open(ns_file, 'r') as f:
            namespaces = json.load(f)
            return set(int(k) for k in namespaces.keys())
    return set()

def save_doc_namespace(user_id: int, namespace: str):
    ns_file = "lifebench_doc_namespaces.json"
    namespaces = {}
    if os.path.exists(ns_file):
        with open(ns_file, 'r') as f:
            namespaces = json.load(f)
    namespaces[str(user_id)] = namespace
    with open(ns_file, 'w') as f:
        json.dump(namespaces, f, indent=2)
    print(f"    Saved namespace for user {user_id} -> {namespace}")

def save_qa_namespace(user_id: int, namespace: str):
    ns_file = "lifebench_qa_namespaces.json"
    namespaces = {}
    if os.path.exists(ns_file):
        with open(ns_file, 'r') as f:
            namespaces = json.load(f)
    namespaces[str(user_id)] = namespace
    with open(ns_file, 'w') as f:
        json.dump(namespaces, f, indent=2)

def flatten_user_context(user_data: dict, user_id: int) -> list:
    records = []
    
    conv = user_data.get('conversation', {})
    for key, value in conv.items():
        if key.startswith('session_') and not key.endswith('_date_time'):
            if isinstance(value, list):
                for turn in value:
                    if isinstance(turn, dict):
                        speaker = turn.get('speaker', 'unknown')
                        text = turn.get('text', '')
                        session_num = key.replace('session_', '')
                        if text:
                            records.append({
                                "cell_id": f"user_{user_id}_session_{session_num}_turn_{len(records)}",
                                "user_id": str(user_id),
                                "session": session_num,
                                "cell_type": "conversation",
                                "content": f"[Session {session_num}] {speaker}: {text}"
                            })
    
    events = user_data.get('event_summary', {})
    for key, value in events.items():
        if key.startswith('events_session_'):
            if isinstance(value, list):
                for event in value:
                    if isinstance(event, dict):
                        summary = event.get('summary', '')
                        session_num = key.replace('events_session_', '')
                        if summary:
                            records.append({
                                "cell_id": f"user_{user_id}_session_{session_num}_event_{len(records)}",
                                "user_id": str(user_id),
                                "session": session_num,
                                "cell_type": "event",
                                "content": f"[Session {session_num}] EVENT: {summary}"
                            })
    
    obs = user_data.get('observation', {})
    for key, value in obs.items():
        if key.endswith('_observation'):
            if value:
                session_num = key.replace('_observation', '').replace('session_', '')
                records.append({
                    "cell_id": f"user_{user_id}_session_{session_num}_obs_{len(records)}",
                    "user_id": str(user_id),
                    "session": session_num,
                    "cell_type": "observation",
                    "content": f"[Session {session_num}] OBSERVATION: {value}"
                })
    
    summaries = user_data.get('session_summary', {})
    for key, value in summaries.items():
        if key.endswith('_summary'):
            if value:
                session_num = key.replace('_summary', '').replace('session_', '')
                records.append({
                    "cell_id": f"user_{user_id}_session_{session_num}_summary_{len(records)}",
                    "user_id": str(user_id),
                    "session": session_num,
                    "cell_type": "summary",
                    "content": f"[Session {session_num}] SUMMARY: {value}"
                })
    
    print(f"    Extracted {len(records)} context cells")
    return records

def upload_user_document(user_id: int, context_records: list, timeout: int = 7200, max_retries: int = 5) -> str:
    if not context_records:
        print(f"    WARNING: No context records for user {user_id}")
        return None
    
    namespace = f"{DOC_NS_PREFIX}{user_id}"
    df = pd.DataFrame(context_records, columns=DOC_ROW_FIELDS)
    
    print(f"    Uploading {len(df)} document rows → namespace: {namespace}")
    print(f"    Document size: {len(df)} cells")
    
    tmp_path = None
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
                df.to_csv(tmp, index=False)
                tmp_path = tmp.name
            
            with open(tmp_path, 'rb') as f:
                print(f"    Attempt {attempt+1}/{max_retries} with {timeout}s timeout...")
                resp = requests.post(
                    f"{SERVER_URL}/build_ingest_data/",
                    headers={"X-API-Key": API_KEY} if API_KEY else {},
                    files={"file": (f"lifebench_doc_{user_id}.csv", f, "text/csv")},
                    data={
                        "dim": EMBED_DIM,
                        "seed": 42,
                        "depth": 3,
                        "db_name": DB_NAME,
                        "namespace": namespace,
                        "template_schema": LIFEBENCH_DOC_SCHEMA,
                        "on_conflict": "update",
                    },
                    timeout=timeout,
                )
            
            if resp.ok:
                result = resp.json()
                rows_added = result.get('rows_added', 0)
                print(f"    ✅ Document uploaded: {namespace} ({rows_added} rows)")
                return namespace
            else:
                print(f"    ❌ Upload failed (attempt {attempt+1}): {resp.status_code} - {resp.text[:200]}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"    Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
        except requests.exceptions.Timeout:
            print(f"    ⏱️ Upload timed out (attempt {attempt+1}) after {timeout}s")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"    ❌ Upload error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                tmp_path = None
    
    return None

def ingest_qa_rows(user_id: int, qa_pairs: list, cache: dict, timeout: int = 7200, max_retries: int = 3) -> str:
    namespace = f"{QA_NS_PREFIX}{user_id}"
    rows = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa.get('question', '')
        rows.append({
            "qa_id": f"user_{user_id}_qa_{i}",
            "user_id": str(user_id),
            "category": str(qa.get('category', '')),
            "question": question,
            "answer": qa.get('answer', ''),
            "evidence": json.dumps(qa.get('evidence', [])),
            "sample_id": str(qa.get('sample_id', i)),
        })
    
    df = pd.DataFrame(rows, columns=QA_ROW_FIELDS)
    tmp_path = None
    
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
                df.to_csv(tmp, index=False)
                tmp_path = tmp.name
            
            with open(tmp_path, 'rb') as f:
                print(f"    Attempt {attempt+1}/{max_retries} with {timeout}s timeout...")
                resp = requests.post(
                    f"{SERVER_URL}/build_ingest_data/",
                    headers={"X-API-Key": API_KEY} if API_KEY else {},
                    files={"file": (f"lifebench_qa_{user_id}.csv", f, "text/csv")},
                    data={
                        "dim": EMBED_DIM,
                        "seed": 42,
                        "depth": 3,
                        "db_name": DB_NAME,
                        "namespace": namespace,
                        "template_schema": LIFEBENCH_QA_SCHEMA,
                        "on_conflict": "update",
                    },
                    timeout=timeout,
                )
            
            if resp.ok:
                result = resp.json()
                rows_added = result.get('rows_added', 0)
                print(f"    ✅ QA ingested: {namespace} ({rows_added} rows)")
                return namespace
            else:
                print(f"    ❌ QA ingest failed (attempt {attempt+1}): {resp.status_code} - {resp.text[:200]}")
                if attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    print(f"    Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
        except requests.exceptions.Timeout:
            print(f"    ⏱️ QA ingest timed out (attempt {attempt+1}) after {timeout}s")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"    ❌ QA ingest error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                tmp_path = None
    
    return None

def precompute_embeddings(users: list) -> dict:
    from sentence_transformers import SentenceTransformer
    print("\n🧠 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_questions = []
    for user in users:
        for qa in user.get('qa', []):
            if question := qa.get('question'):
                all_questions.append(question)
    unique_questions = list(set(all_questions))
    print(f"Precomputing {len(unique_questions)} unique questions...")
    cache = {}
    batch_size = 128
    for i in range(0, len(unique_questions), batch_size):
        batch = unique_questions[i:i+batch_size]
        vectors = model.encode(batch, show_progress_bar=True)
        for text, vec in zip(batch, vectors):
            cache[text] = vec.tolist()
        print(f"  {min(i+batch_size, len(unique_questions))}/{len(unique_questions)}")
    return cache

def wipe_namespace(namespace: str, db_name: str = DB_NAME, timeout: int = 30) -> bool:
    print(f"  Wiping namespace '{namespace}'...")
    try:
        resp = requests.delete(
            f"{SERVER_URL}/db/{db_name}/namespace/{namespace}",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            timeout=timeout,
        )
        if resp.status_code in (200, 404):
            print(f"  ✅ Wiped (status {resp.status_code})")
            return True
        else:
            print(f"  ⚠️  {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"  ❌ Wipe error: {e}")
        return False

def delete_database(db_name: str = DB_NAME) -> bool:
    """Delete the entire database to recreate it with the right dimension."""
    print(f"  Deleting database '{db_name}'...")
    try:
        resp = requests.delete(
            f"{SERVER_URL}/db/{db_name}",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            timeout=60,
        )
        if resp.status_code in (200, 404):
            print(f"  ✅ Deleted (status {resp.status_code})")
            return True
        else:
            print(f"  ⚠️  {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"  ❌ Delete error: {e}")
        return False

def wipe_all_lifebench_data(users: list = None, wipe_db: bool = False):
    if users is None:
        users = range(1, 11)
    
    if wipe_db:
        print(f"\n{'='*60}")
        print(f"Deleting database '{DB_NAME}' to recreate with dim={EMBED_DIM}")
        print(f"{'='*60}")
        delete_database(DB_NAME)
        print(f"\n✅ Database deleted! Will be recreated on first ingest.")
        return
    
    print(f"\n{'='*60}")
    print(f"Wiping LifeBench Data for Users {min(users)}-{max(users)}")
    print(f"{'='*60}")
    print("\n📄 Wiping document namespaces:")
    for user_id in users:
        doc_ns = f"{DOC_NS_PREFIX}{user_id}"
        wipe_namespace(doc_ns)
    print("\n❓ Wiping QA namespaces:")
    for user_id in users:
        qa_ns = f"{QA_NS_PREFIX}{user_id}"
        wipe_namespace(qa_ns)
    ns_files = ["lifebench_doc_namespaces.json", "lifebench_qa_namespaces.json"]
    print("\n🗑️  Clearing namespace tracking files:")
    for ns_file in ns_files:
        if os.path.exists(ns_file):
            os.remove(ns_file)
            print(f"  ✅ Removed {ns_file}")
        else:
            print(f"  - {ns_file} not found")
    print(f"\n{'='*60}")
    print(f"✅ Wipe complete!")
    print(f"{'='*60}")

def check_server_health():
    try:
        resp = requests.get(f"{SERVER_URL}/", timeout=5)
        return resp.status_code == 200
    except:
        return False

def process_user(user_id: int, user_data: dict, cache: dict, timeout: int = 7200):
    print(f"\n{'='*50}")
    print(f"Processing User {user_id}/10")
    print(f"{'='*50}")
    if not check_server_health():
        print(f"  ⚠️  Server not responding! Waiting 60 seconds...")
        time.sleep(60)
        if not check_server_health():
            print(f"  ❌ Server still down. Skipping user {user_id}")
            return None, None
    qa_pairs = user_data.get('qa', [])
    print(f"  QA pairs: {len(qa_pairs)}")
    context = flatten_user_context(user_data, user_id)
    print(f"  Context cells: {len(context)}")
    doc_namespace = upload_user_document(user_id, context, timeout=timeout, max_retries=5)
    if not doc_namespace:
        print(f"  ❌ Upload failed after all retries. Skipping user {user_id}")
        return None, None
    save_doc_namespace(user_id, doc_namespace)
    qa_namespace = ingest_qa_rows(user_id, qa_pairs, cache, timeout=timeout, max_retries=3)
    if not qa_namespace:
        print(f"  ⚠️  QA ingest failed after retries, but document upload succeeded for user {user_id}")
        return doc_namespace, None
    if qa_namespace:
        save_qa_namespace(user_id, qa_namespace)
    time.sleep(2)
    return doc_namespace, qa_namespace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest ALL LifeBench users 1-10')
    parser.add_argument('--wipe', action='store_true', help='Wipe existing data before ingestion')
    parser.add_argument('--wipe-db', action='store_true', help='Delete the entire database and recreate')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout in seconds (default: 7200 = 2 hours)')
    parser.add_argument('--start-user', type=int, default=1, help='Start from specific user ID (1-10)')
    parser.add_argument('--end-user', type=int, default=10, help='End at specific user ID (1-10)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"LifeBench Ingest - ALL Users (1-10)")
    print(f"{'='*60}")
    print(f"Server: {SERVER_URL}")
    print(f"DB: {DB_NAME}")
    print(f"Dimension: {EMBED_DIM}")
    print(f"Timeout: {args.timeout}s ({args.timeout/60:.0f} minutes)")
    print(f"{'='*60}\n")
    
    with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    print(f"Total users in dataset: {len(users)}")
    
    if args.wipe_db:
        wipe_all_lifebench_data(wipe_db=True)
        print("\nProceeding with fresh ingestion...\n")
        time.sleep(2)
    
    if args.wipe:
        wipe_all_lifebench_data(range(args.start_user, args.end_user + 1))
        print("\nProceeding with fresh ingestion...\n")
        time.sleep(2)
    
    existing_users = load_existing_users()
    if existing_users and not args.wipe and not args.wipe_db:
        print(f"Already processed users: {sorted(existing_users)}")
    
    users_to_process = [i for i in range(args.start_user, args.end_user + 1)]
    if not args.wipe and not args.wipe_db:
        users_to_process = [i for i in users_to_process if i not in existing_users]
    
    if not users_to_process:
        print("\n✅ All requested users already processed!")
        sys.exit(0)
    
    print(f"\nUsers to process: {users_to_process}")
    
    cache = precompute_embeddings(users)
    
    successful = 0
    for idx, user_id in enumerate(users_to_process, 1):
        user_data = users[user_id-1]
        try:
            print(f"\n{'🔄'*30}")
            print(f"Starting User {user_id} at {time.strftime('%H:%M:%S')}")
            print(f"{'🔄'*30}")
            doc_ns, qa_ns = process_user(user_id, user_data, cache, timeout=args.timeout)
            if doc_ns and qa_ns:
                successful += 1
                print(f"\n✅ User {user_id} completed successfully!")
            else:
                print(f"\n⚠️  User {user_id} completed with issues")
        except Exception as e:
            print(f"\n💥 ERROR processing user {user_id}: {e}")
            import traceback
            traceback.print_exc()
        print(f"\n📊 Progress: {successful}/{idx} successful so far")
        if user_id < len(users_to_process):
            print("  Waiting 5 seconds before next user...")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print(f"✅ LifeBench ingestion complete!")
    print(f"Successfully processed: {successful}/{len(users_to_process)} users")
    print(f"{'='*60}")