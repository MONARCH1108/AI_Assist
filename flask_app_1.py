from flask import Flask, render_template, request, jsonify, session
import os
import tempfile
import sqlite3
import wikipedia
import requests
import json
import re
import feedparser
import time
from datetime import datetime
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import bs4

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize generative model
genai.configure(api_key="API")
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Search API key
SERPER_API_KEY = "YOUR_SERPER_API_KEY"

# Global vector DB
db = None

# ===========================
# Enhanced Long-Term Memory (SQLite)
# ===========================

MEMORY_DB = "memory.db"

def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    
    # Table for Q&A memory
    c.execute('''CREATE TABLE IF NOT EXISTS memory (
                    question TEXT PRIMARY KEY,
                    answer TEXT,
                    timestamp TEXT
                )''')
    
    # Table for user research topics
    c.execute('''CREATE TABLE IF NOT EXISTS research_topics (
                    user_id TEXT,
                    topic TEXT,
                    timestamp TEXT,
                    PRIMARY KEY (user_id, topic)
                )''')
    
    # Table for article summaries
    c.execute('''CREATE TABLE IF NOT EXISTS article_summaries (
                    article_id TEXT PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    summary TEXT,
                    topics TEXT,
                    timestamp TEXT,
                    user_id TEXT
                )''')
    
    # Table for user preferences
    c.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_sources TEXT,
                    citation_style TEXT,
                    last_topics TEXT
                )''')
    
    conn.commit()
    conn.close()

def save_to_memory(question, answer):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT OR REPLACE INTO memory (question, answer, timestamp) VALUES (?, ?, ?)", 
              (question, answer, timestamp))
    conn.commit()
    conn.close()

def retrieve_from_memory(question):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    c.execute("SELECT answer FROM memory WHERE question=?", (question,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def save_research_topic(user_id, topic):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT OR REPLACE INTO research_topics (user_id, topic, timestamp) VALUES (?, ?, ?)", 
              (user_id, topic, timestamp))
    conn.commit()
    conn.close()

def get_research_topics(user_id):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    c.execute("SELECT topic FROM research_topics WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
    topics = [row[0] for row in c.fetchall()]
    conn.close()
    return topics

def save_article_summary(article_id, title, source, summary, topics, user_id):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT OR REPLACE INTO article_summaries 
                (article_id, title, source, summary, topics, timestamp, user_id) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""", 
              (article_id, title, source, summary, topics, timestamp, user_id))
    conn.commit()
    conn.close()

def get_article_summaries(user_id, topic=None):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    
    if topic:
        c.execute("""SELECT title, source, summary, timestamp FROM article_summaries 
                    WHERE user_id=? AND topics LIKE ? ORDER BY timestamp DESC""", 
                  (user_id, f'%{topic}%'))
    else:
        c.execute("""SELECT title, source, summary, timestamp FROM article_summaries 
                    WHERE user_id=? ORDER BY timestamp DESC""", (user_id,))
    
    summaries = [{"title": row[0], "source": row[1], "summary": row[2], "timestamp": row[3]} 
                for row in c.fetchall()]
    conn.close()
    return summaries

def save_user_preferences(user_id, preferred_sources=None, citation_style=None, last_topics=None):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    
    # Get existing preferences
    c.execute("SELECT preferred_sources, citation_style, last_topics FROM user_preferences WHERE user_id=?", (user_id,))
    result = c.fetchone()
    
    if result:
        existing_sources, existing_style, existing_topics = result
        
        # Update only provided values
        new_sources = json.dumps(json.loads(preferred_sources)) if preferred_sources else existing_sources
        new_style = citation_style if citation_style else existing_style
        new_topics = json.dumps(json.loads(last_topics)) if last_topics else existing_topics
    else:
        # Create new preferences
        new_sources = json.dumps([]) if not preferred_sources else preferred_sources
        new_style = "APA" if not citation_style else citation_style
        new_topics = json.dumps([]) if not last_topics else last_topics
    
    c.execute("""INSERT OR REPLACE INTO user_preferences 
                (user_id, preferred_sources, citation_style, last_topics) 
                VALUES (?, ?, ?, ?)""", 
              (user_id, new_sources, new_style, new_topics))
    conn.commit()
    conn.close()

def get_user_preferences(user_id):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    c.execute("SELECT preferred_sources, citation_style, last_topics FROM user_preferences WHERE user_id=?", (user_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return {
            "preferred_sources": json.loads(result[0]) if result[0] else [],
            "citation_style": result[1] if result[1] else "APA",
            "last_topics": json.loads(result[2]) if result[2] else []
        }
    else:
        return {
            "preferred_sources": [],
            "citation_style": "APA",
            "last_topics": []
        }

# ===========================
# Loaders
# ===========================

def load_from_url(url):
    loader = WebBaseLoader(web_path=url, bs_kwargs=dict(parse_only=bs4.SoupStrainer("p")))
    return loader.load()

def load_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# ===========================
# Splitter
# ===========================

def split_documents(docs, chunk_size=1000, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# ===========================
# Vector DB
# ===========================

def create_vector_store(documents, persist_path="db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_path)
    return vectordb

def load_vector_store(persist_path="db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vectordb

# ===========================
# Research APIs
# ===========================

def get_wikipedia_info(query, sentences=3):
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return {
            "source": "Wikipedia",
            "title": query,
            "content": summary,
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        }
    except:
        return None

def get_realtime_info_from_wikipedia(query):
    """Get real-time information from Wikipedia based on the query"""
    try:
        # First try direct lookup
        wiki_info = get_wikipedia_info(query)
        if wiki_info:
            return f"Wikipedia information for '{query}':\n{wiki_info['content']}\nSource: {wiki_info['url']}"
        
        # If direct lookup fails, try search
        wiki_results = get_mediawiki_search(query)
        if wiki_results:
            info = "Wikipedia search results:\n\n"
            for i, result in enumerate(wiki_results[:2], 1):
                info += f"{i}. {result['title']}: {result['snippet']}\n"
            return info
        
        return "No relevant Wikipedia information found."
    except Exception as e:
        return f"Error retrieving Wikipedia information: {str(e)}"

def search_serper(query, num_results=5):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": num_results
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()
    except:
        return {"organic": []}

def search_arxiv(query, max_results=5):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{query}&start=0&max_results={max_results}'
    
    try:
        response = requests.get(base_url + search_query)
        feed = feedparser.parse(response.content)
        
        results = []
        for entry in feed.entries:
            published = entry.published
            title = entry.title
            authors = ', '.join(author.name for author in entry.authors)
            summary = entry.summary
            pdf_url = next((link.href for link in entry.links if link.type == 'application/pdf'), None)
            
            results.append({
                "source": "arXiv",
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
                "pdf_url": pdf_url,
                "url": entry.link
            })
        
        return results
    except:
        return []

def get_mediawiki_search(query, limit=5):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "source": "Wikipedia",
                "title": item["title"],
                "snippet": re.sub(r'<.*?>', '', item["snippet"]),
                "url": f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"
            })
        
        return results
    except:
        return []

def summarize_article(url, title=None):
    try:
        # First try to load and process the content
        documents = load_from_url(url)
        content = "\n\n".join([doc.page_content for doc in documents])
        
        if not title:
            title = url.split("/")[-1].replace("-", " ").title()
        
        # Use the AI model to generate a summary
        prompt = f"""
        Please summarize the following article:
        Title: {title}
        
        Content:
        {content[:15000]}  # Limit content to avoid token limits
        
        Provide a concise summary (250-300 words) that captures the main points, findings, and conclusions.
        """
        
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        return {
            "title": title,
            "url": url,
            "summary": summary
        }
    except Exception as e:
        return {
            "title": title if title else "Unknown Title",
            "url": url,
            "summary": f"Failed to summarize article: {str(e)}"
        }

# ===========================
# Research Organization Functions
# ===========================

def organize_research_by_topic(user_id, topics=None):
    """Organize research by topics"""
    if not topics:
        topics = get_research_topics(user_id)
    
    organized_research = {}
    
    for topic in topics:
        article_summaries = get_article_summaries(user_id, topic)
        organized_research[topic] = article_summaries
    
    return organized_research

def detect_research_topics(query):
    """Use AI to detect potential research topics in a query"""
    try:
        prompt = f"""
        Analyze this research query and extract the main research topics (maximum 3):
        
        Query: {query}
        
        Return only a JSON array of topic strings, nothing else.
        """
        
        response = model.generate_content(prompt)
        topics_text = response.text.strip()
        
        # Extract the JSON array from the response
        topics_match = re.search(r'\[.*\]', topics_text, re.DOTALL)
        if topics_match:
            topics_json = topics_match.group(0)
            topics = json.loads(topics_json)
            return topics
        else:
            # If no JSON array was found, try to extract topics as a list
            topics = [t.strip() for t in topics_text.split(',')]
            return topics[:3]  # Limit to 3 topics
            
    except Exception as e:
        print(f"Error detecting topics: {str(e)}")
        # Extract simple keywords as fallback
        words = re.findall(r'\b[A-Za-z]{4,}\b', query)
        return list(set(words))[:3]

def generate_research_notes(user_id, topic):
    """Generate structured research notes based on a topic"""
    article_summaries = get_article_summaries(user_id, topic)
    
    if not article_summaries:
        return f"No research has been collected on the topic: {topic}"
    
    notes = f"# Research Notes: {topic}\n\n"
    notes += f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    notes += "## Key Findings\n\n"
    
    # Get AI to synthesize the key findings
    if article_summaries:
        summaries_text = "\n\n".join([f"Title: {s['title']}\nSummary: {s['summary']}" for s in article_summaries[:5]])
        
        prompt = f"""
        Based on these article summaries about {topic}, provide 3-5 key findings or main points:
        
        {summaries_text}
        
        Format as bullet points.
        """
        
        try:
            response = model.generate_content(prompt)
            key_findings = response.text.strip()
            notes += key_findings + "\n\n"
        except:
            notes += "- Could not generate key findings\n\n"
    
    notes += "## Article Summaries\n\n"
    
    for idx, summary in enumerate(article_summaries, 1):
        notes += f"### {idx}. {summary['title']}\n"
        notes += f"Source: {summary['source']}\n"
        notes += f"Date: {summary['timestamp'][:10]}\n\n"
        notes += f"{summary['summary']}\n\n"
    
    notes += "## Research Gaps\n\n"
    
    # Get AI to identify research gaps
    if article_summaries:
        prompt = f"""
        Based on these article summaries about {topic}, identify 2-3 potential research gaps or areas needing further investigation:
        
        {summaries_text}
        
        Format as bullet points.
        """
        
        try:
            response = model.generate_content(prompt)
            gaps = response.text.strip()
            notes += gaps
        except:
            notes += "- Could not identify research gaps"
    
    return notes

# ===========================
# LLM + Retrieval Chain
# ===========================

def get_context_from_query(query, vectordb, k=4):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)

def get_user_context(user_id):
    """Get personalized context for the user"""
    # Get research topics
    topics = get_research_topics(user_id)
    topics_str = ", ".join(topics) if topics else "No specific topics yet"
    
    # Get preferences
    prefs = get_user_preferences(user_id)
    preferred_sources = ", ".join(prefs["preferred_sources"]) if prefs["preferred_sources"] else "None specified"
    
    # Get recent articles
    recent_summaries = get_article_summaries(user_id)
    recent_articles = ""
    for i, summary in enumerate(recent_summaries[:3], 1):
        recent_articles += f"{i}. {summary['title']} ({summary['source']})\n"
    
    if not recent_articles:
        recent_articles = "No articles summarized yet"
    
    context = f"""
    USER RESEARCH PROFILE:
    Research topics: {topics_str}
    Preferred sources: {preferred_sources}
    Citation style: {prefs['citation_style']}
    
    RECENT RESEARCH ARTICLES:
    {recent_articles}
    """
    
    return context

def ask_with_context(query, user_id="default_user"):
    if not db and not query.lower().startswith(("search", "find", "look up", "what are", "can you find")):
        return "Pipeline not initialized. Please upload a PDF or enter a Wikipedia URL first for document-specific questions."

    # Check long-term memory
    memory_response = retrieve_from_memory(query)
    if memory_response:
        return f"(From Memory)\n{memory_response}"
    
    # Detect if this is a research query
    research_request = any(keyword in query.lower() for keyword in [
        "research", "find", "search", "latest", "articles", "papers", "study", 
        "summarize", "organize", "source", "arxiv"
    ])
    
    # Handle research-specific requests
    if research_request:
        # Detect research topics
        detected_topics = detect_research_topics(query)
        
        # Save detected topics
        for topic in detected_topics:
            save_research_topic(user_id, topic)
            
        # Update last topics in preferences
        prefs = get_user_preferences(user_id)
        last_topics = list(set(prefs["last_topics"] + detected_topics))[:5]  # Keep last 5 topics
        save_user_preferences(user_id, last_topics=json.dumps(last_topics))
        
        # Check if it's a summarization request
        if "summarize" in query.lower() and ("article" in query.lower() or "http" in query.lower()):
            # Extract URL from the query
            url_match = re.search(r'https?://[^\s]+', query)
            if url_match:
                url = url_match.group(0)
                summary_result = summarize_article(url)
                
                # Save the summary
                article_id = url.split("/")[-1]
                save_article_summary(
                    article_id, 
                    summary_result["title"], 
                    url, 
                    summary_result["summary"], 
                    json.dumps(detected_topics), 
                    user_id
                )
                
                return f"""## Article Summary: {summary_result['title']}

Source: {url}

{summary_result['summary']}

*This summary has been saved to your research collection under topics: {', '.join(detected_topics)}*
"""
        
        # For general research queries, search multiple sources
        arxiv_results = search_arxiv(query) if any(keyword in query.lower() for keyword in ["paper", "research", "study", "scientific", "arxiv"]) else []
        wiki_results = get_mediawiki_search(query)
        serper_results = search_serper(query)["organic"] if "organic" in search_serper(query) else []
        
        # Format results
        formatted_results = "## Research Results\n\n"
        
        if arxiv_results:
            formatted_results += "### Scientific Papers (arXiv)\n\n"
            for i, result in enumerate(arxiv_results[:3], 1):
                formatted_results += f"{i}. **{result['title']}**\n"
                formatted_results += f"   Authors: {result['authors']}\n"
                formatted_results += f"   [Read Paper]({result['url']}) | [PDF]({result['pdf_url']})\n"
                formatted_results += f"   *{result['summary'][:150]}...*\n\n"
        
        if wiki_results:
            formatted_results += "### Encyclopedia Articles\n\n"
            for i, result in enumerate(wiki_results[:3], 1):
                formatted_results += f"{i}. **{result['title']}**\n"
                formatted_results += f"   [Read on Wikipedia]({result['url']})\n"
                formatted_results += f"   *{result['snippet']}...*\n\n"
        
        if serper_results:
            formatted_results += "### Web Results\n\n"
            for i, result in enumerate(serper_results[:3], 1):
                formatted_results += f"{i}. **{result.get('title', 'Untitled')}**\n"
                formatted_results += f"   [Read Article]({result.get('link', '#')})\n"
                formatted_results += f"   *{result.get('snippet', 'No snippet available')}*\n\n"
        
        # Add user context and recommendations
        user_context = get_user_context(user_id)
        
        prompt = f"""
        You are a research assistant helping with the following query:
        
        {query}
        
        Based on the user's research profile and the search results, provide a helpful response that:
        1. Answers their question directly
        2. References the most relevant sources
        3. Connects to their existing research interests if relevant
        
        USER CONTEXT:
        {user_context}
        
        SEARCH RESULTS:
        {formatted_results}
        
        Give a concise, helpful response that prioritizes the most relevant information.
        """
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Save to long-term memory
        save_to_memory(query, answer)
        
        return answer
    
    # For document-specific queries when DB is available
    if db:
        context = get_context_from_query(query, db)
        wiki_info = get_realtime_info_from_wikipedia(query)
        
        user_context = get_user_context(user_id)
        
        prompt = f"""
        You are an AI research assistant. Use the context provided below to answer the question.
        If the answer isn't in the context, leverage the external info or say "I don't have specific information on that in the current documents."

        <user_context>
        {user_context}
        </user_context>

        <document_context>
        {context}
        </document_context>

        <external_info>
        {wiki_info}
        </external_info>

        Question: {query}
        
        Provide a helpful, direct answer.
        """
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Save to long-term memory
        save_to_memory(query, answer)
        
        return answer
    
    # For general questions when no document is loaded
    wiki_info = get_realtime_info_from_wikipedia(query)
    user_context = get_user_context(user_id)
    
    prompt = f"""
    You are a research assistant. Answer the following question based on your knowledge and any external information provided.
    
    <user_context>
    {user_context}
    </user_context>
    
    <external_info>
    {wiki_info}
    </external_info>
    
    Question: {query}
    
    Provide a helpful, concise answer. If you don't have enough information, suggest what kinds of sources might help.
    """
    
    response = model.generate_content(prompt)
    answer = response.text.strip()
    
    # Save to long-term memory
    save_to_memory(query, answer)
    
    return answer

# ===========================
# Research Organization Endpoints
# ===========================

def organize_research_notes(user_id, format_type="markdown"):
    """Organize all research into a formatted document"""
    topics = get_research_topics(user_id)
    
    if not topics:
        return "No research topics found. Start by asking research questions to build your knowledge base."
    
    organized_research = organize_research_by_topic(user_id, topics)
    
    if format_type == "markdown":
        notes = "# Research Notes Collection\n\n"
        notes += f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        for topic, summaries in organized_research.items():
            notes += f"## Topic: {topic}\n\n"
            
            if not summaries:
                notes += "No articles collected for this topic yet.\n\n"
                continue
                
            for idx, summary in enumerate(summaries, 1):
                notes += f"### {idx}. {summary['title']}\n"
                notes += f"Source: {summary['source']}\n"
                notes += f"Date: {summary['timestamp'][:10]}\n\n"
                notes += f"{summary['summary']}\n\n"
    
    return notes

# ===========================
# Setup & Query Functions
# ===========================

def setup_pipeline_from_url(url, persist_path="db"):
    global db
    docs = load_from_url(url)
    chunks = split_documents(docs)
    db = create_vector_store(chunks, persist_path)
    return "Pipeline set up successfully from URL"

def setup_pipeline_from_pdf(file_path, persist_path="db"):
    global db
    docs = load_from_pdf(file_path)
    chunks = split_documents(docs)
    db = create_vector_store(chunks, persist_path)
    return "Pipeline set up successfully from PDF"

def load_existing_pipeline(persist_path="db"):
    global db
    try:
        db = load_vector_store(persist_path)
        return "Existing pipeline loaded"
    except:
        return "No existing pipeline found"

def ask(query, user_id="default_user"):
    return ask_with_context(query, user_id)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ===========================
# Flask Routes
# ===========================

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = f"user_{int(time.time())}"
    return render_template('index.html')

@app.route('/process_wiki', methods=['POST'])
def process_wiki():
    wiki_url = request.form.get('wiki_url')
    if not wiki_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        result = setup_pipeline_from_url(wiki_url)
        session['source_type'] = 'wiki'
        session['source'] = wiki_url
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = setup_pipeline_from_pdf(filepath)
            session['source_type'] = 'pdf'
            session['source'] = filename
            return jsonify({"message": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No question provided"}), 400

    user_id = session.get('user_id', 'default_user')

    try:
        answer = ask(query, user_id)
        return jsonify({
            "answer": answer,
            "source_type": session.get('source_type', ''),
            "source": session.get('source', '')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    # Keep user_id for persistence across chat clearing
    user_id = session.get('user_id', 'default_user')
    session.clear()
    session['user_id'] = user_id
    
    global db
    db = None
    return jsonify({"message": "Chat cleared but research memory preserved"})

@app.route('/search_arxiv', methods=['POST'])
def arxiv_search():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No search query provided"}), 400
    
    results = search_arxiv(query)
    return jsonify({"results": results})

@app.route('/organize_notes', methods=['POST'])
def organize_notes():
    user_id = session.get('user_id', 'default_user')
    format_type = request.form.get('format', 'markdown')
    
    notes = organize_research_notes(user_id, format_type)
    return jsonify({"notes": notes})

@app.route('/topic_notes/<topic>', methods=['GET'])
def topic_notes(topic):
    user_id = session.get('user_id', 'default_user')
    notes = generate_research_notes(user_id, topic)
    return jsonify({"notes": notes})

@app.route('/save_preferences', methods=['POST'])
def save_prefs():
    user_id = session.get('user_id', 'default_user')
    
    preferred_sources = request.form.get('preferred_sources')
    citation_style = request.form.get('citation_style')
    
    save_user_preferences(
        user_id, 
        preferred_sources=preferred_sources, 
        citation_style=citation_style
    )
    
    return jsonify({"message": "Preferences saved successfully"})

@app.route('/get_topics', methods=['GET'])
def get_topics():
    user_id = session.get('user_id', 'default_user')
    topics = get_research_topics(user_id)
    return jsonify({"topics": topics})

@app.route('/summarize_url', methods=['POST'])
def summarize_url():
    url = request.form.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    user_id = session.get('user_id', 'default_user')
    title = request.form.get('title')
    
    summary_result = summarize_article(url, title)
    
    # Save the summary
    article_id = url.split("/")[-1]
    topics = detect_research_topics(summary_result["title"])
    
    save_article_summary(
        article_id, 
        summary_result["title"], 
        url, 
        summary_result["summary"], 
        json.dumps(topics), 
        user_id
    )
    
    for topic in topics:
        save_research_topic(user_id, topic)
    
    return jsonify(summary_result)

@app.route('/get_user_profile', methods=['GET'])
def get_profile():
    user_id = session.get('user_id', 'default_user')
    
    # Get research topics
    topics = get_research_topics(user_id)
    
    # Get preferences
    preferences = get_user_preferences(user_id)
    
    # Get article counts
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM article_summaries WHERE user_id=?", (user_id,))
    article_count = c.fetchone()[0]
    conn.close()
    
    return jsonify({
        "topics": topics,
        "preferences": preferences,
        "article_count": article_count
    })

if __name__ == '__main__':
    init_memory_db()
    load_existing_pipeline()  # Try to load existing pipeline if available
    app.run(debug=True)