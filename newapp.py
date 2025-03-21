import nltk
import re
import string
import gensim
import PyPDF2
import pandas as pd
import spacy
import time
import io
import os
from collections import Counter
from gensim import corpora
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
app = Flask(__name__)
#CORS(app)

# Download necessary NLTK resources with improved error handling
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {str(e)}")
        return False

# Load spaCy model with proper error handling
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            print("Downloading spaCy model. This may take a moment...")
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Error loading spaCy model: {str(e)}")
            return None

# Initialize resources
resources_loaded = download_nltk_resources()
nlp = load_spacy_model()

if not resources_loaded or nlp is None:
    print("Failed to load required resources. Please check your installation.")

# Define skill-related keywords to help identify skill contexts - expanded for better coverage
SKILL_CONTEXT_MARKERS = [
    'experience with', 'skilled in', 'proficient in', 'knowledge of', 
    'familiar with', 'ability to', 'competent in', 'expertise in', 
    'qualified in', 'capable of', 'proficiency with', 'years of experience',
    'certification in', 'degree in', 'trained in', 'specialization in', 
    'background in', 'competency in', 'mastery of', 'fluent in',
    'skills include', 'technical skills', 'experienced in', 'understanding of',
    'demonstrated ability', 'proven experience', 'working knowledge',
    'hands-on experience', 'strong background', 'working with'
]

# Expanded and categorized technical skills dictionary
TECHNICAL_SKILLS_DICT = {
    'Programming': {'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'php', 'swift', 
                   'kotlin', 'rust', 'scala', 'perl', 'r', 'matlab', 'dart', 'bash', 'powershell', 'groovy', 
                   'objective-c', 'vba', 'cobol', 'fortran', 'assembly', 'haskell', 'clojure', 'elixir', 
                   'erlang', 'f#', 'lua', 'prolog', 'scheme', 'lisp', 'julia', 'abap', 'sas'},
    
    'Web Development': {'html', 'css', 'sass', 'less', 'react', 'angular', 'vue', 'svelte', 'node.js', 
                       'express.js', 'django', 'flask', 'spring', 'laravel', 'asp.net', 'jquery', 'bootstrap', 
                       'tailwind', 'webpack', 'babel', 'responsive design', 'progressive web apps', 'spa', 
                       'server-side rendering', 'jamstack', 'rest api', 'graphql', 'oauth', 'jwt', 
                       'web accessibility', 'micro frontends', 'hugo', 'gatsby', 'next.js', 'nuxt.js', 
                       'astro', 'htmx', 'alpine.js', 'fastapi', 'webassembly', 't3 stack'},

    'Databases': {'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'sqlite', 'dynamodb', 
                 'redis', 'cassandra', 'couchdb', 'elasticsearch', 'mariadb', 'firebase', 'neo4j', 'hbase', 
                 'influxdb', 'realm', 'supabase', 'cockroachdb', 'rdbms', 'nosql', 'database modeling', 
                 'indexing', 'sharding', 'replication', 'bigquery', 'timescaledb', 'snowflake', 'clickhouse', 
                 'graph databases', 'data warehousing', 'data lakes', 'etl pipelines', 'presto', 'trino'},

    'Cloud & DevOps': {'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'serverless', 'terraform', 
                      'ansible', 'chef', 'puppet', 'jenkins', 'github actions', 'gitlab ci', 'circleci', 
                      'travis ci', 'ci/cd', 'infrastructure as code', 'monitoring', 'logging', 'elk stack', 
                      'prometheus', 'grafana', 'cloudformation', 'helm', 'openshift', 'istio', 'service mesh', 
                      'argo cd', 'linkerd', 'harbor', 'k3s', 'k9s', 'istio', 'knative', 'consul', 'nomad', 
                      'aws lambda', 'google cloud run', 'bicep', 'terraform cloud', 'fly.io'},

    'Data Science': {'machine learning', 'deep learning', 'artificial intelligence', 'data mining', 'data analysis', 
                    'statistical analysis', 'predictive modeling', 'pandas', 'numpy', 'scipy', 'scikit-learn', 
                    'tensorflow', 'pytorch', 'keras', 'computer vision', 'nlp', 'natural language processing', 
                    'reinforcement learning', 'neural networks', 'data visualization', 'tableau', 'power bi', 
                    'matplotlib', 'seaborn', 'feature engineering', 'data wrangling', 'etl', 'time series analysis', 
                    'huggingface', 'openai api', 'llms', 'data governance', 'mlops', 'fastai', 'h2o.ai', 'xgboost', 
                    'lightgbm', 'bayesian methods', 'auto ml', 'synthetic data', 'big data', 'spark', 'hadoop', 
                    'databricks', 'apache flink', 'kubeflow', 'vertex ai', 'ray'},

    'Mobile Development': {'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin', 'objective-c', 
                          'mobile app development', 'responsive design', 'xamarin', 'ionic', 'cordova', 
                          'native apps', 'hybrid apps', 'app store optimization', 'push notifications', 
                          'mobile ui/ux', 'mobile testing', 'jetpack compose', 'swift ui', 'dart ffi', 
                          'flutter web', 'flutter desktop', 'firebase cloud messaging', 'app clips', 
                          'wearable apps', 'tvOS', 'android automotive', 'cross-platform development'},

    'Software Development': {'oop', 'object-oriented programming', 'functional programming', 'tdd', 'bdd', 
                           'test-driven development', 'agile', 'scrum', 'kanban', 'git', 'version control', 
                           'jira', 'confluence', 'design patterns', 'algorithms', 'data structures', 'api development', 
                           'microservices', 'soa', 'rest', 'graphql', 'grpc', 'websocket', 'solid principles', 
                           'clean code', 'code review', 'pair programming', 'debugging', 'system design', 
                           'scalability', 'event-driven architecture', 'distributed systems', 'message queues', 
                           'rabbitmq', 'kafka', 'event sourcing', 'actor model', 'grpc-web', 'edge computing', 
                           'progressive enhancement', 'domain-driven design', 'hexagonal architecture'},

    'Security': {'cybersecurity', 'information security', 'network security', 'application security', 
                'encryption', 'authentication', 'authorization', 'oauth', 'openid', 'penetration testing', 
                'vulnerability assessment', 'security auditing', 'siem', 'iam', 'dlp', 'ssl/tls', 'vpn', 
                'firewall', 'ids/ips', 'zero trust', 'devsecops', 'threat modeling', 'owasp', 'security by design', 
                'secure coding practices', 'ransomware mitigation', 'sast', 'dast', 'fuzz testing', 'compliance', 
                'gdpr', 'hipaa', 'soc 2', 'iso 27001', 'pci-dss', 'hsm', 'hardware security', 'iot security', 
                'cloud security', 'endpoint protection', 'malware analysis', 'incident response', 'forensics'},

    'Soft Skills': {'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking', 
                  'time management', 'adaptability', 'project management', 'conflict resolution', 
                  'emotional intelligence', 'creativity', 'presentation skills', 'negotiation', 
                  'decision making', 'stress management', 'self-motivation', 'customer service', 
                  'interpersonal skills', 'active listening', 'collaboration', 'mentoring', 'feedback handling', 
                  'public speaking', 'writing skills', 'growth mindset', 'resilience', 'networking', 
                  'storytelling', 'cultural awareness', 'mindfulness', 'ethics in tech'}
}

# Flatten the dictionary for faster lookups
TECHNICAL_SKILLS = set()
for category, skills in TECHNICAL_SKILLS_DICT.items():
    TECHNICAL_SKILLS.update(skills)

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file with improved error handling and chunking"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
            except Exception as e:
                # Continue even if one page fails
                continue
            
        if not text.strip():
            return ""
            
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def preprocess_text(text, max_length=1000000):
    """Preprocess text with lemmatization and cleaning - optimized for performance"""
    if not text or len(text.strip()) == 0:
        return []
    
    # Limit text size for performance in very large documents
    if len(text) > max_length:
        text = text[:max_length]
    
    # Basic cleaning
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with space
    
    # Remove emails and URLs more efficiently with precompiled regex
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Get stopwords but keep certain terms that might be important for skill matching
    stop_words = set(stopwords.words('english'))
    skill_related_words = {'c', 'c++', 'r', 'java', 'go', 'scala', 'no', 'not', 'sql', 'sas', 'spark'}
    stop_words -= skill_related_words  # Keep programming languages and negations
    
    # Process tokens
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = []
    
    # More efficient token filtering
    for token in tokens:
        if (
            token not in stop_words and
            len(token) > 1 and  # Remove single characters
            not token.isdigit() and  # Remove pure numbers
            not all(c in string.punctuation for c in token)  # Remove punctuation
        ):
            filtered_tokens.append(lemmatizer.lemmatize(token))
    
    return filtered_tokens

def extract_ngrams(text, n_range=(1, 3), max_ngrams=5000):
    """Extract n-grams from text with performance limit"""
    tokens = preprocess_text(text)
    if not tokens:
        return []
        
    ngrams = []
    count = 0
    
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            if count >= max_ngrams:
                return ngrams
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
            count += 1
    
    return ngrams

def extract_skills_with_tfidf(resume_text, job_text):
    """Extract important skills using TF-IDF vectorization with improved precision"""
    if not resume_text or not job_text:
        return {}, {}
        
    # Create TF-IDF vectorizer with improved parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,  # Increased from 500 for better coverage
        ngram_range=(1, 3),  # Consider up to trigrams
        stop_words='english',
        min_df=1,  # Minimum document frequency
        max_df=0.9,  # Increased from 0.8 to capture more domain-specific terms
        sublinear_tf=True  # Apply sublinear tf scaling for better results
    )
    
    try:
        # Prepare corpus with resume and job text
        corpus = [resume_text, job_text]
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        
        # Get feature names (terms)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for resume and job
        resume_tfidf = tfidf_matrix[0].toarray()[0]
        job_tfidf = tfidf_matrix[1].toarray()[0]
        
        # Create dictionaries of terms and their scores with better filtering
        resume_skills = {
            feature_names[i]: resume_tfidf[i] 
            for i in range(len(feature_names)) 
            if resume_tfidf[i] > 0 and len(feature_names[i]) > 2
        }
        
        job_skills = {
            feature_names[i]: job_tfidf[i] 
            for i in range(len(feature_names)) 
            if job_tfidf[i] > 0 and len(feature_names[i]) > 2
        }
        
# Get top skills based on TF-IDF scores
        top_resume_skills = dict(sorted(resume_skills.items(), key=lambda x: x[1], reverse=True)[:50])
        top_job_skills = dict(sorted(job_skills.items(), key=lambda x: x[1], reverse=True)[:50])
        
        return top_resume_skills, top_job_skills
    except Exception as e:
        print(f"Error in TF-IDF skill extraction: {str(e)}")
        return {}, {}

def extract_skills_from_text(text, skill_set=TECHNICAL_SKILLS):
    """Extract skills from text using pattern matching and context analysis"""
    if not text:
        return set()
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    extracted_skills = set()
    
    # Extract skills using spaCy for better NER and context analysis
    if nlp:
        # Process text with spaCy with document length limit
        max_length = min(len(text), 100000)  # Limit to 100k chars for performance
        doc = nlp(text_lower[:max_length])
        
        # Extract skills by matching tokens against skill set
        for token in doc:
            if token.text in skill_set:
                extracted_skills.add(token.text)
            
        # Extract multi-word skills
        for skill in skill_set:
            if ' ' in skill and skill in text_lower:
                extracted_skills.add(skill)
    else:
        # Fallback to basic pattern matching if spaCy is not available
        words = text_lower.split()
        for skill in skill_set:
            if ' ' not in skill and skill in words:
                extracted_skills.add(skill)
            elif ' ' in skill and skill in text_lower:
                extracted_skills.add(skill)
    
    # Additional context-based skill detection
    for marker in SKILL_CONTEXT_MARKERS:
        if marker in text_lower:
            # Find the position of the marker
            marker_pos = text_lower.find(marker)
            # Extract the context around the marker
            context_start = max(0, marker_pos - 50)
            context_end = min(len(text_lower), marker_pos + 100)
            context = text_lower[context_start:context_end]
            
            # Check for skills in the context
            for skill in skill_set:
                if skill in context:
                    extracted_skills.add(skill)
    
    return extracted_skills

def categorize_skills(skills):
    """Categorize extracted skills into their respective categories"""
    categorized = {}
    for category, category_skills in TECHNICAL_SKILLS_DICT.items():
        matches = [skill for skill in skills if skill in category_skills]
        if matches:
            categorized[category] = matches
    
    # Add "Other" category for skills not in predefined categories
    other_skills = [skill for skill in skills if not any(skill in cat_skills for cat_skills in TECHNICAL_SKILLS_DICT.values())]
    if other_skills:
        categorized["Other"] = other_skills
    
    return categorized

def analyze_skill_match(resume_skills, job_skills):
    """Analyze the match between resume skills and job skills with improved scoring"""
    if not resume_skills or not job_skills:
        return 0, [], []
    
    # Convert to sets for efficient comparison
    resume_skill_set = set(resume_skills)
    job_skill_set = set(job_skills)
    
    # Find matching skills
    matching_skills = resume_skill_set.intersection(job_skill_set)
    
    # Find missing skills
    missing_skills = job_skill_set - resume_skill_set
    
    # Calculate match score (percentage of job skills found in resume)
    match_score = 0
    if job_skill_set:
        match_score = len(matching_skills) / len(job_skill_set) * 100
    
    return match_score, list(matching_skills), list(missing_skills)

def generate_skill_report(resume_text, job_text):
    """Generate a comprehensive skill analysis report"""
    start_time = time.time()
    report = {}
    
    # Extract skills using TF-IDF
    top_resume_skills_tfidf, top_job_skills_tfidf = extract_skills_with_tfidf(resume_text, job_text)
    
    # Extract direct skills from texts
    resume_skills = extract_skills_from_text(resume_text)
    job_skills = extract_skills_from_text(job_text)
    
    # Merge TF-IDF skills with direct skills for better coverage
    all_resume_skills = set(resume_skills) | set(top_resume_skills_tfidf.keys())
    all_job_skills = set(job_skills) | set(top_job_skills_tfidf.keys())
    
    # Categorize skills
    categorized_resume_skills = categorize_skills(all_resume_skills)
    categorized_job_skills = categorize_skills(all_job_skills)
    
    # Analyze skill match
    match_score, matching_skills, missing_skills = analyze_skill_match(all_resume_skills, all_job_skills)
    
    # Generate report
    report = {
        "match_score": match_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "resume_skills": {
            "all": list(all_resume_skills),
            "categorized": categorized_resume_skills
        },
        "job_skills": {
            "all": list(all_job_skills),
            "categorized": categorized_job_skills
        },
        "processing_time": time.time() - start_time
    }
    
    return report


ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

def extract_text_from_file(file, ext):
    """Extract text from a resume file based on file type."""
    try:
        if ext == 'pdf':
            reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif ext == 'txt':
            return file.read().decode('utf-8', errors='ignore')
        elif ext == 'docx':
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None

@app.route('/api/skill', methods=['POST'])
def analyze_skills():
    """API endpoint to analyze resume (file) and job description (text)."""
    try:
        # Check if resume file is provided
        if 'resume' not in request.files:
            return jsonify({"error": "Resume file is required"}), 400

        # Check if job description is provided as text
        job_text = request.form.get("job")
        if not job_text:
            return jsonify({"error": "Job description text is required"}), 400

        resume_file = request.files['resume']

        # Validate file extension
        resume_ext = resume_file.filename.split('.')[-1].lower()
        if resume_ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Only PDF, DOCX, and TXT files are supported"}), 400

        # Extract text from resume
        resume_text = extract_text_from_file(resume_file, resume_ext)
        if not resume_text:
            return jsonify({"error": "Failed to extract text from the resume file"}), 400

        # Generate skill report (Dummy function, replace with actual logic)
        report = generate_skill_report(resume_text, job_text)

        return jsonify(report)

    except Exception as e:
        print(f"Error in skill analysis: {str(e)}")
        return jsonify({"error": "An error occurred during analysis", "details": str(e)}), 500
    

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health check"""
    return jsonify({
        "status": "healthy",
        "resources_loaded": resources_loaded,
        "spacy_loaded": nlp is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)