import re
import os
import PyPDF2
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from flask import Flask, request, jsonify
from io import BytesIO

NLTK_DIR = "/temp/nltk_data"  # Render-specific location
os.makedirs(NLTK_DIR, exist_ok=True)

# Append to nltk path
nltk.data.path.append(NLTK_DIR)

# Download required NLTK resources
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

# Initialize Flask app
app = Flask(__name__)
# Download necessary NLTK resources
'''try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')'
    '''

# Define skills categories
TECHNICAL_SKILLS = [
    'python', 'java', 'c++', 'c#', 'javascript', 'html', 'css', 'react', 
    'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 
    'sql', 'mysql', 'postgresql', 'mongodb', 'firebase', 'aws', 'azure', 
    'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'linux', 
    'windows', 'macos', 'bash', 'powershell', 'excel', 'word', 'powerpoint',
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'tableau',
    'power bi', 'r', 'matlab', 'flutter', 'kotlin', 'swift', 'ios', 'android'
]

SOFT_SKILLS = [
    'communication', 'teamwork', 'leadership', 'problem solving', 
    'critical thinking', 'decision making', 'time management', 
    'project management', 'creativity', 'adaptability', 'flexibility',
    'organization', 'planning', 'detail oriented', 'analytical', 
    'interpersonal', 'verbal communication', 'written communication'
]

# Common ATS keywords by industry
ATS_KEYWORDS = {
    'Software Development': [
        'software engineer', 'developer', 'full stack', 'backend', 'frontend',
        'devops', 'ci/cd', 'api', 'rest', 'json', 'database', 'cloud',
        'microservices', 'algorithms', 'data structures', 'agile', 'scrum'
    ],
    'Data Science': [
        'data scientist', 'machine learning', 'deep learning', 'data analysis',
        'statistics', 'predictive modeling', 'natural language processing',
        'computer vision', 'big data', 'data visualization', 'a/b testing'
    ],
    'Marketing': [
        'marketing', 'digital marketing', 'social media', 'seo', 'sem',
        'content strategy', 'branding', 'email marketing', 'analytics',
        'campaign management', 'customer acquisition', 'cro'
    ]
}

# Resume helper functions
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_email(text):
    """Extract email address from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

def extract_phone(text):
    """Extract phone number from text"""
    phone_pattern = r'(\+\d{1,3}[-.\s]??)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else None

def extract_skills(text, skill_list):
    """Extract skills from text that match a given skill list"""
    text_lower = text.lower()
    found_skills = []
    for skill in skill_list:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.append(skill)
    return found_skills

def count_bullet_points(text):
    """Count bullet points in the text"""
    bullet_patterns = [r'•', r'·', r'\\-', r'\\*', r'\\+']
    count = 0
    for pattern in bullet_patterns:
        count += len(re.findall(pattern, text))
    return count

def calculate_word_count(text):
    """Calculate total word count"""
    words = word_tokenize(text)
    return len(words)

def analyze_education(text):
    """Detect education details"""
    education_keywords = [
        'bachelor', 'master', 'phd', 'doctorate', 'degree', 'university', 
        'college', 'institute', 'diploma', 'certification'
    ]
    education_pattern = r'\b(?:' + '|'.join(education_keywords) + r')\b'
    education_matches = re.finditer(education_pattern, text.lower())
    
    education_found = False
    for match in education_matches:
        education_found = True
        break
    
    return education_found

def analyze_experience_format(text):
    """Analyze job experience format quality"""
    date_pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s+(?:to|–|-)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}|\d{4}\s+(?:to|–|-)\s+\d{4}|present\b'
    date_matches = re.findall(date_pattern, text.lower())
    
    # Check for action verbs at the beginning of bullet points
    action_verbs = [
        'developed', 'implemented', 'created', 'designed', 'managed', 
        'led', 'executed', 'coordinated', 'achieved', 'improved',
        'increased', 'decreased', 'negotiated', 'organized', 'analyzed',
        'evaluated', 'researched', 'trained', 'supervised'
    ]
    
    action_verb_pattern = r'(?:^|\n)(?:\s*[•·\-\*\+]\s*)(' + '|'.join(action_verbs) + r')\b'
    action_verbs_found = re.findall(action_verb_pattern, text.lower())
    
    return {
        'date_format_count': len(date_matches),
        'action_verbs_count': len(action_verbs_found)
    }

def check_quantifiable_results(text):
    """Check for quantifiable results in experience"""
    # Look for percentages, currencies, and numbers followed by specific keywords
    patterns = [
        r'\d+%',  # Percentages
        r'[$€£¥]\s*\d+(?:[,.]\d+)*',  # Currencies
        r'\d+\s*(?:million|billion|thousand)',  # Large numbers
        r'increased\s+\w+\s+by\s+\d+',  # Increases
        r'decreased\s+\w+\s+by\s+\d+',  # Decreases
        r'generated\s+\w+\s+\d+',  # Generation
        r'saved\s+\w+\s+\d+',  # Savings
        r'improved\s+\w+\s+by\s+\d+'  # Improvements
    ]
    
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text.lower()))
    
    return count

def analyze_resume(text):
    """Analyze resume text and provide feedback"""
    word_count = calculate_word_count(text)
    bullet_points = count_bullet_points(text)
    email = extract_email(text)
    phone = extract_phone(text)
    education_found = analyze_education(text)
    
    tech_skills = extract_skills(text, TECHNICAL_SKILLS)
    soft_skills = extract_skills(text, SOFT_SKILLS)
    
    experience_analysis = analyze_experience_format(text)
    quantifiable_results = check_quantifiable_results(text)
    
    # Determine industry based on skills
    industry_scores = {}
    for industry, keywords in ATS_KEYWORDS.items():
        industry_skills = extract_skills(text, keywords)
        industry_scores[industry] = len(industry_skills)
    
    likely_industry = max(industry_scores, key=industry_scores.get)
    missing_industry_keywords = [kw for kw in ATS_KEYWORDS[likely_industry] 
                               if kw not in extract_skills(text, ATS_KEYWORDS[likely_industry])]
    
    # Calculate section scores
    format_score = min(100, (bullet_points / 15) * 100) if bullet_points > 0 else 30
    
    contact_score = 0
    if email:
        contact_score += 50
    if phone:
        contact_score += 50
    
    skills_score = min(100, ((len(tech_skills) + len(soft_skills)) / 12) * 100)
    
    experience_score = 0
    if experience_analysis['date_format_count'] > 0:
        experience_score += 40
    if experience_analysis['action_verbs_count'] > 5:
        experience_score += 30
    if quantifiable_results >= 3:
        experience_score += 30
    else:
        experience_score += quantifiable_results * 10
    
    education_score = 70 if education_found else 30
    
    ats_score = min(100, (len(extract_skills(text, ATS_KEYWORDS[likely_industry])) / 
                         len(ATS_KEYWORDS[likely_industry])) * 100)
    
    overall_score = (format_score * 0.2 + contact_score * 0.1 + skills_score * 0.25 + 
                    experience_score * 0.25 + education_score * 0.1 + ats_score * 0.1)
    
    # Generate feedback dictionary
    return {
        "summary": {
            "score": round(overall_score),
            "word_count": word_count,
            "likely_industry": likely_industry,
            "strengths": get_strengths(tech_skills, soft_skills, experience_analysis, quantifiable_results),
            "weaknesses": get_weaknesses(tech_skills, soft_skills, experience_analysis, quantifiable_results, missing_industry_keywords)
        },
        "sections": {
            "format": {
                "score": round(format_score),
                "bullet_points": bullet_points,
                "feedback": get_format_feedback(format_score, bullet_points),
                "suggestions": get_format_suggestions(format_score, bullet_points)
            },
            "contact": {
                "score": round(contact_score),
                "has_email": email is not None,
                "has_phone": phone is not None,
                "feedback": "Contact information is complete and easy to find." if contact_score == 100 else "Contact information is incomplete."
            },
            "skills": {
                "score": round(skills_score),
                "technical_skills": tech_skills,
                "soft_skills": soft_skills,
                "missing_technical": list(set(TECHNICAL_SKILLS) - set(tech_skills))[:5],
                "missing_soft": list(set(SOFT_SKILLS) - set(soft_skills))[:3],
                "feedback": get_skills_feedback(skills_score, tech_skills, soft_skills)
            },
            "experience": {
                "score": round(experience_score),
                "date_formats": experience_analysis['date_format_count'],
                "action_verbs": experience_analysis['action_verbs_count'],
                "quantifiable_results": quantifiable_results,
                "feedback": get_experience_feedback(experience_score, quantifiable_results),
                "suggestions": get_experience_suggestions(experience_score, quantifiable_results)
            },
            "education": {
                "score": round(education_score),
                "education_found": education_found,
                "feedback": "Education section is well-structured." if education_found else "Education section needs improvement or was not found."
            }
        },
        "ats": {
            "score": round(ats_score),
            "passLikelihood": get_ats_likelihood(ats_score),
            "likely_industry": likely_industry,
            "industry_keywords_found": extract_skills(text, ATS_KEYWORDS[likely_industry]),
            "missing_keywords": missing_industry_keywords[:5],
            "suggestions": get_ats_suggestions(ats_score, missing_industry_keywords)
        }
    }

def get_strengths(tech_skills, soft_skills, experience_analysis, quantifiable_results):
    """Generate list of resume strengths"""
    strengths = []
    
    if len(tech_skills) >= 5:
        strengths.append("Strong technical skill set")
    
    if len(soft_skills) >= 3:
        strengths.append("Good balance of soft skills")
    
    if experience_analysis['action_verbs_count'] >= 5:
        strengths.append("Effective use of action verbs")
    
    if experience_analysis['date_format_count'] >= 2:
        strengths.append("Clear work history timeline")
    
    if quantifiable_results >= 3:
        strengths.append("Excellent use of quantifiable achievements")
    
    # Ensure we have at least 2 strengths
    if len(strengths) < 2:
        if len(tech_skills) > 0:
            strengths.append("Has relevant technical skills")
        if len(soft_skills) > 0:
            strengths.append("Includes soft skills")
    
    return strengths

def get_weaknesses(tech_skills, soft_skills, experience_analysis, quantifiable_results, missing_keywords):
    """Generate list of resume weaknesses"""
    weaknesses = []
    
    if len(tech_skills) < 5:
        weaknesses.append("Limited technical skills highlighted")
    
    if len(soft_skills) < 3:
        weaknesses.append("Few soft skills mentioned")
    
    if experience_analysis['action_verbs_count'] < 5:
        weaknesses.append("Limited use of action verbs")
    
    if quantifiable_results < 2:
        weaknesses.append("Lacks quantifiable achievements")
    
    if len(missing_keywords) > 3:
        weaknesses.append("Missing important industry keywords")
    
    return weaknesses

def get_format_feedback(score, bullet_points):
    """Generate feedback for resume format"""
    if score >= 80:
        return "Your resume has good formatting with effective use of bullet points."
    elif score >= 60:
        return "Your resume formatting is acceptable but could be improved."
    else:
        return "Your resume formatting needs significant improvement. Consider using more bullet points."

def get_format_suggestions(score, bullet_points):
    """Generate formatting suggestions"""
    suggestions = []
    
    if bullet_points < 10:
        suggestions.append("Use more bullet points to highlight achievements and responsibilities")
    
    suggestions.append("Ensure consistent formatting throughout")
    suggestions.append("Use clear section headings")
    
    if score < 60:
        suggestions.append("Consider using a professional resume template")
        suggestions.append("Improve readability with better spacing and organization")
    
    return suggestions

def get_skills_feedback(score, tech_skills, soft_skills):
    """Generate feedback for skills section"""
    if score >= 80:
        return "Your skills section is comprehensive and well-balanced."
    elif score >= 60:
        return "Your skills section is adequate but could be expanded."
    else:
        return "Your skills section needs significant improvement. Consider adding more relevant skills."

def get_experience_feedback(score, quantifiable_results):
    """Generate feedback for experience section"""
    if score >= 80:
        return "Your experience section effectively demonstrates your impact."
    elif score >= 60:
        return "Your experience section is acceptable but could be strengthened."
    else:
        return "Your experience section needs improvement with more quantifiable achievements."

def get_experience_suggestions(score, quantifiable_results):
    """Generate experience section suggestions"""
    suggestions = []
    
    if quantifiable_results < 3:
        suggestions.append("Add more metrics and quantifiable results (%, $, numbers)")
    
    suggestions.append("Begin bullet points with strong action verbs")
    suggestions.append("Focus on achievements rather than responsibilities")
    
    if score < 70:
        suggestions.append("Include clear employment dates for each position")
        suggestions.append("Remove outdated or irrelevant experience")
    
    return suggestions

def get_ats_likelihood(score):
    """Determine likelihood of passing ATS"""
    if score >= 80:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

def get_ats_suggestions(score, missing_keywords):
    """Generate ATS optimization suggestions"""
    suggestions = [
        "Tailor your resume to each job description",
        "Use standard section headings (Experience, Skills, Education)"
    ]
    
    if len(missing_keywords) > 0:
        suggestions.append("Add industry-specific keywords that match job descriptions")
    
    if score < 70:
        suggestions.append("Avoid unusual formatting or graphics that ATS might not process")
        suggestions.append("Use both spelled-out terms and acronyms (e.g., 'Artificial Intelligence (AI)')")
    
    return suggestions

# Sample function to analyze a PDF resume
def analyze_resume_from_pdf(pdf_file):
    """
    Analyze a resume from a PDF file
    
    Args:
        pdf_file: The PDF file object
        
    Returns:
        dict: Analysis results
    """
    resume_text = extract_text_from_pdf(pdf_file)
    return analyze_resume(resume_text)

# Example of a function that could be used in an API endpoint
def analyze_resume_api(pdf_file_bytes):
    """
    Function to be used in API endpoint
    
    Args:
        pdf_file_bytes: Bytes of the PDF file
        
    Returns:
        dict: Analysis results
    """
    pdf_file = BytesIO(pdf_file_bytes)
    return analyze_resume_from_pdf(pdf_file)

# Flask routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Resume analyzer API is running"})

@app.route('/api/analyze', methods=['POST'])
def analyze_resume_endpoint():
    """Endpoint to analyze a resume"""
    # Check if request contains file
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    
    file = request.files['resume']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check if file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    try:
        # Analyze the resume
        analysis = analyze_resume_from_pdf(file)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/text', methods=['POST'])
def analyze_resume_text_endpoint():
    """Endpoint to analyze resume text"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "No resume text provided"}), 400
    
    try:
        resume_text = request.json['text']
        analysis = analyze_resume(resume_text)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Configure allowed file upload size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)