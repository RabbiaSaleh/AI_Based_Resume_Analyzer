import os
import re
import pdfplumber
import docx
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, make_response
from werkzeug.utils import secure_filename
import logging
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from datetime import datetime
import pdfkit  # Requires wkhtmltopdf installed

# Import real DB functions
from db import (
    save_user_credentials,
    verify_user_credentials,
    save_user_info,
    save_resume_data
)

# Load spaCy model (en_core_web_sm); handle if not installed
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Load model and vectorizer
model = None
vectorizer = None
try:
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
except Exception as e:
    # If not found, keep as None and log the error
    logging.warning(f"Could not load model/vectorizer: {e}")

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

SECTION_ALIASES = {
    'summary': [
        'summary', 'professional summary', 'Profile', 'about', 'objective', 'career summary', 'personal profile', 'overview', 'professional overview', 'introduction'
    ],
    'skills': [
        'skills', 'technical skills', 'key skills', 'core skills', 'expertise', 'competencies', 'areas of expertise', 'technical proficiency', 'proficiencies', 'core competencies'
    ],
    'experience': [
        'experience', 'work experience', 'professional experience', 'employment history', 'work history', 'career history', 'professional background', 'employment', 'career', 'positions held'
    ],
    'education': [
        'education', 'academic background', 'academic qualifications', 'educational background', 'qualifications'
    ]
}

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads folder exists

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path, ext):
    if ext == 'pdf':
        text_list = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_list.append(page_text)
                elif pytesseract and Image:
                    # Fallback to OCR if no text detected
                    img = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(img)
                    text_list.append(ocr_text)
                else:
                    text_list.append('')
        return '\n'.join(text_list)
    elif ext == 'docx':
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

def normalize_heading(text):
    return re.sub(r'[^a-z0-9 ]', '', text.strip().lower())

def fuzzy_find_heading(norm_lines, section_key):
    aliases = SECTION_ALIASES[section_key]
    for i, line in enumerate(norm_lines):
        for alias in aliases:
            if difflib.SequenceMatcher(None, alias, line).ratio() > 0.8 or alias in line:
                return i
    return -1

def find_section_bounds(norm_lines, section_key):
    start = fuzzy_find_heading(norm_lines, section_key)
    if start == -1:
        return -1, -1
    # Find the next section heading
    for j in range(start + 1, len(norm_lines)):
        for key, aliases in SECTION_ALIASES.items():
            if key == section_key:
                continue
            for alias in aliases:
                if difflib.SequenceMatcher(None, alias, norm_lines[j]).ratio() > 0.8 or alias in norm_lines[j]:
                    return start, j
    return start, len(norm_lines)

def extract_summary(lines, norm_lines):
    # Only extract a short intro or overview, not the full experience
    s_start, s_end = find_section_bounds(norm_lines, 'summary')
    if s_start != -1:
        summary_lines = [lines[i].strip() for i in range(s_start+1, s_end) if lines[i].strip()]
        # Limit to first 3-4 lines or 400 chars
        summary_lines = summary_lines[:4]
        summary_text = ' '.join(summary_lines).strip()
        return summary_text[:400].rsplit('.', 1)[0] + '.' if len(summary_text) > 400 else summary_text
    else:
        # No summary heading: use first 3-4 lines before experience/education
        exp_start = fuzzy_find_heading(norm_lines, 'experience')
        edu_start = fuzzy_find_heading(norm_lines, 'education')
        first_section = min([i for i in [exp_start, edu_start] if i != -1], default=len(lines))
        intro_lines = []
        for i in range(0, first_section):
            line = lines[i].strip()
            if line:
                intro_lines.append(line)
            if len(intro_lines) >= 4:
                break
        summary_text = ' '.join(intro_lines)
        return summary_text[:400].rsplit('.', 1)[0] + '.' if len(summary_text) > 400 else summary_text

def extract_name_and_contact(lines):
    # Try to extract name (first non-empty line, not a heading)
    name = ""
    contact = ""
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Skip lines that look like headings
        if re.match(r'^(summary|profile|about|objective|skills?|experience|education|contact|personal|professional)', l, re.I):
            continue
        # If looks like an email or phone, treat as contact
        if re.search(r'@', l) or re.search(r'\b\d{10,}\b', l) or re.search(r'linkedin\.com', l, re.I):
            contact = l
            continue
        # If not set, treat as name (first non-heading, non-contact line)
        if not name:
            name = l
        # If both found, break
        if name and contact:
            break
    # If contact not found, try to find in all lines
    if not contact:
        for line in lines:
            if re.search(r'@', line) or re.search(r'\b\d{10,}\b', line) or re.search(r'linkedin\.com', line, re.I):
                contact = line.strip()
                break
    return name, contact

def extract_experience(lines, norm_lines):
    # Find explicit experience section
    e_start, e_end = find_section_bounds(norm_lines, 'experience')
    exp_entries = []
    # Patterns for job roles and organizations
    job_role_pattern = re.compile(
        r'(intern|teacher|assistant|faculty|instructor|professor|engineer|developer|manager|analyst|consultant|officer|specialist|scientist|administrator|coordinator|lead|head|director|volunteer|company|organization|school|university|college|department|firm|agency|office|campus|group|inc|ltd|llc|corporation|district)', re.I
    )
    date_pattern = re.compile(r'(\d{4}|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4})', re.I)
    # If experience section found, extract all non-empty, non-heading lines under it (all subheadings and bullets)
    if e_start != -1:
        for i in range(e_start+1, e_end):
            line = lines[i].strip()
            # Skip empty lines, headings, and project-like lines
            if not line or re.match(r'^(skills?|education|summary|projects?|profile|about|objective|personal|professional)', line, re.I):
                continue
            # Exclude lines that look like projects (e.g., contain 'project', 'github', or links)
            if re.search(r'(project|github\.com|http[s]?://)', line, re.I):
                continue
            # Accept lines with job role/org and date pattern
            if job_role_pattern.search(line) and date_pattern.search(line):
                exp_entries.append(line)
            # Accept lines with job role/org and not a project
            elif job_role_pattern.search(line) and not re.search(r'(project|github\.com|http[s]?://)', line, re.I):
                exp_entries.append(line)
    # If not found, fallback to summary for experience-like lines
    if not exp_entries:
        s_start, s_end = find_section_bounds(norm_lines, 'summary')
        summary_lines = lines[s_start+1:s_end] if s_start != -1 else lines[:10]
        for line in summary_lines:
            line = line.strip()
            if not line or len(line) < 6:
                continue
            # Only pick lines that look like job entries (year + role + org)
            if re.search(r'(project|github\.com|http[s]?://)', line, re.I):
                continue
            has_date = bool(date_pattern.search(line))
            has_role = bool(job_role_pattern.search(line))
            if has_date and has_role:
                exp_entries.append(line)
    exp_entries = list(dict.fromkeys([e for e in exp_entries if e]))
    return '\n'.join(exp_entries) if exp_entries else 'Not detected'

def extract_skills(lines, norm_lines):
    # Improved: extract from skills section, summary, and experience bullets
    k_start, k_end = find_section_bounds(norm_lines, 'skills')
    skills = []
    skill_pattern = re.compile(
        r'\b(python|java|c\+\+|c#|sql|oracle|oracle sql|machine learning|deep learning|data science|nlp|tensorflow|keras|scikit-learn|pandas|numpy|linux|cloud|aws|azure|devops|docker|kubernetes|matlab|simulink|fpga|vhdl|verilog|solidworks|autocad|ansys|catia|thermodynamics|hvac|html|css|javascript|php|network|microcontroller|embedded|pcb|arduino|project management|crm|erp|excel|statistics|data analysis|object[- ]oriented programming|oop|ml|ai|artificial intelligence|computer vision|oracle sql|sql server|postgresql|mongodb|react|angular|vue|spring|django|flask|rest api|git|github|bitbucket|jira|scrum|kanban|unit testing|pytest|selenium|spark|hadoop|tableau|powerbi|sas|r studio|jupyter|colab|keras|pytorch|opencv|bash|shell|powershell|linux admin|windows server|vmware|virtualization|network security|cybersecurity|penetration testing|ethical hacking|blockchain|iot|internet of things|robotics|raspberry pi|arduino|verilog|vhdl|embedded systems|plc|scada|automation|mechatronics|catia|proe|creo|abaqus|hypermesh|cam|cad|fea|cfd|ansys fluent|solid edge|nx|siemens|sap|oracle erp|quickbooks|tally|business intelligence|bi|etl|data warehouse|data mining|big data|hive|pig|hbase|zookeeper|kafka|elasticsearch|kibana|logstash|splunk|prometheus|grafana|jenkins|ansible|chef|puppet|terraform|cloudformation|aws lambda|azure functions|gcp|google cloud|cloud computing|devops engineer|site reliability engineer|sre|full stack|frontend|backend|mobile app|android|ios|swift|kotlin|objective c|flutter|react native|xamarin|unity|unreal engine|game development|vr|ar|augmented reality|virtual reality)\b',
        re.I
    )
    # 1. Extract from skills section
    if k_start != -1:
        for i in range(k_start+1, k_end):
            line = lines[i].strip()
            if not line:
                continue
            parts = re.split(r'[•\u2022\-\*\·,;|\t]', line)
            for part in parts:
                skill = part.strip()
                skill = re.sub(r'[^\w\-\+\# ]', '', skill)
                if skill and len(skill) < 40 and skill_pattern.search(skill):
                    skills.append(skill)
    # 2. Also extract from summary and experience sections
    for section in ['summary', 'experience']:
        section_lines = []
        if section == 'summary':
            s_start, s_end = find_section_bounds(norm_lines, 'summary')
            section_lines = lines[s_start+1:s_end] if s_start != -1 else lines[:10]
        elif section == 'experience':
            e_start, e_end = find_section_bounds(norm_lines, 'experience')
            section_lines = lines[e_start+1:e_end] if e_start != -1 else []
        for line in section_lines:
            for skill in skill_pattern.findall(line):
                skill = skill.strip()
                skill = re.sub(r'[^\w\-\+\# ]', '', skill)
                if skill and len(skill) < 40 and skill.lower() not in [s.lower() for s in skills]:
                    skills.append(skill)
    # Remove duplicates, keep order
    skills = list(dict.fromkeys([s for s in skills if s]))
    return ', '.join(skills) if skills else 'Not detected'

def extract_section_keywords(parsed):
    # Only extract technical skills, tools, job titles, research areas
    section_keywords = []
    technical_pattern = re.compile(
        r'\b('
        r'python|java|c\+\+|c#|sql|mysql|postgresql|oracle|oracle sql|mongodb|firebase|'
        r'machine learning|deep learning|data science|nlp|llm|transformers|huggingface|tensorflow|keras|pytorch|'
        r'scikit-learn|pandas|numpy|matplotlib|seaborn|openai|chatgpt|open cv|computer vision|'
        r'linux|unix|bash|shell scripting|'
        r'cloud|aws|azure|gcp|cloud computing|'
        r'devops|docker|kubernetes|jenkins|terraform|ansible|'
        r'html|css|javascript|typescript|react|angular|vue|next js|node js|express|php|flask|django|bootstrap|tailwind|'
        r'fpga|vhdl|verilog|matlab|simulink|microcontroller|embedded|pcb|arduino|raspberry pi|iot|'
        r'solidworks|autocad|ansys|catia|hvac|thermodynamics|fea|cfd|mechanical design|'
        r'excel|power bi|tableau|data visualization|data analysis|statistics|analytics|'
        r'erp|sap|odoo|quickbooks|crm|zoho|project management|scrum|agile|jira|kanban|'
        r'photoshop|illustrator|figma|canva|ui/ux|adobe xd|after effects|animation|branding|'
        r'professor|lecturer|assistant|instructor|teacher|educator|trainer|'
        r'engineer|developer|scientist|researcher|manager|consultant|analyst|administrator|coordinator|specialist|'
        r'director|head|lead|officer|strategist|intern|student|trainee'
        r')\b',
        re.I
    )
    for section in ['skills', 'experience', 'education', 'summary']:
        content = parsed.get(section, "")
        tokens = re.split(r'[,\|\;/\n\-\u2022\*\(\)\[\]\.]+', content)
        for token in tokens:
            token = token.strip(" .,-;:()[]")
            token = re.sub(r'[^a-zA-Z0-9\-\+\# ]', '', token)
            # Remove tokens that are just years or numbers
            if re.fullmatch(r'\d{4}', token) or re.fullmatch(r'\d{4}\s*-\s*\d{4}', token) or re.fullmatch(r'[\d\s\-–to]+', token, re.I):
                continue
            if 2 < len(token) < 40 and technical_pattern.search(token):
                if token and token.lower() not in [k.lower() for k in section_keywords]:
                    section_keywords.append(token)
    return section_keywords

# Define keywords for each category/domain
CATEGORY_KEYWORDS = {
    "COMPUTER_SCIENCE": [
        "python", "java", "c++", "c#", "sql", "machine learning", "deep learning", "nlp", "data science",
        "object-oriented programming", "oop", "algorithms", "data structures", "software engineering",
        "web development", "django", "flask", "react", "angular", "cloud", "aws", "azure", "linux", "git"
    ],
    "ELECTRICAL_ENGINEERING": [
        "fpga", "vhdl", "verilog", "embedded", "microcontroller", "pcb", "arduino", "matlab", "simulink",
        "circuit", "signal processing", "power systems", "electronics", "control systems", "raspberry pi"
    ],
    "MECHANICAL_ENGINEERING": [
        "solidworks", "autocad", "catia", "ansys", "hvac", "thermodynamics", "mechanical", "fea", "cfd",
        "manufacturing", "cad", "cam", "mechatronics", "proe", "creo"
    ],
    "MANAGEMENT": [
        "project management", "crm", "erp", "business intelligence", "sap", "oracle erp", "leadership",
        "team management", "strategy", "operations", "marketing", "finance", "sales", "analytics"
    ],
      "BBA_AND_BUSINESS": [
        "bba", "business administration", "management", "marketing", "finance", "hr", "human resources",
        "entrepreneurship", "operations", "strategy", "sales", "customer relationship", "crm",
        "analytics", "organizational behavior", "project management", "leadership"
    ],
    "CIVIL_ENGINEERING": [
        "civil engineering", "autocad", "staad pro", "etabs", "surveying", "structural analysis",
        "site engineer", "construction", "foundation design", "concrete", "urban planning", "revit"
    ],
    "CHEMICAL_ENGINEERING": [
        "chemical engineering", "process engineering", "reaction engineering", "heat exchanger",
        "mass transfer", "catalyst", "thermodynamics", "distillation", "petrochemicals", "plant design"
    ],
    "BIOMEDICAL_ENGINEERING": [
        "biomedical", "medical devices", "bioinstrumentation", "clinical engineering",
        "rehabilitation engineering", "imaging systems", "biosensors", "biomaterials"
    ],
    "INDUSTRIAL_ENGINEERING": [
        "industrial engineering", "operations research", "lean manufacturing", "six sigma",
        "supply chain", "production planning", "workflow optimization", "erp", "inventory management"
    ],
    "ENVIRONMENTAL_ENGINEERING": [
        "environmental engineering", "wastewater", "air pollution", "sustainability",
        "climate change", "water treatment", "solid waste", "environmental impact", "green design"
    ],
    "HR_AND_TALENT_MANAGEMENT": [
        "human resources", "recruitment", "talent acquisition", "performance management",
        "training and development", "employee relations", "hrms", "benefits", "hr policies"
    ],
    "ECONOMICS_AND_RESEARCH": [
        "economics", "market research", "data interpretation", "macroeconomics", "microeconomics",
        "econometrics", "policy analysis", "statistics", "financial markets"
    ],
    "LAW_AND_LEGAL": [
        "law", "legal", "contract", "corporate law", "intellectual property", "litigation",
        "legal research", "compliance", "civil law", "criminal law", "regulatory affairs"
    ],
     "DATA_ANALYTICS": [
        "data analysis", "excel", "power bi", "tableau", "data visualization", "data cleaning",
        "sql", "python", "statistics", "pandas", "numpy", "etl", "dashboards", "insights", "kpi"
    ],
    "ARTIFICIAL_INTELLIGENCE": [
        "neural networks", "machine learning", "deep learning", "computer vision", "tensorflow",
        "keras", "pytorch", "ai", "nlp", "reinforcement learning", "generative ai", "llm"
    ],
    "CIVIL_ENGINEERING": [
        "autocad", "staad pro", "etabs", "construction", "structural design", "surveying", "concrete",
        "civil 3d", "building materials", "roads", "bridges", "site engineer", "architecture"
    ],
    "HEALTHCARE_AND_BIOLOGY": [
        "clinical research", "biotechnology", "bioinformatics", "healthcare", "pharma", "medical",
        "molecular biology", "laboratory", "diagnostics", "patient care", "drug development"
    ],
    "GRAPHIC_DESIGN_AND_MEDIA": [
        "photoshop", "illustrator", "canva", "figma", "adobe xd", "ui/ux", "branding", "animation",
        "video editing", "after effects", "logo design", "infographics", "poster design"
    ],
    "EDUCATION_AND_TRAINING": [
        "teaching", "lesson planning", "curriculum", "educator", "training", "pedagogy",
        "learning management system", "tutoring", "classroom", "student engagement"
    ],
    "ACCOUNTING_AND_FINANCE": [
        "accounting", "bookkeeping", "financial analysis", "excel", "quickbooks", "taxation",
        "auditing", "budgeting", "cost accounting", "payroll", "erp", "sap", "tally"
    ],
    "UNKNOWN": []
}

def detect_category_and_confidence(parsed):
    # Use all sections (including summary) for domain scoring
    all_sections = {
        "skills": parsed.get('skills', '').lower(),
        "experience": parsed.get('experience', '').lower(),
        "education": parsed.get('education', '').lower(),
        "summary": parsed.get('summary', '').lower()
    }
    boost_terms = [
            # 💻 Computer Science & Software
        "data structures", "algorithms", "object-oriented programming", "oop", "design patterns",
        "python", "java", "c++", "c#", "software engineering", "git", "github", "version control",
        "sql", "mysql", "postgresql", "mongodb", "firebase", "api development", "rest api", "oop principles",

        # 🤖 AI / ML / Data Science
        "machine learning", "deep learning", "ml", "dl", "nlp", "natural language processing",
        "neural networks", "tensorflow", "keras", "pytorch", "huggingface", "transformers", "llm",
        "data analysis", "exploratory data analysis", "data visualization", "pandas", "numpy", 
        "matplotlib", "seaborn", "jupyter notebook", "statistics", "prediction", "classification",

        # ☁️ DevOps / Cloud / Tools
        "docker", "kubernetes", "devops", "aws", "azure", "gcp", "cloud computing", "ci/cd",
        "jenkins", "terraform", "linux", "bash", "shell scripting", "terminal", "command line",

        # ⚙️ Engineering Fields

        ## Electrical / Embedded / Robotics
        "fpga", "vhdl", "verilog", "embedded systems", "microcontroller", "arduino", 
        "raspberry pi", "pcb", "matlab", "simulink", "electronics", "circuit design",
        
        ## Mechanical / Civil / Industrial
        "solidworks", "autocad", "ansys", "catia", "fea", "cfd", "hvac", "thermodynamics",
        "construction", "structural analysis", "revit", "staad pro", "manufacturing", "mechatronics",
        "lean manufacturing", "six sigma", "supply chain", "operations research",

        # 💼 Business / Management / HR / BBA
        "bba", "mba", "business administration", "marketing", "finance", "hr", "human resources",
        "project management", "crm", "erp", "sap", "leadership", "team management", "strategic planning",
        "analytics", "data-driven", "operations", "budgeting", "training and development", "performance management",

        # 🎨 UI/UX / Graphic Design / Media
        "ui/ux", "user experience", "user interface", "figma", "adobe xd", "photoshop", "illustrator",
        "canva", "after effects", "branding", "wireframing", "prototyping", "graphic design", "animation",

        # 🧠 Soft Skills (Universal Gold 💛)
        "problem solving", "critical thinking", "communication", "leadership", "teamwork",
        "adaptability", "collaboration", "attention to detail", "creativity", "time management",
        "self-motivated", "decision making"

    ]
    category_scores = {}
    section_weights = {"skills": 5, "experience": 5, "education": 2, "summary": 3}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        total = 0
        for section, weight in section_weights.items():
            section_content = all_sections[section]
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', section_content, re.IGNORECASE):
                    if kw.lower() in boost_terms or len(kw.split()) > 1:
                        score += weight * 3
                    else:
                        score += weight
            total += weight * len(keywords)
        category_scores[category] = score / (total + 1e-6)
    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]
    # Avoid UNKNOWN unless truly no match
    if best_score < 0.02:
        best_category = "UNKNOWN"
    # Confidence: scale to 70-99% based on score and completeness
    completeness = sum(1 for v in all_sections.values() if v and v != 'not detected')
    if best_score > 0.20 and completeness >= 3:
        confidence = 92 + int((best_score - 0.20) * 20)
    elif best_score > 0.10:
        confidence = 80 + int((best_score - 0.10) * 100)
    else:
        confidence = 70 + int(best_score * 100)
    return best_category, min(confidence, 99)

def extract_top_keywords(parsed, category):
    # Always return at least 10-15 meaningful, cleaned keywords
    all_keywords = extract_section_keywords(parsed)
    cat_keywords = set([kw.lower() for kw in CATEGORY_KEYWORDS.get(category, [])])
    top_keywords = []
    seen = set()
    for kw in all_keywords:
        kw_clean = kw.lower()
        if kw_clean in cat_keywords and kw_clean not in seen:
            top_keywords.append(kw)
            seen.add(kw_clean)
    for kw in all_keywords:
        kw_clean = kw.lower()
        if kw_clean not in seen:
            top_keywords.append(kw)
            seen.add(kw_clean)
        if len(top_keywords) >= 15:
            break
    top_keywords = [re.sub(r'[^\w\-\+\# ]', '', k).strip() for k in top_keywords]
    top_keywords = [k for k in top_keywords if len(k) > 2 and not re.match(r'^\d{4}$', k)]
    return top_keywords[:15] if len(top_keywords) >= 10 else top_keywords

def spacy_cleaner(text):
    """
    Cleans and lemmatizes text using the loaded spaCy model.
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def parse_resume(text):
    """
    Parses the resume text and extracts summary, опыта, skills, and education sections.
    """
    lines = [line.strip('\r\n') for line in text.splitlines()]
    norm_lines = [normalize_heading(line) for line in lines]
    name, contact = extract_name_and_contact(lines)
    summary = extract_summary(lines, norm_lines)
    experience = extract_experience(lines, norm_lines)
    skills = extract_skills(lines, norm_lines)
    # Education extraction: similar to summary/experience
    edu_start, edu_end = find_section_bounds(norm_lines, 'education')
    if edu_start != -1:
        education_lines = [lines[i].strip() for i in range(edu_start+1, edu_end) if lines[i].strip()]
        education = ' '.join(education_lines).strip()
    else:
        education = 'Not detected'
    return {
        'name': name if name else 'Not detected',
        'contact': contact if contact else 'Not detected',
        'summary': summary,
        'experience': experience,
        'skills': skills,
        'education': education
    }

def render_ats_html(parsed, category, confidence, keywords):
    # Use .replace('\n', '<br>') outside the f-string to avoid backslash in f-string expressions
    summary = (parsed.get("summary", "Not detected") or "").replace('\n', '<br>')
    experience = (parsed.get("experience", "Not detected") or "").replace('\n', '<br>')
    skills = (parsed.get("skills", "Not detected") or "").replace('\n', '<br>')
    education = (parsed.get("education", "Not detected") or "").replace('\n', '<br>')
    html = (
        "<html>"
        "<head>"
        "<meta charset='utf-8'>"
        "<style>"
        "body { font-family: Arial, sans-serif; font-size: 13px; color: #222; margin: 1.2in 0.8in; }"
        "h1 { font-size: 1.7em; margin-bottom: 0.2em; }"
        "h2 { font-size: 1.2em; margin-top: 1.2em; margin-bottom: 0.2em; border-bottom: 1px solid #bbb; }"
        ".field-label { font-weight: bold; width: 120px; display: inline-block; }"
        ".section { margin-bottom: 1.1em; }"
        ".keywords { color: #222; font-weight: bold; }"
        ".block { margin-bottom: 0.7em; }"
        "</style>"
        "</head>"
        "<body>"
        "<h1>ATS Resume Analysis Report</h1>"
        "<div class='section'>"
        f"<span class='field-label'>Name:</span> {parsed.get('name', 'Not detected')}<br>"
        f"<span class='field-label'>Contact:</span> {parsed.get('contact', 'Not detected')}<br>"
        f"<span class='field-label'>Category:</span> {category}<br>"
        f"<span class='field-label'>Confidence:</span> {confidence:.1f}%<br>"
        f"<span class='field-label'>Top Keywords:</span> <span class='keywords'>{', '.join(keywords)}</span>"
        "</div>"
        "<div class='section'>"
        "<h2>Summary</h2>"
        f"<div class='block'>{summary}</div>"
        "</div>"
        "<div class='section'>"
        "<h2>Experience</h2>"
        f"<div class='block'>{experience}</div>"
        "</div>"
        "<div class='section'>"
        "<h2>Skills</h2>"
        f"<div class='block'>{skills}</div>"
        "</div>"
        "<div class='section'>"
        "<h2>Education</h2>"
        f"<div class='block'>{education}</div>"
        "</div>"
        "</body>"
        "</html>"
    )
    return html

def generate_ats_pdf(parsed, category, confidence, keywords, username=None):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x = 1 * inch
    y = height - 1 * inch

    def draw_heading(text):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, text)
        y -= 0.28 * inch

    def draw_field(label, value):
        nonlocal y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"{label}:")
        c.setFont("Helvetica", 11)
        c.drawString(x + 1.2*inch, y, value if value else "Not detected")
        y -= 0.22 * inch

    def draw_multiline(label, value):
        nonlocal y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"{label}:")
        y -= 0.18 * inch
        c.setFont("Helvetica", 11)
        for line in (value or "Not detected").split('\n'):
            for subline in [line[i:i+90] for i in range(0, len(line), 90)]:
                c.drawString(x + 0.3*inch, y, subline)
                y -= 0.18 * inch
        y -= 0.08 * inch

    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "ATS Resume Analysis Report")
    y -= 0.4 * inch

    draw_field("Name", parsed.get("name", ""))
    draw_field("Contact", parsed.get("contact", ""))
    draw_field("Category", category)
    draw_field("Confidence", f"{confidence:.1f}%")
    draw_field("Top Keywords", ', '.join(keywords))

    y -= 0.15 * inch
    draw_heading("Summary")
    draw_multiline("", parsed.get("summary", ""))

    draw_heading("Experience")
    draw_multiline("", parsed.get("experience", ""))

    draw_heading("Skills")
    draw_multiline("", parsed.get("skills", ""))

    draw_heading("Education")
    draw_multiline("", parsed.get("education", ""))

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Create Flask app instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.secret_key = 'supersecretkey'  # Needed for session

@app.route('/')
def index():
    return render_template('index.html')

def is_valid_email(email):
    """Simple email validation using regex."""
    return bool(re.match(r"^[^@]+@[^@]+\.[^@]+$", email))

# Merged signup function handling both GET/POST and AJAX/non-AJAX requests
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        phone = request.form.get('phone')  # May be None if not sent
        name = f"{first_name} {last_name}".strip()

        # Validate email format
        if not is_valid_email(email):
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.form.get('ajax'):
                return jsonify({'success': False, 'message': 'Invalid email format.'})
            return render_template('index.html', signup_error="Invalid email format.")

        # Save user credentials
        user_id = save_user_credentials(email, password)
        if not user_id:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.form.get('ajax'):
                return jsonify({'success': False, 'message': 'Signup failed. Email may already exist.'})
            return render_template('index.html', signup_error="Signup failed. Email may already exist.")

        # If resume_result exists, extract contact info from parsed resume
        if session.get('resume_result'):
            parsed = session['resume_result']['parsed']
            # Prefer extracted name/contact/phone from resume if available
            extracted_name = parsed.get('name') if parsed.get('name') and parsed.get('name') != 'Not detected' else name
            extracted_contact = parsed.get('contact') if parsed.get('contact') and parsed.get('contact') != 'Not detected' else email
            # Try to extract phone from contact if possible
            phone_from_resume = None
            contact_val = parsed.get('contact', '')
            phone_match = re.search(r'(\+?\d[\d\s\-]{7,}\d)', contact_val)
            if phone_match:
                phone_from_resume = phone_match.group(1)
            extracted_phone = phone_from_resume if phone_from_resume else phone
            save_user_info(user_id, extracted_name, extracted_contact, extracted_phone, file_path=session['resume_result'].get('filepath'))
            save_report_to_db(user_id)
        else:
            # No resume uploaded yet, save provided info
            save_user_info(user_id, name, email, phone, file_path=None)

        # Set session
        session['user'] = user_id

        # Response handling
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.form.get('ajax'):
            return jsonify({'success': True, 'user_id': user_id})
        return redirect(url_for('results'))
    
    # GET request – show signup page
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return "Invalid file", 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    ext = filename.rsplit('.', 1)[1].lower()
    # Extract text from the uploaded file
    text = extract_text(save_path, ext)
    # Parse the resume using your NLP logic
    parsed = parse_resume(text)
    # Detect category and confidence
    category, confidence = detect_category_and_confidence(parsed)
    # Extract top keywords
    keywords = extract_top_keywords(parsed, category)

    # Extract name, email, and phone from parsed/contact using regex
    extracted_name = parsed.get('name') if parsed.get('name') and parsed.get('name') != 'Not detected' else None
    extracted_contact = parsed.get('contact') if parsed.get('contact') and parsed.get('contact') != 'Not detected' else None
    extracted_email = None
    extracted_phone = None
    if extracted_contact:
        # Extract email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', extracted_contact)
        if email_match:
            extracted_email = email_match.group(0)
        # Extract phone (accepts various formats)
        phone_match = re.search(r'(\+?\d[\d\s\-().]{7,}\d)', extracted_contact)
        if phone_match:
            extracted_phone = phone_match.group(1)
    # Fallback: try to find email/phone in the whole text if not found in contact
    if not extracted_email:
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        if email_match:
            extracted_email = email_match.group(0)
    if not extracted_phone:
        phone_match = re.search(r'(\+?\d[\d\s\-().]{7,}\d)', text)
        if phone_match:
            extracted_phone = phone_match.group(1)

    # Store all data in session for results page
    session['resume_result'] = {
        'filename': filename,
        'filepath': save_path,
        'parsed': parsed,
        'category': category,
        'confidence': confidence,
        'keywords': keywords
    }

    # If user is logged in, update user_info with extracted data from resume
    if session.get('user'):
        user_id = session['user']
        save_user_info(
            user_id,
            extracted_name,
            extracted_email,
            extracted_phone,
            file_path=save_path
        )
        save_report_to_db(user_id)

    return redirect(url_for('results'))

@app.route('/results')
def results():
    result = session.get('resume_result')
    if not result:
        return redirect(url_for('index'))
    return render_template(
        'results.html',
        filename=result['filename'],
        parsed=result['parsed'],
        category=result['category'],
        confidence=result['confidence'],
        keywords=result['keywords']
    )

@app.route('/serve_file/<filename>')
def serve_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/download_report', methods=['GET'])
def download_report():
    # Only allow download if user is logged in
    if not session.get('user'):
        return '', 401
    result = session.get('resume_result')
    if not result:
        return '', 400
    # Generate PDF (replace with your actual PDF generation logic)
    pdf_path = os.path.join(UPLOAD_FOLDER, f"report_{result['filename']}.pdf")
    # For demo, just send the original file
    return send_file(result['filepath'], as_attachment=True, download_name="ATS_Report.pdf")

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    user_id = verify_user_credentials(email, password)

    if user_id:
        session['user'] = user_id

        # Save user/report data to DB after login (if available)
        if session.get('resume_result'):
            save_report_to_db(user_id)

        # If it's an AJAX request, return JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.form.get('ajax'):
            return jsonify({'success': True})

        # Fallback (not AJAX, just in case)
        return jsonify({'success': True})
    
    else:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.form.get('ajax'):
            return jsonify({'success': False, 'message': 'Invalid email or password.'})
        return jsonify({'success': False, 'message': 'Invalid email or password.'})


def save_report_to_db(user_id):
    result = session.get('resume_result')
    if not result:
        return
    parsed = result['parsed']
    save_user_info(user_id, parsed.get('name'), parsed.get('contact'), None, result['filepath'])
    save_resume_data(
        user_id,
        parsed.get('skills'),
        'BSc Computer Science',  # Dummy education
        parsed.get('experience'),
        result.get('category'),
        result.get('confidence')
    )

@app.route('/download_original')
def download_original():
    if not session.get('user'):
        # Return 401 so JS can open modal, do not redirect
        return '', 401
    analysis = session.get('last_analysis')
    if not analysis:
        return "No analysis found. Please analyze a resume first.", 400
    parsed = analysis['parsed']
    category = analysis['category']
    confidence = analysis['confidence']
    keywords = analysis['keywords']
    username = parsed.get("name", "User").replace(" ", "_")
    html = render_ats_html(parsed, category, confidence, keywords)
    # --- FIX: Use reportlab fallback if wkhtmltopdf is not installed ---
    try:
        # Try to use pdfkit (wkhtmltopdf)
        pdf = pdfkit.from_string(html, False, options={
            'quiet': '',
            'enable-local-file-access': '',
            'page-size': 'A4',
            'encoding': "UTF-8"
        })
        filename = f"ATS_Report_{username}.pdf"
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except OSError:
        # Fallback to reportlab PDF (no HTML styling, but always works)
        buffer = generate_ats_pdf(parsed, category, confidence, keywords, username=username)
        filename = f"ATS_Report_{username}.pdf"
        response = make_response(buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

if __name__ == '__main__':
    app.run(debug=True)