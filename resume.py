import streamlit as st
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from streamlit import markdown

#helper function

def read_pdf(file):
    text = extract_text(file)
    return text
#streamlit App

st.set_page_config(page_title="Resume Ranker AI", layout="centered")
st.title("📝Resume Ranker AI")
st.write("Upload your **Resume** and a **Job Description (JD)** to check how well they match.")
#upload a resume
resume_file= st.file_uploader("Upload Job Description (PDF or .txt)", type=["pdf","txt"])
#upload job description
jd_file = st.file_uploader("Upload job description (pdf or .txt)",type=["pdf",".txt"])
# Read & Display Contents
if jd_file and resume_file:
    #read resume
    if resume_file.type == "application/pdf":
        resume_text = read_pdf(resume_file)
    else:
        resume_text = resume_file.read().decode("utf-8")
    # Read JD
    if jd_file.type == "application/pdf":
        jd_text = read_pdf(jd_file)
    else:
        jd_text = jd_file.decode("utf-8")
    st.subheader("Your Resume Text:")
    st.text_area("Resume",resume_text,height=250)
    st.subheader("Your Job description text:")
    st.text_area("Job Description",jd_text,height=250)
    st.success("Both Files Loaded! Ready for Matching.")
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

    # Compute Cosine Similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percent = round(similarity_score * 100, 2)

    st.subheader("🔍 Match Score:")
    st.markdown(f"<h2 style='color: green;'>{match_percent}% match</h2>", unsafe_allow_html=True)

    # Simple interpretation
    if match_percent >= 75:
        st.success("✅ Strong match! You're well aligned with the JD.")
    elif match_percent >= 50:
        st.warning("⚠️ Moderate match. Consider updating some skills/keywords.")
    else:
        st.error("❌ Low match. Try tailoring your resume better to the JD.")
    
    nltk.download('punkt')
    nltk.download('stopwords')

    # --- Keyword Analysis ---
    def extract_keywords(text):
    # Use regex to split into words instead of nltk.word_tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Keep only alphanumeric
        tokens = [word for word in tokens if word not in stopwords.words("english")]
        return Counter(tokens)

    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    matched_keywords = list(set(resume_keywords) & set(jd_keywords))
    missing_keywords = list(set(jd_keywords) - set(resume_keywords))

    st.subheader("✅ Matched Keywords:")
    st.write(", ".join(sorted(matched_keywords)))

    st.subheader("❌ Missing Important Keywords from Resume:")
    if missing_keywords:
        st.warning(", ".join(sorted(missing_keywords[:20])))
    else:
        st.success("Your resume covers all major keywords!")
    

    st.subheader("📌 Smart Suggestions")

    # Filter and rank important missing keywords
    missing_counter = Counter([word for word in missing_keywords if len(word) > 3])
    top_missing = missing_counter.most_common(10)

    if top_missing:
        st.info("🎯 Your resume is missing the following **important keywords** from the JD. Try to include them:")

        # Stylish box layout
        st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 10px;'>", unsafe_allow_html=True)

        for word, freq in top_missing:
            badge_html = f"""
            <div style="
                background-color: #f0f8ff;
                border-radius: 10px;
                padding: 8px 15px;
                font-size: 16px;
                color: #333;
                border: 1px solid #d3d3d3;
                box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
            ">
                🔍 <b>{word}</b> <span style="color:#666;">({freq}× in JD)</span>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.success("✅ Your resume already covers all key terms in the job description!")


