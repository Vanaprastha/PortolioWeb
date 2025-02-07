from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from langdetect import detect
import os
import logging

app = Flask(__name__)

# Konfigurasi CORS agar mengizinkan semua metode dan header
CORS(app, resources={r"/chatbot": {"origins": "*"}})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Gemini AI
API_KEY = os.getenv("GEMINI_API_KEY")

# Inisialisasi Gemini API Client
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Semua informasi tentang Vana Prastha
PERSONAL_INFO_ID = """
Nama saya Vana Prastha, atau Anda bisa memanggil saya Atha dalam bahasa Inggris. Saya seorang Data Analyst yang memiliki latar belakang akademik di  Electronic Engineering Polytechnic Institute of Surabaya (EEPIS), dengan fokus pada Data Science. Saya memilih bidang ini karena saya percaya bahwa Data Science tidak akan pernah mati, terutama dengan perkembangan pesat dalam AI dan Big Data. Inspirasi saya berasal dari video Jack Ma yang mengatakan bahwa "Data is the new oil," yang membuat saya yakin bahwa potensi data sangat besar dan akan terus berkembang. Selain itu, saya tertarik dengan bagaimana teknologi berkembang, dan saya ingin menjadi bagian dari revolusi digital ini.

Saya memiliki minat yang mendalam dalam analisis data, pemrograman, dan dunia kreatif. Saya sangat menikmati eksplorasi data, machine learning, serta data visualization, dan bagaimana menghubungkan data dengan solusi bisnis yang nyata. Saya juga sering menonton tutorial di YouTube atau mengikuti kursus online untuk terus memperdalam pemahaman saya di bidang ini. Saya percaya bahwa belajar tidak hanya dilakukan di kelas, tetapi juga dari berbagai sumber yang tersedia secara daring. Selain itu, saya juga menikmati menonton film, drama, dan donghua sebagai bentuk hiburan sekaligus sumber inspirasi. Saya sering membaca artikel di internet dan jurnal akademik untuk terus memperkaya wawasan saya.

Keterampilan teknis saya meliputi pemrograman dalam Python, SQL, dan R, serta database seperti MySQL. Saya memiliki pengalaman dalam visualisasi data menggunakan Tableau, google looker, dan Microsoft power Bi, serta telah bekerja dengan berbagai model machine learning untuk analisis prediktif dan klasifikasi. Saya paling sering menggunakan Google Colab dan VS Code untuk coding, serta sangat terbantu dengan ChatGPT dalam berbagai aspek pekerjaan dan pembelajaran saya. Saya menyukai Pytorch sebagai framework deep learning favorit saya, terutama karena saya menggunakannya dalam proyek CLIP saya. Selain itu, saya juga memiliki pengalaman dalam cloud computing dan big data. Saya telah mempelajari dasar-dasar cloud computing, menggunakan Virtual Machine dan Docker untuk beberapa tugas kuliah, serta memahami konsep Big Data, meskipun masih terkendala oleh keterbatasan perangkat yang mumpuni.

Di dunia akademik, saya telah mengerjakan berbagai proyek yang berfokus pada analisis data dan pemanfaatan kecerdasan buatan. Beberapa di antaranya adalah Membuat program untuk Prediksi kategori berita menggunakan 3 metode model
pembelajaran, melakukan prediksi jenis ikan dan kemungkinan
seseorang membeli laptop berdasarkan data yang diberikan menggunakan model pembelajaran
naive bayes dan bahasa python, membuat program untuk melakukan perangkuman kata dari dokumen dan dapat mengetahui kata yang sering muncul pada dua
dokumen berbeda yang diinputkan oleh pengguna kemudian kata kata yang sering muncul akan
divisualisasikan yang menggunakan model pembelajaran supervised dan unsupervised learning,  melakukan prediksi orang yang terkena penyakit
jantung dan gangguan pada sebuah server dari data yang diberikan dengan menggunakan
bahasa python dan algoritma decision tree.,  melakukan clustering jenis bunga iris dengan 4
metode linkage yang kemudian akan dibandingkan hasilnya, Melakukan analisa clustering berdasarkan pendapatan
tahunan dan profesi pekerjaan dengan menggunakan bahasa pemrograman python dan metode
supervised learning dengan k-mean clustering.
saat ini saya mendapatkan pencapaian berupa beasiswa pemerintah kota surabaya dan beasiswa asrama melalui yayasan majlis amal sholeh. 
Saya memiliki pengalaman di antara nya kegiatan keorganisasiaan maupun volunteer karena tekad saya untuk bisa berkontribusi kembali kepada masyarakat diantaranya organisasi kampus, volunteer mengajar, dan kegiatan sosial lainnya. 

"""

PERSONAL_INFO_EN = """
My name is Vana Prastha, or you can call me Atha in English. I am a Data Analyst with an academic background in Electronic Engineering Polytechnic Institute of Surabaya (EEPIS), focusing on Data Science. I chose this field because I believe that Data Science will never die, especially with the rapid development in AI and Big Data. My inspiration comes from Jack Ma's video saying that “Data is the new oil,” which makes me believe that the potential of data is huge and will continue to grow. In addition, I am interested in how technology is evolving, and I want to be part of this digital revolution.

I have a deep interest in data analysis, programming, and the creative world. I really enjoy exploring data, machine learning, and data visualization, and how to connect data to real business solutions. I also often watch tutorials on YouTube or take online courses to continue deepening my understanding in this field. I believe that learning is not only done in the classroom, but also from various resources available online. In addition, I also enjoy watching movies, dramas and donghua as a form of entertainment as well as a source of inspiration. I often read articles on the internet and academic journals to continue enriching my knowledge.

My technical skills include programming in Python, SQL, and R, as well as databases such as MySQL. I have experience in data visualization using Tableau, google looker, and Microsoft power Bi, and have worked with various machine learning models for predictive analysis and classification. I use Google Colab and VS Code the most for coding, and have found ChatGPT very helpful in various aspects of my work and learning. I like Pytorch as my favorite deep learning framework, especially since I used it in my CLIP project. In addition, I also have experience in cloud computing and big data. I have learned the basics of cloud computing, used Virtual Machine and Docker for some coursework, and understood the concept of Big Data, although I am still constrained by the limitations of qualified devices.

In the academic world, I have worked on various projects that focus on data analysis and the utilization of artificial intelligence. Some of them are Creating a program for news category prediction using 3 methods of learning model
learning model, predicting fish species and the likelihood of
someone buys a laptop based on the given data using naive bayes learning model and python language.
naive bayes and python language, create a program to summarize words from documents and can find out words that often appear in two different documents inputted by users.
documents inputted by the user then the words that often appear will be visualized using the naive bayes learning model and python language.
visualized using supervised and unsupervised learning models, predicting people affected by heart disease and disorders on a server from data.
heart disease and disorders on a server from the given data by using the
python language and decision tree algorithm, clustering iris flower types with 4 linkage methods and then comparing the results.
linkage method which will then be compared the results, Analyzing clustering based on annual income and job profession by using the
and job profession using python programming language and supervised learning method with k-means clustering.
supervised learning method with k-mean clustering.
I currently have achievements in the form of Surabaya city government scholarships and dormitory scholarships through the majlis amal sholeh foundation. 
I have experience in organizational and volunteer activities because of my determination to be able to contribute back to society including campus organizations, volunteer teaching, and other social activities. 
"""

# Handling CORS secara manual jika masih terjadi error
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Endpoint utama untuk mengecek API berjalan
@app.route("/")
def home():
    return "Chatbot API is Running!"

# Endpoint untuk chatbot
@app.route("/chatbot", methods=["POST", "OPTIONS"])
def chatbot():
    # Handle preflight OPTIONS request (agar CORS tidak memblokir POST request)
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight passed"}), 200

    try:
        data = request.get_json()
        user_message = data.get("message")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"User asked: {user_message}")

        # Gunakan AI untuk menjawab pertanyaan
        ai_response = ask_gemini(user_message)
        return jsonify({"response": ai_response})

    except Exception as e:
        logger.error(f"Error in server: {e}")
        return jsonify({"error": "Terjadi kesalahan pada server."}), 500

# Fungsi untuk mendeteksi bahasa
def detect_language(text):
    try:
        lang = detect(text)
        return "en" if lang == "en" else "id"
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return "id"

# Fungsi untuk memproses pertanyaan dengan Gemini AI
def ask_gemini(question):
    try:
        lang = detect_language(question)
        info = PERSONAL_INFO_EN if lang == "en" else PERSONAL_INFO_ID

        # Handle greetings
        greetings = ["hi", "hello", "hey", "hola", "hai", "halo"]
        if question.lower().strip() in greetings:
            return "Hello! How can I assist you today?" if lang == "en" else "Halo! Ada yang bisa saya bantu?"

        prompt = f"""
        You are Atha, Vana Prastha's personal chatbot. Your answers should be based on the following information:

        {info}

        If the question is not related to the information above, provide a neutral answer or help the user with general information. Your answer should be accurate, professional, and easy to understand.

        **User question:** {question}
        """

        response = model.generate_content(prompt)

        if response and hasattr(response, 'text') and response.text.strip():
            return response.text.strip()
        else:
            return "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan itu." if lang == "id" else "Sorry, I don't have enough information to answer that question."

    except Exception as e:
        logger.error(f"Error in Gemini API: {e}")
        return "Maaf, saya mengalami kendala teknis dalam menjawab pertanyaan ini." if lang == "id" else "Sorry, I am experiencing a technical issue in answering this question."

# Konfigurasi Gunicorn untuk Render
if __name__ != "__main__":
    gunicorn_app = app
