# 🚀 Adaptive AI Interview Assistant

## 📌 Overview
The **Adaptive AI Interview Assistant** is a Streamlit-based application that conducts AI-driven interviews, evaluates responses using Google's Gemini API, and provides feedback with an ML-based similarity score.

## ✨ Features
- **Adaptive Question Generation**: Generates interview questions based on job title and confidence level.
- **AI-Powered Answer Evaluation**: Uses Google's Gemini API to evaluate answers.
- **ML-Based Scoring**: Calculates similarity scores using TF-IDF and cosine similarity.
- **SQLite Database Integration**: Stores interview details and responses for review.
- **User-Friendly UI**: Built with Streamlit for an interactive experience.

## 🛠️ Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/adaptive-ai-interview.git
cd adaptive-ai-interview
```

### 2️⃣ Setup Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the project root and add:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 5️⃣ Run the Application
```sh
streamlit run app.py
```

## 📂 Project Structure
```
├── app.py                  # Main Streamlit app
├── interview_system.py      # AdaptiveInterviewSystem class
├── .env                    # Environment variables (excluded in Git)
├── .gitignore              # Ignore sensitive files
├── requirements.txt        # Required dependencies
├── README.md               # Documentation
```

## 📝 Usage
1. Enter the **Job Title** and select **Confidence Level**.
2. Answer generated questions.
3. View AI evaluation and similarity score.
4. Review interview rounds and save results.

## 🔒 Security Notes
- Ensure `.env` is **not** committed to GitHub (already added in `.gitignore`).
- Replace the placeholder `GEMINI_API_KEY` with a valid key.

## 📌 Future Enhancements
- Multi-round interview customization.
- Speech-to-text for answer input.
- Integration with other AI models.

## 🤝 Contributing
Feel free to fork this repo, create issues, or submit pull requests!

## 📜 License
This project is licensed under the MIT License.

