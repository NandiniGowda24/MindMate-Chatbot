# MindMate - AI-Powered Mental Health Support Chatbot

MindMate is an AI-driven mental health support application designed to provide empathetic conversations, emotional check-ins, and personalized self-care recommendations. It combines natural language processing (NLP) techniques with an advanced fallback model to ensure meaningful and supportive interactions.

---

## Prototype
<img width="853" height="772" alt="image" src="https://github.com/user-attachments/assets/714033af-bdf7-4ea0-a630-49399a1bcca2" />


## 🛠️ Tech Stack

**Frontend:** HTML, CSS, JavaScript  
**Backend:** Python (Flask / FastAPI)  
**Techniques & Algorithms:**  
- NLP (TF-IDF, Vectorizer)  
- Logistic Regression  
**Fallback Model:** Ollama (gemma:2b)  

---

## 📊 Features

### Chatbot & Emotional Support
- NLP-powered chatbot for empathetic conversations.
- Sentiment analysis to gauge user mood and provide appropriate responses.
- Mood journaling and emotional check-ins.

### Recommendations & Self-Care
- Breathing exercises, meditation, and self-care tips.
- Personalized advice based on user sentiment.

### Crisis & Professional Help
- Integration with crisis helplines for urgent support.
- Community platform for peer support and motivation.
- Option to find nearby therapists if professional help is needed.

---

## 📂 Dataset
- **Sentiment Analysis for Mental Health** (Kaggle)  
- Data used for training the chatbot to respond empathetically to user inputs.

---

## ⚙️ How It Works
1. User interacts with the chatbot via a web interface.
2. Sentiment of the input is analyzed using NLP techniques.
3. If the model cannot generate a confident response, the fallback LLM (`Ollama gemma:2b`) is used.
4. Based on sentiment and context, the system provides:
   - Textual advice
   - Mood journaling prompts
   - Self-care recommendations
5. For serious emotional distress, the user is guided to crisis helplines or professional help.

---

## View the demo here: demoo.mp4