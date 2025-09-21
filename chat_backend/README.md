# Chat with Resume Backend

## Setup
1. Place your resume PDF as `resume.pdf` in this folder (or update `RESUME_PATH` in `.env`).
2. Add your Gemini API key to `.env` as `GEMINI_API_KEY`.
3. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Run the server:
   ```powershell
   python app.py
   ```

## API
- POST `/chat` with JSON `{ "message": "your question" }`
- Returns `{ "answer": "..." }`

## Note
- Gemini API integration is a placeholder. Replace `ask_gemini` in `app.py` with real API logic.
