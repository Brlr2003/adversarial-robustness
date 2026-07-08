FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Force a light theme so the app (and its screenshots) are readable on B&W print,
# regardless of the viewer's system dark mode.
ENV STREAMLIT_THEME_BASE=light \
    STREAMLIT_THEME_BACKGROUNDCOLOR="#ffffff" \
    STREAMLIT_THEME_SECONDARYBACKGROUNDCOLOR="#f0f2f6" \
    STREAMLIT_THEME_TEXTCOLOR="#1b2631"

# HuggingFace Spaces expects port 7860
CMD ["streamlit", "run", "frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
