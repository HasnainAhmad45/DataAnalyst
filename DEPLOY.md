# Render Deployment - Quick Setup

## In Render Dashboard

### 1. Create Web Service
- **Repository**: Connect your GitHub repo
- **Branch**: main
- **Root Directory**: (leave blank)

### 2. Build Settings
**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
gunicorn app:app
```

### 3. Environment Variables
Add this:
```
GEMINI_API_KEY=your_api_key_here
```

### 4. Advanced Settings
- **Python Version**: Will use runtime.txt (3.11.5)
- **Auto-Deploy**: Yes
- **Health Check Path**: `/api/health`

### 5. Click "Create Web Service"

That's it! Your app will be live in ~5 minutes.

## Troubleshooting

If build fails, check:
1. `runtime.txt` exists and says `python-3.11.5`
2. `GEMINI_API_KEY` is set in environment variables
3. Build logs for specific error messages

## Local Test

```bash
pip install -r requirements.txt
gunicorn app:app
```

Visit: http://localhost:8000
