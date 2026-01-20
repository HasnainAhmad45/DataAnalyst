# 🚀 Render Deployment Guide

## Quick Deploy to Render

### 1. **Configure Render Service**

In your Render dashboard:

**Service Type:** Web Service

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
```

### 2. **Environment Variables**

Add these in Render dashboard under "Environment":

```
GEMINI_API_KEY=your_api_key_here
PYTHON_VERSION=3.11.9
```

### 3. **Advanced Settings**

- **Auto-Deploy:** Yes
- **Health Check Path:** `/api/health`
- **Instance Type:** Starter (or higher for production)

### 4. **Deploy**

Click "Create Web Service" and Render will:
1. Clone your repo
2. Install dependencies
3. Start the application

Your app will be live at: `https://your-app-name.onrender.com`

---

## Troubleshooting

### Build Fails
- Check that `runtime.txt` specifies `python-3.11.9`
- Verify all environment variables are set
- Review build logs for specific errors

### App Crashes
- Check application logs in Render dashboard
- Ensure `GEMINI_API_KEY` is set correctly
- Verify database credentials if using external MySQL

### Slow Performance
- Upgrade to a paid instance type
- Increase worker count in start command
- Enable persistent disk for uploads

---

## Local Testing

Test the production setup locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
```

---

## Database Setup

For production MySQL:

1. Create a managed MySQL database (Render, AWS RDS, etc.)
2. Add credentials as environment variables:
   ```
   DB_HOST=your-db-host
   DB_PORT=3306
   DB_USER=your-user
   DB_PASSWORD=your-password
   DB_NAME=your-database
   ```
3. Users can still override via UI when connecting

---

## Monitoring

- **Health Check:** `GET /api/health`
- **Logs:** View in Render dashboard
- **Metrics:** Enable in Render settings

---

## Scaling

To handle more traffic:

1. Increase workers: `--workers 4`
2. Upgrade instance type
3. Add Redis for session storage
4. Use CDN for static files
