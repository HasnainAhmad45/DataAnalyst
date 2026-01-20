# 🔄 PostgreSQL Migration Guide

## Step 1: Get PostgreSQL Connection URL

1. Go to your **Render Dashboard**
2. Click on your **PostgreSQL database**
3. Scroll down to find **"Connections"** section
4. Copy the **External Database URL** (NOT Internal!)
   - External URL looks like: `postgresql://user:password@dpg-xxxxx.oregon-postgres.render.com/database`
   - Internal URL (don't use): `postgresql://user:password@dpg-xxxxx-a/database`

## Step 2: Update Environment Variables

### **In Render Web Service:**

1. Go to your **Web Service** → **Environment** tab
2. Add this variable:
   ```
   DATABASE_URL=postgresql://user:password@host/database
   ```
   (Paste the URL you copied from Step 1)

### **In Local .env File:**

Add to your `.env` file:
```
DATABASE_URL=postgresql://user:password@host/database
DB_PASSWORD=your_local_mysql_password
```

## Step 3: Run Migration Script

```bash
# Install psycopg2 locally
pip install psycopg2-binary

# Run migration
python migrate_to_postgres.py
```

## Step 4: Update App Configuration

Your app now supports **both MySQL and PostgreSQL**!

### **For PostgreSQL (Render):**
The app will automatically use `DATABASE_URL` from environment variables.

### **For MySQL (Local/Custom):**
Users can still connect via the UI by entering credentials.

## Step 5: Deploy Updated Code

```bash
git add .
git commit -m "Add PostgreSQL support"
git push origin main
```

Render will auto-deploy with PostgreSQL support!

## 🎯 What Changed

- ✅ Added `psycopg2-binary` to requirements
- ✅ App now supports both MySQL and PostgreSQL
- ✅ Migration script to transfer data
- ✅ Environment variable configuration

## 🔧 Troubleshooting

**Migration fails:**
- Check `DATABASE_URL` is correct
- Verify local MySQL is running
- Update `DB_PASSWORD` in `.env`

**App can't connect to PostgreSQL:**
- Ensure `DATABASE_URL` is set in Render environment variables
- Check the URL format is correct

## 📊 After Migration

Your deployed app will:
- Use PostgreSQL for database connections
- Still support CSV/PDF uploads
- Allow users to connect custom databases via UI

Done! 🎉
