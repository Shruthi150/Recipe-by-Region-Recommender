# Deployment Guide: Recipe by Region Recommender

This guide provides step-by-step instructions to deploy the Recipe by Region Recommender to **Streamlit Cloud** (recommended) or **Hugging Face Spaces**.

---

## Option 1: Deploy to Streamlit Cloud (Recommended) ⭐

### Prerequisites
- GitHub account (https://github.com)
- Streamlit Community Cloud account (free)

### Step 1: Push Code to GitHub

If you haven't already, initialize and push your project to GitHub:

```bash
cd "c:\Users\Shruthi.Bathini\Desktop\Statistical Methods in AI\Recipe by Region Recommender"

# Initialize git (skip if already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Recipe by Region Recommender"

# Add remote repository (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/recipe-recommender.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://streamlit.io/cloud
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the deployment form**:
   - **Repository**: `YOUR_USERNAME/recipe-recommender`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): Leave blank for auto-generated, or choose custom

5. **Click "Deploy"**

**Deployment will take 2-5 minutes.** You'll get a live URL like:
```
https://your-app-name.streamlit.app
```

### Troubleshooting Streamlit Cloud

| Issue | Solution |
|-------|----------|
| App crashes on startup | Check `requirements.txt` - may need to pin versions: `scikit-learn==1.3.0` |
| CSV file not found | Ensure `Data/indian_food.csv` is committed to GitHub |
| Images don't load | Check paths use `/` not `\` (Windows vs. Unix paths) |
| Model takes too long to load | Reduce dataset or optimize preprocessing |

---

## Option 2: Deploy to Hugging Face Spaces

### Prerequisites
- Hugging Face account (https://huggingface.co)
- GitHub account (for public repo connection)

### Step 1: Create a Hugging Face Space

1. **Go to Hugging Face Spaces**: https://huggingface.co/spaces
2. **Click "Create new Space"**
3. **Fill in the form**:
   - **Space name**: `recipe-recommender`
   - **Space type**: `Streamlit`
   - **Visibility**: `Public` (for submission link)
   - **Select a license**: `MIT` (recommended)

4. **Click "Create Space"**

### Step 2: Connect GitHub Repository

1. **In your Hugging Face Space**, go to **"⚙️ Settings"**
2. **Scroll to "Linked Model/Space"** section
3. **Link your GitHub repository**:
   - Enter your repo URL: `https://github.com/YOUR_USERNAME/recipe-recommender`
   - HF will auto-sync main branch

### Step 3: Verify Deployment

- HF Spaces will automatically build and deploy your app
- Check **"Building" → "Running"** status in the Space
- Access your app at: `https://huggingface.co/spaces/YOUR_USERNAME/recipe-recommender`

### Troubleshooting Hugging Face Spaces

| Issue | Solution |
|-------|----------|
| Space stuck on "Building" | Check GitHub Actions logs for errors |
| Module import errors | Pin package versions in requirements.txt |
| Slow startup (>30s) | HF Spaces may put you on slow tier; upgrade to GPU (paid) |
| File paths broken | Use relative paths: `./Data/indian_food.csv` not absolute Windows paths |

---

## Post-Deployment Checklist

After deployment, verify:

- [ ] App loads without errors (check browser console)
- [ ] Recipe Finder tab works (search returns results)
- [ ] Pantry Checker tab works (ingredient selection filters correctly)
- [ ] Images load successfully
- [ ] All filters (Region, Course, Diet, Flavor) update results
- [ ] Response time is <1 second per query

### Test Query Examples

Try these to verify functionality:

| Query | Expected Result |
|-------|-----------------|
| Region: North, Query: "sweet cardamom" | Should return desserts with milk/cardamom |
| Pantry: [milk, sugar, rice, cardamom] | Should find recipes with high coverage % |
| Region: South, Course: Main Course | Should return South Indian curries |

---

## Sharing Your Submission Link

Once deployed, share this link with instructors:

**Format:**
```
Streamlit Cloud:
https://your-app-name.streamlit.app

OR

Hugging Face Spaces:
https://huggingface.co/spaces/YOUR_USERNAME/recipe-recommender
```

**Also include in your submission:**
- Link to GitHub repository
- Link to Technical Report (PDF or GitHub link)
- Link to Training Notebook (GitHub or `.ipynb` file)

---

## Advanced: Custom Domain (Optional)

### Streamlit Cloud
1. Go to your app settings in Streamlit Cloud
2. Under "Custom domain", add your domain (requires DNS configuration)
3. Cost: Free

### Hugging Face Spaces
1. Spaces are served under `huggingface.co/spaces/...`
2. Custom domains not available on free tier

---

## Monitoring & Logs

### Streamlit Cloud
- **View logs**: Click "Manage app" → "View logs"
- **Restarts**: App automatically restarts if code changes

### Hugging Face Spaces
- **View logs**: Click "View logs" button in Space
- **Auto-updates**: Changes to GitHub branch auto-deploy

---

## Environment Variables (If Needed)

If you later add API keys or secrets:

### Streamlit Cloud
```
In app settings:
1. Go to "Secrets" section
2. Add as TOML:
   [passwords]
   api_key = "your_key_here"
```

### Hugging Face Spaces
```
In Space settings:
1. Go to "Repository secrets"
2. Add each secret individually
```

Access in code:
```python
import streamlit as st
api_key = st.secrets.get("api_key", "default")
```

---

## Estimated Costs

| Platform | Free Tier | Cost |
|----------|-----------|------|
| **Streamlit Cloud** | ✓ Yes (2GB RAM, 1GB disk) | Free |
| **Hugging Face Spaces** | ✓ Yes (CPU, 16GB disk) | Free (upgrade to GPU for $7.50/month) |
| **AWS/GCP** | ✗ No | $20-50/month |

---

## Deployment Summary

| Aspect | Streamlit Cloud | HF Spaces |
|--------|-----------------|-----------|
| Setup time | 5 minutes | 10 minutes |
| Deployment time | 2-5 minutes | 5-15 minutes |
| Free tier | ✓ Yes (sufficient) | ✓ Yes (sufficient) |
| Custom domain | ✓ Available | ✗ Not on free |
| GitHub sync | ✓ Automatic | ✓ Automatic |
| Easiest for beginners | ✓✓✓ | ✓✓ |

**Recommendation**: Use **Streamlit Cloud** for fastest, easiest deployment.

---

## Support & Troubleshooting

If deployment fails:

1. **Check requirements.txt**: All packages installed locally?
   ```bash
   pip install -r requirements.txt
   ```

2. **Test locally**:
   ```bash
   streamlit run app.py
   ```

3. **View deployment logs**: Check Streamlit Cloud / HF Spaces logs for error messages

4. **Common errors**:
   - `ModuleNotFoundError`: Add missing package to requirements.txt
   - `FileNotFoundError`: Use relative paths (`./ Data/indian_food.csv`)
   - `Port already in use`: Normal on cloud (ignore, cloud will assign port)

5. **Performance tips**:
   - Cache model loading: `@st.cache_data`
   - Reduce dataset size if startup >10s
   - Use lazy loading for images

---

## Next Steps

After successful deployment:

1. ✅ Test all features thoroughly
2. ✅ Collect user feedback (optional survey)
3. ✅ Fix any bugs found during testing
4. ✅ Submit link to instructors
5. 📈 Monitor app usage in cloud dashboard

---

**For questions or issues, refer to:**
- Streamlit Docs: https://docs.streamlit.io
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- GitHub Pages: https://pages.github.com

---

Last updated: May 6, 2026
