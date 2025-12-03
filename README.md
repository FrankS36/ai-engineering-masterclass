<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1MY640nQ7Js-iUJ5i6FnNSmxJFbwAJu_2

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Deploy to Vercel

The app is configured for deployment on Vercel.

1. **Push to GitHub**: Your code is already connected to GitHub
2. **Import to Vercel**: 
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Import your GitHub repository
   - Vercel will auto-detect the Vite framework
3. **Set Environment Variables**:
   - In Vercel project settings, add `GEMINI_API_KEY` as an environment variable
   - Set it to your Gemini API key value
   - Redeploy after adding the variable
4. **Deploy**: Vercel will automatically deploy on every push to the `main` branch

**Live URL**: https://aiengineeringmasterclass.vercel.app
