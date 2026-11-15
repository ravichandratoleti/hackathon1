# CODEX PROJECT SPECIFICATION
# VERITAS AI — AI-Powered Fact Verification & Deepfake Audit Engine

## 1. PROJECT OVERVIEW

Build a modern, full-stack AI-powered web application called **VERITAS AI**.

**Tagline:** Don't Just Believe It. Verify It.

VERITAS AI helps users identify potentially manipulated digital content and verify suspicious information before trusting or sharing it.

The platform should analyse:
- Images for possible AI generation or manipulation.
- Videos for suspicious frames, inconsistencies and possible manipulation.
- Audio recordings for possible voice cloning or synthetic speech.
- Text and news claims by checking them against credible online sources.

The application must not simply display a fake/real percentage. It should provide understandable explanations, supporting evidence, source links and an honest confidence assessment.

**Target users:** Students, journalists, researchers, citizens and general internet users.

**Primary objective:** Create a functional hackathon-ready prototype with a polished user experience and working analysis pipelines.

---

## 2. TECHNOLOGY STACK

### Frontend
- Next.js with React
- TypeScript
- Tailwind CSS
- Framer Motion for animations
- Lucide React for icons
- Recharts for visualisations

### Backend
- Python
- FastAPI
- Pydantic
- REST API architecture

### AI and Media Processing
- PyTorch
- Hugging Face Transformers
- OpenCV
- Pillow
- Librosa
- FFmpeg

### Database
- SQLite for the prototype
- SQLAlchemy ORM

### Other
- Axios or Fetch API
- PDF report generation
- Environment variables using `.env`
- Git-friendly project structure

Use stable, compatible package versions. Avoid unnecessary dependencies.

---

## 3. APPLICATION DESIGN AND UI

Create a premium cybersecurity and digital-forensics-inspired interface.

### Visual style
- Dark navy/black background.
- Electric blue and violet accents.
- Subtle gradients and glassmorphism.
- Clean typography.
- Smooth transitions and micro-interactions.
- Professional dashboard layout.
- Responsive design for desktop, tablet and mobile.

Avoid excessive animations, clutter and generic template styling.

### Colour palette
- Background: #080B14
- Secondary background: #101827
- Primary accent: #6366F1
- Secondary accent: #06B6D4
- Success: #22C55E
- Warning: #F59E0B
- Danger: #EF4444
- Main text: #F8FAFC
- Muted text: #94A3B8

### Branding
Name: VERITAS AI

Create a simple logo using a shield/checkmark or verification symbol.

The interface should feel like a professional digital investigation platform.

---

## 4. MAIN APPLICATION PAGES

### A. Landing Page

Create an impressive landing page containing:

1. Navigation bar:
   - VERITAS AI logo.
   - Home.
   - Verify Content.
   - How It Works.
   - About.
   - Get Started button.

2. Hero section:
   - Heading: "In a World of Fake Content, Find the Truth."
   - Supporting text explaining the platform.
   - Primary CTA: Start Verification.
   - Secondary CTA: Explore How It Works.
   - Animated digital verification visual.

3. Statistics section:
   - Number of files analysed.
   - Claims verified.
   - Reports generated.

   IMPORTANT: Only show real application statistics. Do not invent impressive-looking numbers.

4. Feature cards:
   - Image Verification.
   - Video Forensics.
   - Audio Deepfake Detection.
   - AI Fact Checking.

5. How It Works:
   - Upload or enter content.
   - AI analyses the content.
   - Evidence is collected.
   - Results and explanations are displayed.

6. Footer:
   - About.
   - Privacy.
   - Contact.
   - Disclaimer.

### B. Main Verification Dashboard

This is the most important page.

Create a dashboard with four verification modes:

1. Image Analysis.
2. Video Analysis.
3. Audio Analysis.
4. Text / News Fact Checking.

Each mode should have a dedicated interface.

Include:
- Upload area with drag-and-drop support.
- File format and size validation.
- Preview of uploaded content.
- Analysis button.
- Processing indicator.
- Analysis history.
- Recent verification reports.

The user must be able to switch between the four modes easily.

### C. Analysis Results Page

After analysis, show a detailed report.

Display:

**Overall verdict**
- Likely Authentic.
- Potentially Manipulated.
- Inconclusive.

**Important metrics**
- Manipulation likelihood.
- Model confidence.
- Evidence coverage.
- Analysis duration.

Do not treat model confidence as proof that a file is fake or genuine.

Use clear visual indicators:
- Green: lower detected manipulation risk.
- Orange: uncertain or requires review.
- Red: suspicious indicators detected.

Include:
- Summary of findings.
- Suspicious indicators.
- Evidence and explanations.
- Model limitations.
- Recommended next steps.

For fact-checking, also display:
- Claim being checked.
- Verdict: Supported, Contradicted, Mixed, or Insufficient Evidence.
- Sources used.
- Source publication dates where available.
- Direct source links.
- Short explanation of how each source relates to the claim.

Never fabricate evidence, citations, sources or analysis results.

### D. History Page

Display previous analyses.

Each record should contain:
- File or claim name.
- Analysis type.
- Date and time.
- Result.
- Confidence.
- View report option.
- Delete option.

Store analysis metadata in SQLite.

### E. About and How It Works

Explain:
- Why deepfake detection matters.
- Why misinformation is dangerous.
- How the verification process works.
- The limitations of AI detection.

Make it clear that AI results are indicators, not absolute proof.

---

## 5. CORE FUNCTIONAL MODULES

### MODULE 1: IMAGE VERIFICATION

Allow users to upload:
- JPG
- JPEG
- PNG
- WEBP

Processing pipeline:

1. Validate uploaded file.
2. Read image metadata.
3. Generate a preview.
4. Analyse the image using an available pretrained AI image detection model.
5. Examine relevant image inconsistencies and manipulation indicators.
6. Generate a structured result.
7. Save the analysis metadata.

Where possible, include:
- AI-generated image likelihood.
- Image manipulation indicators.
- Metadata observations.
- Model confidence.
- Explanation of detected indicators.

Do not claim that metadata alone proves an image is fake.

### MODULE 2: VIDEO FORENSICS

Allow users to upload:
- MP4
- MOV
- AVI
- WEBM

Processing pipeline:

1. Validate the video.
2. Extract video metadata.
3. Extract representative frames using FFmpeg/OpenCV.
4. Analyse selected frames using an available image manipulation or deepfake detection model.
5. Identify frames with suspicious indicators.
6. Aggregate frame-level results carefully.
7. Generate a video analysis report.

Display:
- Video preview.
- Duration.
- Number of frames analysed.
- Suspicious timestamps.
- Frame-level findings.
- Overall assessment.

Do not claim that analysing a few frames proves the entire video is authentic or manipulated.

### MODULE 3: AUDIO DEEPFAKE DETECTION

Allow users to upload:
- WAV
- MP3
- M4A
- FLAC

Processing pipeline:

1. Validate audio.
2. Convert to a compatible format using FFmpeg if needed.
3. Extract audio features using Librosa.
4. Run an available pretrained synthetic speech or audio deepfake detection model.
5. Generate findings.
6. Save the analysis metadata.

Display:
- Audio duration.
- Waveform visualisation.
- Suspicious audio indicators.
- Model confidence.
- Overall assessment.

Do not claim that ordinary audio features alone can conclusively detect cloned voices.

### MODULE 4: AI FACT CHECKER

This module is essential because the hackathon challenge includes fact verification as well as deepfake detection.

Allow users to:
- Paste a news headline.
- Enter a statement.
- Paste a paragraph.
- Submit a claim they want to verify.

Processing pipeline:

1. Accept the user's claim.
2. Extract the main factual assertion.
3. Search credible online sources using an appropriate search API.
4. Retrieve relevant source information.
5. Compare the claim against available evidence.
6. Generate an evidence-based assessment.
7. Display citations and source links.

Possible verdicts:
- SUPPORTED
- CONTRADICTED
- MIXED
- INSUFFICIENT EVIDENCE

For every result, display:
- Original claim.
- Extracted factual assertion.
- Verdict.
- Explanation.
- Supporting evidence.
- Contradicting evidence.
- Source title.
- Publisher/domain.
- Publication date, if available.
- Direct source URL.

Use a configurable search provider.

Read API credentials from environment variables.

Do not hardcode API keys.

Do not generate fake citations or pretend that a search occurred when no search provider is configured.

If external search is unavailable, clearly display that live source verification is unavailable and return an insufficient-evidence result rather than inventing information.

---

## 6. AI ANALYSIS AND RESULT INTEGRITY

This section is extremely important.

Do not build a fake AI interface that generates random scores or random verdicts.

Use actual available pretrained models or clearly identified fallback analysis methods.

If a model is unavailable:
- Show that the feature is not configured.
- Explain which component is missing.
- Provide setup instructions.
- Do not pretend that the analysis succeeded.

For every analysis, distinguish:

1. Manipulation likelihood: the model's estimated likelihood of manipulation.
2. Model confidence: how confident the model is in its prediction.
3. Evidence coverage: how much of the content was actually analysed.

These are different measurements and must not be presented as interchangeable.

Use an internal result structure similar to:

{
  "analysis_id": "unique-id",
  "media_type": "image",
  "verdict": "inconclusive",
  "manipulation_likelihood": null,
  "model_confidence": null,
  "evidence_coverage": "not_available",
  "findings": [],
  "limitations": [],
  "created_at": "timestamp"
}

Use null when a measurement is unavailable.

Never manufacture numerical confidence values.

All results should include a disclaimer:

"AI-generated verification results are probabilistic and may contain errors. This assessment is not definitive proof of authenticity or manipulation. Verify important findings using independent sources."

---

## 7. BACKEND API STRUCTURE

Create these endpoints:

### Health
GET /api/health

Returns backend status and model availability.

### Image
POST /api/analyze/image

Accepts image upload and returns analysis results.

### Video
POST /api/analyze/video

Accepts video upload and returns analysis results.

### Audio
POST /api/analyze/audio

Accepts audio upload and returns analysis results.

### Fact Checking
POST /api/fact-check

Accepts a text claim and returns:
- Claim.
- Verdict.
- Explanation.
- Supporting sources.
- Contradicting sources.
- Citations.

### Analysis History
GET /api/history

Returns previous analysis records.

GET /api/history/{analysis_id}

Returns a specific analysis.

DELETE /api/history/{analysis_id}

Deletes an analysis record and associated temporary files where applicable.

### PDF Reports
GET /api/report/{analysis_id}

Generates and downloads a PDF verification report.

Use appropriate HTTP status codes and consistent JSON response structures.

---

## 8. DATABASE DESIGN

Use SQLite and SQLAlchemy.

Create an analyses table with fields:

- id
- analysis_type
- original_filename
- verdict
- manipulation_likelihood
- model_confidence
- evidence_coverage
- summary
- findings_json
- sources_json
- limitations_json
- created_at

Store only necessary metadata.

Avoid storing uploaded media permanently by default.

Provide an option to delete analysis records.

---

## 9. SECURITY AND PRIVACY

Implement:

- File type validation.
- File size limits.
- Secure temporary file storage.
- Unique filenames.
- File cleanup after processing.
- Input validation.
- Safe error handling.
- Environment-based API credentials.
- No hardcoded secrets.
- No execution of uploaded files.
- Protection against path traversal.
- Restricted CORS configuration.

Set reasonable upload limits and communicate them in the interface.

The application should explain how uploaded files are handled.

Do not claim that files are permanently deleted unless deletion is actually implemented and verified.

---

## 10. PROJECT STRUCTURE

Use a clean structure similar to:

veritas-ai/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── services/
│   ├── types/
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── image_analysis.py
│   │   │   ├── video_analysis.py
│   │   │   ├── audio_analysis.py
│   │   │   └── fact_check.py
│   │   ├── database/
│   │   └── utils/
│   ├── requirements.txt
│   └── .env.example
│
├── README.md
└── .gitignore

You may improve this structure if necessary, but keep frontend and backend clearly separated.

---

## 11. USER EXPERIENCE REQUIREMENTS

The application must be genuinely interactive.

Implement:
- Working navigation.
- Drag-and-drop uploads.
- File previews.
- Loading animations.
- Analysis progress states.
- Error and success notifications.
- Results visualisations.
- Expandable evidence sections.
- Clickable source links.
- Downloadable PDF reports.
- Searchable analysis history.
- Responsive layouts.

Do not leave buttons that look functional but do nothing.

Do not use placeholder data as if it were real analysis.

If demo data is used, clearly label it as sample data.

---

## 12. DEVELOPMENT PRIORITIES

Build the project in this order:

### Phase 1 — Foundation
- Create frontend and backend.
- Configure communication between them.
- Build the landing page.
- Build the dashboard.
- Create navigation and responsive design.

### Phase 2 — Working Fact Checker
- Implement claim submission.
- Integrate a real search provider.
- Retrieve sources.
- Generate evidence-based verdicts.
- Display source citations.

### Phase 3 — Image Analysis
- Integrate a real pretrained model.
- Implement image upload and processing.
- Display explainable results.

### Phase 4 — Video Analysis
- Extract representative frames.
- Analyse frames.
- Display timestamps and findings.

### Phase 5 — Audio Analysis
- Process audio.
- Integrate a compatible pretrained model.
- Display results.

### Phase 6 — Reports and History
- SQLite integration.
- Analysis history.
- PDF report generation.
- Delete functionality.

### Phase 7 — Testing and Polish
- Test all endpoints.
- Test invalid files.
- Test large files.
- Test API failures.
- Test missing models and credentials.
- Fix frontend/backend integration issues.
- Improve responsiveness.

Prioritise a working, honest fact checker and at least one functional media-analysis pipeline over several incomplete AI features.

---

## 13. CODEX EXECUTION INSTRUCTIONS

Follow these instructions carefully:

1. First inspect the existing project directory before creating files.
2. If the project is empty, initialise the complete project structure.
3. If code already exists, preserve useful existing work.
4. Build the application in logical phases.
5. Do not generate the entire project as disconnected placeholder components.
6. Connect every frontend feature to its actual backend endpoint.
7. Use real model integrations where possible.
8. If model weights or external API credentials are unavailable, provide clear setup instructions and graceful unavailable states.
9. Do not use random AI scores, fabricated reports, fake sources or fake citations.
10. Add comments for important processing logic.
11. Handle errors properly.
12. Create README.md with installation, configuration and execution instructions.
13. Create `.env.example` listing all required environment variables.
14. Ensure the project can run locally on Windows.
15. Test the application and fix errors before declaring the project complete.

### Required final deliverables

- Complete frontend.
- Complete backend.
- Functional image analysis pipeline.
- Functional fact-checking pipeline.
- Video and audio pipeline integrations or clearly documented unavailable states.
- Database and history.
- PDF report generation.
- README with exact setup commands.
- `.env.example`.
- Working local application.

### FINAL INSTRUCTION

Build VERITAS AI as a polished, functional hackathon project, not just a static UI demonstration.

The most important priorities are:
1. Accuracy and honesty of AI results.
2. Evidence-backed fact verification.
3. Explainable deepfake analysis.
4. Excellent UI/UX.
5. Reliable end-to-end functionality.

Start by inspecting the project, then implement the application step by step. Do not stop after generating the frontend.