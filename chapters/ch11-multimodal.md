# Chapter 11: Multimodal AI

Language models that only process text are increasingly the exception. The models shipping today -- GPT-4o, Claude, Gemini -- accept images, audio, and video alongside text, and the engineering patterns for working with these inputs are maturing fast. This chapter covers the practical side: how to send images and audio to APIs, how to build pipelines that process complex documents, and where these capabilities genuinely work versus where they will quietly fail on you.

---

## Vision Models

### What They Can Do

Modern vision-language models handle a broad range of image understanding tasks without any special training:

- **Image description and analysis**: Describe what is in a photo, identify objects, read scenes.
- **OCR and text extraction**: Read text from screenshots, signs, handwritten notes, documents. Accuracy varies with image quality, but is strong on clean printed text.
- **Visual question answering**: Answer specific questions about an image ("What brand is the laptop in this photo?", "How many people are in the room?").
- **Chart and diagram interpretation**: Extract data from bar charts, read flowcharts, interpret architectural diagrams.
- **Code from screenshots**: Convert UI mockups or whiteboard sketches into working code.
- **Comparison**: Spot differences between two images or compare a design mockup to a screenshot.

### Provider Capabilities

| Capability | GPT-4o | Claude | Gemini |
|---|---|---|---|
| Max images per request | 20+ | 20 | 16 (native), 3600 frames (video) |
| Image input formats | PNG, JPEG, GIF, WebP | PNG, JPEG, GIF, WebP | PNG, JPEG, GIF, WebP, plus native video |
| Max image size | 20MB | 5MB per image | 20MB |
| Resolution handling | Auto, low, high modes | Auto-scales, max 1568px on long side | Auto-scales |
| OCR quality | Strong | Strong | Strong |
| Spatial reasoning | Moderate | Moderate | Moderate |
| Counting accuracy | Unreliable above ~10 | Unreliable above ~10 | Unreliable above ~10 |

### Where They Fail

Vision models are confidently wrong more often than text models. Key weaknesses:

- **Counting**: Ask "how many windows are on this building" and you will get inconsistent answers. Anything above roughly 10 items becomes unreliable.
- **Spatial reasoning**: "Is the red car to the left or right of the blue car?" produces errors at a surprisingly high rate, especially in cluttered scenes.
- **Fine-grained text**: Small text, low contrast text, text at angles, and handwriting all degrade OCR accuracy.
- **Hallucinated details**: The model may confidently describe text on a sign that does not exist, or misread numbers. Always validate extracted data against ground truth when accuracy matters.
- **Coordinates and measurements**: Models cannot reliably give pixel coordinates, measure distances, or determine exact sizes.

---

## Working with Images

### Encoding and Sending

Images go to APIs either as base64-encoded strings or as URLs. Base64 is more reliable (no dependency on URL accessibility) and is the standard for production systems.

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

def analyze_image(image_path: str, question: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    base64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Determine media type from extension
    suffix = Path(image_path).suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "image/png")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_image}},
                {"type": "text", "text": question},
            ],
        }],
    )
    return response.content[0].text

result = analyze_image("invoice.png", "Extract the invoice number, date, total amount, and line items as JSON.")
```

The OpenAI equivalent uses a `image_url` content block with either a URL or a data URI:

```python
from openai import OpenAI
import base64

client = OpenAI()

with open("invoice.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            {"type": "text", "text": "Extract the invoice number, date, total amount, and line items as JSON."},
        ],
    }],
)
```

### Resolution and Token Cost

Image tokens are expensive. A high-resolution image in GPT-4o can consume 1,000+ tokens. The cost scales with resolution:

| Resolution | Approximate tokens (GPT-4o) | Use when |
|---|---|---|
| Low (512x512) | ~85 tokens | General understanding, no fine detail needed |
| High (up to 2048x2048) | 300-1,600 tokens | OCR, reading small text, detailed analysis |
| Auto | Varies | Let the API decide based on content |

Optimization strategies that matter in production:

- **Resize before sending**: If you only need to read a header, crop to the relevant region. Do not send a 4K image to read a 200x50 pixel text box.
- **Use low detail when possible**: For classification tasks ("is this a cat or a dog?"), low resolution is sufficient and 10-20x cheaper.
- **Batch regions of interest**: If you need to read 5 sections of a large document, crop each section and send as separate images rather than sending the full page 5 times with different prompts.

---

## Audio

### Speech-to-Text with Whisper

OpenAI's Whisper is the de facto standard for speech-to-text. Available as both a cloud API and a local model.

```python
from openai import OpenAI

client = OpenAI()

# Transcribe audio file
with open("meeting.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",  # includes timestamps
        timestamp_granularities=["segment"],
    )

for segment in transcript.segments:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
```

For local deployment, `faster-whisper` provides the same accuracy with 4x better speed through CTranslate2 optimization:

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("meeting.mp3", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

### Real-Time Transcription

For live audio (call centers, live captioning), you need streaming transcription. The pattern: capture audio in chunks, send each chunk for transcription, and stitch results together.

Deepgram and AssemblyAI offer WebSocket-based streaming APIs purpose-built for this. OpenAI's Realtime API supports bidirectional audio streaming for conversational use cases.

### Text-to-Speech

TTS has gotten good enough for production use. OpenAI's TTS API generates natural-sounding speech:

```python
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",
    input="Your order has been confirmed. Expected delivery is Thursday between 2 and 5 PM.",
)

response.stream_to_file("confirmation.mp3")
```

For local TTS, Coqui TTS and Piper offer open-source alternatives with reasonable quality. ElevenLabs provides the highest-quality voice cloning if that is your use case.

---

## Video

### The Practical Approach: Frame Sampling

No widely available API processes video natively at a reasonable cost (Gemini is the exception with native video input). The standard approach is frame sampling: extract key frames and send them as images.

```python
import cv2
import base64

def extract_frames(video_path: str, interval_seconds: float = 2.0) -> list[dict]:
    """Extract frames from video at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode("utf-8")
            timestamp = frame_count / fps
            frames.append({"base64": b64, "timestamp": timestamp})
        frame_count += 1

    cap.release()
    return frames

# Extract one frame every 2 seconds from a video
frames = extract_frames("product_demo.mp4", interval_seconds=2.0)

# Send to vision model
content = []
for frame in frames[:20]:  # limit to 20 frames for cost control
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame['base64']}", "detail": "low"}})
content.append({"type": "text", "text": "Describe what happens in this product demo video, step by step."})
```

### Frame Sampling Strategies

- **Fixed interval**: One frame every N seconds. Simple and predictable. Good for surveillance, lectures.
- **Scene change detection**: Use OpenCV to detect significant visual changes and only capture those frames. Efficient for edited content.
- **Shot boundary detection**: Identify cuts in edited video and sample from each shot. Best for analyzing produced video content.

For Gemini, you can upload video directly via the File API:

```python
import google.genai as genai

client = genai.Client()

# Upload video file
video_file = client.files.upload(file="product_demo.mp4")

# Wait for processing
while video_file.state.name == "PROCESSING":
    video_file = client.files.get(name=video_file.name)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[video_file, "Describe what happens in this product demo, step by step."],
)
```

---

## Multimodal RAG

Standard RAG pipelines break when documents contain tables, charts, diagrams, and images that carry meaning not captured in extracted text.

### Approach 1: Rich Text Extraction

Convert everything to text, preserving structure. Tables become markdown tables. Chart descriptions are generated by vision models. This keeps your embedding and retrieval pipeline text-based.

```python
import anthropic
import base64

client = anthropic.Anthropic()

def describe_page_image(page_image_b64: str) -> str:
    """Use a vision model to extract all content from a document page image."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": page_image_b64}},
                {"type": "text", "text": "Extract ALL content from this document page. Preserve table structure as markdown tables. Describe any charts or diagrams in detail. Include all visible text."},
            ],
        }],
    )
    return response.content[0].text
```

### Approach 2: Image Embeddings with CLIP

Use CLIP (or similar models) to embed images directly into the same vector space as text. At retrieval time, a text query can match both text chunks and relevant images.

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load a multimodal embedding model
model = SentenceTransformer("clip-ViT-L-14")

# Embed text and images into the same vector space
text_embedding = model.encode("quarterly revenue growth chart")
image_embedding = model.encode(Image.open("revenue_chart.png"))

# These embeddings are directly comparable via cosine similarity
```

### Approach 3: Hybrid Pipeline

The most robust approach for complex documents: extract text normally, render pages as images, embed both text chunks and page images, and retrieve both modalities. At generation time, pass the retrieved text and images together to a vision-language model.

---

## Document Processing

PDFs, invoices, forms, and contracts are the most common multimodal workload in enterprise AI. Here is a practical extraction pipeline:

### Step 1: Render to Images

Convert each page to an image. This sidesteps all the problems with PDF text extraction (broken encoding, layout issues, scanned documents).

```python
import fitz  # PyMuPDF

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[bytes]:
    """Convert each PDF page to a PNG image."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images
```

### Step 2: Extract with Vision Model

Send each page image to a vision model with a schema-specific prompt:

```python
import json

def extract_invoice_data(page_images: list[bytes]) -> dict:
    """Extract structured data from invoice page images."""
    all_content = []
    for img_bytes in page_images:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        all_content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})

    all_content.append({"type": "text", "text": """Extract the following from this invoice:
- vendor_name: string
- invoice_number: string
- invoice_date: string (YYYY-MM-DD)
- due_date: string (YYYY-MM-DD)
- line_items: array of {description, quantity, unit_price, total}
- subtotal: number
- tax: number
- total: number

Return valid JSON only."""})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": all_content}],
    )
    return json.loads(response.content[0].text)
```

### Step 3: Validate and Post-Process

Never trust extraction output blindly. Validate totals (do line items sum correctly?), check date formats, and flag confidence-critical fields for human review.

---

## Cost and Latency

Multimodal requests are significantly more expensive and slower than text-only requests. Budgeting matters.

| Input type | Approximate cost per unit (GPT-4o) | Latency impact |
|---|---|---|
| Text (1K tokens) | $0.0025 input | Baseline |
| Image (low detail) | ~$0.0007 (85 tokens) | +200-500ms |
| Image (high detail) | $0.003-$0.01 (300-1,600 tokens) | +500-2,000ms |
| Audio (1 minute, Whisper) | $0.006 | 5-15 seconds |
| Video (1 min, 30 frames low-res) | ~$0.02 | +5-15 seconds |

### Optimization Strategies

1. **Resize aggressively**: Most document processing works fine at 150-200 DPI. Sending 300 DPI images doubles cost for marginal quality gain.
2. **Crop regions of interest**: If you only need the header of an invoice, do not send the entire page.
3. **Cache extracted content**: Once you have extracted structured data from an image, cache it. Re-extraction is wasteful.
4. **Use cheaper models for triage**: Route images through a fast, cheap model first (GPT-4o-mini, Gemini Flash) to classify or check if detailed extraction is needed, then use a more expensive model only when necessary.
5. **Batch processing**: When processing hundreds of documents, use async requests to maximize throughput without hitting per-request latency.

---

## Known Limitations

These are not edge cases. They are fundamental limitations of current vision-language models that you will encounter in production:

**Hallucinated text**: Models may "read" text that is not in the image, especially when the image is low quality or the text is partially obscured. Always cross-reference critical extractions.

**Spatial reasoning failures**: Asking about relative positions of objects, directions, or layouts produces unreliable results. "Which column is the total in?" works better than "What is to the right of the date field?"

**Counting problems**: Any task that requires counting objects above approximately 7-10 items will produce errors. If you need accurate counts, use traditional computer vision or ask the model to list items individually and count programmatically.

**Text in images**: While OCR capability has improved dramatically, the models still struggle with handwriting, stylized fonts, text at unusual angles, and low-contrast text. For high-accuracy OCR on clean documents, dedicated OCR services (Google Document AI, AWS Textract) may still outperform general vision models.

**Inconsistency across runs**: The same image with the same prompt can produce different extractions on different runs. For critical data, run extraction multiple times and take the consensus, or use temperature=0 (though even this does not guarantee determinism).

**No pixel-level precision**: Models cannot reliably identify exact bounding boxes, pixel coordinates, or precise measurements within images. If you need localization, use dedicated object detection models.

These limitations are real but manageable. The key is designing systems that account for them -- validation layers, human review for high-stakes decisions, and fallback to specialized tools when general-purpose vision models are insufficient.