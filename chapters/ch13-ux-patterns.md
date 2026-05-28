# Chapter 13: AI UX Patterns

Traditional software UX assumes determinism: the same input produces the same output, every time. AI breaks that contract. A language model might generate a brilliant summary on one request and a mediocre one on the next. A classification model might be 97% accurate overall but fail catastrophically on the exact input your most important customer just submitted. Every AI-powered interface must be designed around this fundamental reality: your system will sometimes be wrong, and the user needs to know what to do when it is.

This chapter covers the UX patterns that make AI products usable, trustworthy, and accessible. These are not theoretical frameworks. They are the specific design decisions that separate AI products people actually use from ones they abandon after a week.

> AI PM is not software PM. In software, you're eliminating uncertainty. In AI, you're managing it. The "what" and the "how" are entangled — the data shapes what's possible. The PM who doesn't engage with those realities will build a roadmap the system can't deliver.

## The Core Challenge: Designing for Probabilistic Systems

In traditional software, a bug is a deviation from specified behavior. In AI, "deviation from expected behavior" is the normal operating mode. The model's output falls on a distribution. Sometimes it lands near the center — exactly what the user wanted. Sometimes it lands at the tail — confusing, wrong, or offensive.

This means every AI interface must answer three questions that traditional interfaces never have to:

1. **How do we communicate uncertainty?** The user needs to understand that what they are seeing is a suggestion, not a fact.
2. **How do we recover from errors?** The system will be wrong. The user needs a fast path back to productivity.
3. **How do we learn from users?** Every correction is training signal. The interface needs to capture it.

The patterns below are your toolkit for answering these questions.

## Pattern 1: Suggestion, Not Automation

The most common mistake in AI product design is presenting AI output as a decision rather than a recommendation. When an AI auto-fills a field and the user has to notice it was wrong, you have created a system that is worse than no AI at all — because the user now has to verify everything the AI did, which is harder than doing it themselves.

The fix is simple: present AI outputs as suggestions that require explicit user action.

```tsx
function EmailDraftSuggestion({ draft, onAccept, onEdit, onReject }) {
  return (
    <div
      role="region"
      aria-label="AI-suggested email draft"
      className="ai-suggestion"
    >
      <div className="suggestion-header">
        <span className="suggestion-badge">Suggested Draft</span>
        <span className="suggestion-hint">Review before sending</span>
      </div>
      <div className="suggestion-body">
        <p className="draft-text">{draft.text}</p>
      </div>
      <div className="suggestion-actions">
        <button onClick={onAccept} className="btn-accept">
          Use This Draft
        </button>
        <button onClick={onEdit} className="btn-edit">
          Edit
        </button>
        <button onClick={onReject} className="btn-reject">
          Discard
        </button>
      </div>
    </div>
  );
}
```

Key design decisions here: the label says "Suggested Draft," not "Your Draft." There is no auto-send. The user must take an explicit action. The "Edit" option is given equal visual weight to "Accept" — you want editing to feel like a normal workflow, not a correction.

Code completion interfaces like those in IDEs follow the same principle. The suggestion appears as ghost text, dimmed. The user presses Tab to accept, keeps typing to ignore. The AI never inserts code without the user's explicit gesture.

## Pattern 2: Progressive Disclosure

AI systems often have rich reasoning behind their outputs — retrieved documents, confidence breakdowns, intermediate steps. Dumping all of this on the user by default creates cognitive overload. Hiding all of it creates a black box. Progressive disclosure gives you both: a clean default experience with depth available on demand.

```tsx
function AnalysisResult({ summary, details, sources }) {
  const [depth, setDepth] = useState("summary");

  return (
    <div className="analysis-result">
      <div className="result-summary">
        <p>{summary}</p>
      </div>

      {depth === "summary" && (
        <button onClick={() => setDepth("details")}>
          Show reasoning
        </button>
      )}

      {depth === "details" && (
        <div className="result-details">
          <h4>Analysis Details</h4>
          {details.map((step, i) => (
            <div key={i} className="reasoning-step">
              <span className="step-number">{i + 1}</span>
              <p>{step.explanation}</p>
            </div>
          ))}
          <button onClick={() => setDepth("sources")}>
            View sources
          </button>
        </div>
      )}

      {depth === "sources" && (
        <div className="result-sources">
          <h4>Source Documents</h4>
          {sources.map((source, i) => (
            <div key={i} className="source-item">
              <a href={source.url}>{source.title}</a>
              <p className="source-excerpt">{source.excerpt}</p>
              <span className="source-score">
                Relevance: {(source.score * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

Three levels work well for most AI products: **summary** (the answer), **details** (how the system got there), and **raw data** (the sources or evidence). Power users will drill down. Most users will stay at the summary level — and that is fine, as long as the option exists.

## Pattern 3: Confidence Indicators

Showing confidence scores seems like an obvious transparency win. In practice, it is a minefield.

**When confidence indicators help:**
- Expert users making high-stakes decisions (radiologists reviewing AI-flagged scans)
- Triage workflows where confidence determines routing (auto-approve above 95%, human review below)
- Developer-facing tools where the user understands probability

**When confidence indicators hurt:**
- Consumer products where users interpret 80% confidence as "probably wrong" (it is not — 80% is strong for many tasks)
- Interfaces where the number creates false precision ("92.3% confident" implies a level of calibration the model probably does not have)
- Situations where showing low confidence on a correct answer erodes trust more than showing no confidence at all

If you do show confidence, use qualitative buckets rather than raw numbers for non-expert audiences:

```tsx
function ConfidenceIndicator({ score }) {
  const level =
    score >= 0.9 ? "high" : score >= 0.7 ? "medium" : "low";

  const labels = {
    high: "High confidence",
    medium: "Moderate confidence — review recommended",
    low: "Low confidence — manual verification needed",
  };

  return (
    <div
      className={`confidence confidence-${level}`}
      role="status"
      aria-label={labels[level]}
    >
      <div
        className="confidence-bar"
        style={{ width: `${score * 100}%` }}
      />
      <span className="confidence-label">{labels[level]}</span>
    </div>
  );
}
```

Never show confidence without also providing a clear action tied to it. "Low confidence" with no guidance on what to do next just makes the user anxious.

## Pattern 4: Graceful Degradation

AI failures come in different shapes, and each shape needs a different recovery path. A timeout is different from a nonsensical output, which is different from a content policy violation. Treating them all the same — "Something went wrong, please try again" — wastes the user's time and erodes trust.

| Error Type | User Message | Recovery Action |
|---|---|---|
| Timeout / rate limit | "This is taking longer than expected. We're still working on it." | Auto-retry with backoff; show progress indicator |
| Low confidence result | "We found a result but aren't fully confident. Please review carefully." | Show result with prominent edit/override controls |
| No result found | "We couldn't find a good answer for this. Here's what we tried." | Show partial results; suggest query reformulation |
| Content policy violation | "This request falls outside what we can help with." | Suggest alternative phrasing; link to usage guidelines |
| Model unavailable | "Our AI service is temporarily unavailable. You can still [do X manually]." | Provide non-AI fallback; queue request for later |
| Malformed input | "We need a bit more to work with. Try adding [specific detail]." | Highlight the issue; provide input examples |

The critical principle: always give the user something to do next. A dead end is the worst possible AI UX.

```tsx
function AIFallback({ error, onRetry, manualPath }) {
  return (
    <div role="alert" className="ai-fallback">
      <p className="fallback-message">{error.userMessage}</p>
      <div className="fallback-actions">
        {error.retryable && (
          <button onClick={onRetry}>Try Again</button>
        )}
        {manualPath && (
          <a href={manualPath} className="manual-fallback">
            Do this manually instead
          </a>
        )}
      </div>
    </div>
  );
}
```

The "do this manually instead" link is not a failure — it is a safety net that makes users willing to try the AI path in the first place.

## Pattern 5: Feedback Loops

Every AI interface should capture user feedback, but the mechanism matters. Thumbs up/down is the minimum viable feedback loop. Corrections are gold.

```tsx
function AIResponseWithFeedback({ response, onFeedback }) {
  const [feedbackGiven, setFeedbackGiven] = useState(null);
  const [correction, setCorrection] = useState("");

  const submitFeedback = (type) => {
    setFeedbackGiven(type);
    onFeedback({
      type,
      responseId: response.id,
      correction: type === "negative" ? correction : null,
      timestamp: Date.now(),
    });
  };

  return (
    <div className="ai-response">
      <div className="response-content">{response.text}</div>

      {!feedbackGiven && (
        <div className="feedback-controls" role="group" aria-label="Rate this response">
          <button
            onClick={() => submitFeedback("positive")}
            aria-label="This response was helpful"
          >
            Helpful
          </button>
          <button
            onClick={() => submitFeedback("negative")}
            aria-label="This response was not helpful"
          >
            Not helpful
          </button>
        </div>
      )}

      {feedbackGiven === "negative" && (
        <div className="correction-input">
          <label htmlFor="correction">
            What would a better response look like?
          </label>
          <textarea
            id="correction"
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            placeholder="Describe what you expected..."
          />
          <button onClick={() => submitFeedback("correction")}>
            Submit Correction
          </button>
        </div>
      )}

      {feedbackGiven && (
        <p className="feedback-thanks" role="status">
          Thanks for the feedback.
        </p>
      )}
    </div>
  );
}
```

What you do with feedback data matters more than collecting it. At minimum, log feedback alongside the prompt, response, and model version. This gives you a dataset for evaluation. At best, negative feedback with corrections becomes fine-tuning data or few-shot examples for prompt improvement. Aggregate feedback by topic or query type to find systematic failures — those are where your next model improvement will have the most impact.

## Pattern 6: Inline Editing

Letting users edit AI output directly is the highest-signal feedback mechanism you have. Every edit tells you exactly where the model fell short and what the correct output should have been.

```tsx
function EditableAIOutput({ initialText, onSave, responseId }) {
  const [text, setText] = useState(initialText);
  const [isEditing, setIsEditing] = useState(false);

  const handleSave = () => {
    const hasEdits = text !== initialText;
    onSave({
      responseId,
      finalText: text,
      wasEdited: hasEdits,
      originalText: initialText,
      editDistance: hasEdits ? computeEditDistance(initialText, text) : 0,
    });
    setIsEditing(false);
  };

  return (
    <div className="editable-output">
      {isEditing ? (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="edit-area"
          aria-label="Edit AI-generated content"
        />
      ) : (
        <div
          className="display-text"
          onClick={() => setIsEditing(true)}
          role="button"
          tabIndex={0}
          aria-label="Click to edit AI-generated content"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") setIsEditing(true);
          }}
        >
          {text}
        </div>
      )}
      <div className="edit-actions">
        {isEditing ? (
          <>
            <button onClick={handleSave}>Save</button>
            <button onClick={() => { setText(initialText); setIsEditing(false); }}>
              Cancel
            </button>
          </>
        ) : (
          <button onClick={() => setIsEditing(true)}>Edit</button>
        )}
      </div>
    </div>
  );
}
```

Track the edit distance (how much the user changed) and the location of edits. If users consistently rewrite the opening sentence, that tells you something specific about your prompt or model behavior. Aggregate edit patterns are more valuable than individual corrections.

## Pattern 7: Regeneration

"Try again" is deceptively simple. A naive implementation just re-runs the same prompt and often gets a similar result, which frustrates the user. Effective regeneration varies the approach.

```tsx
function RegeneratableResponse({ prompt, onGenerate }) {
  const [responses, setResponses] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);

  const strategies = [
    { label: "Try again", config: { temperature: 0.7 } },
    { label: "More creative", config: { temperature: 1.0 } },
    { label: "More precise", config: { temperature: 0.3 } },
    { label: "Different approach", config: { systemPrompt: "alternative" } },
  ];

  const regenerate = async (strategy) => {
    const result = await onGenerate(prompt, strategy.config);
    setResponses((prev) => [...prev, result]);
    setActiveIndex(responses.length);
  };

  return (
    <div className="regeneratable-response">
      {responses.length > 0 && (
        <div className="response-display">
          <p>{responses[activeIndex].text}</p>
          {responses.length > 1 && (
            <div className="response-nav" role="tablist" aria-label="Response variations">
              {responses.map((_, i) => (
                <button
                  key={i}
                  role="tab"
                  aria-selected={i === activeIndex}
                  onClick={() => setActiveIndex(i)}
                >
                  Version {i + 1}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="regenerate-actions">
        {strategies.map((strategy) => (
          <button
            key={strategy.label}
            onClick={() => regenerate(strategy)}
          >
            {strategy.label}
          </button>
        ))}
      </div>
    </div>
  );
}
```

Keep previous generations available so the user can compare and pick the best one. This also gives you preference data — which generation the user ultimately selected tells you what "good" looks like for that prompt.

## Accessibility

AI interfaces introduce accessibility challenges that traditional interfaces do not. Streaming text, dynamically generated content, and confidence indicators all need careful attention.

**ARIA labels for AI-generated content.** Screen readers need to know that content was AI-generated, because that changes how the user should interpret it.

```html
<div role="region" aria-label="AI-generated summary — review for accuracy">
  <p>The quarterly revenue increased by 12% compared to last year...</p>
</div>
```

**Live regions for streaming responses.** When text streams token by token, screen readers need to be told about updates without reading the entire block every time.

```html
<div aria-live="polite" aria-atomic="false" aria-relevant="additions">
  <!-- Streaming tokens append here -->
</div>
```

Use `aria-live="polite"` so the screen reader waits for a pause before announcing new content. Use `aria-atomic="false"` so it only announces the new additions, not the entire region. For status updates like "Generating..." or "Complete," use `role="status"`.

**Reduced motion.** Typing indicators, streaming animations, and loading spinners should respect `prefers-reduced-motion`. Replace animations with static indicators for users who have enabled this preference.

```css
@media (prefers-reduced-motion: reduce) {
  .typing-indicator {
    animation: none;
  }
  .typing-indicator::after {
    content: "Generating response...";
  }
  .streaming-cursor {
    animation: none;
    border-color: transparent;
  }
}
```

**Keyboard navigation.** Every AI interaction — accepting suggestions, providing feedback, navigating between regenerated responses, editing outputs — must be fully operable via keyboard. Tab order should follow logical reading order: response content first, then feedback controls, then regeneration options.

**Screen reader considerations.** When an AI suggestion appears (such as inline code completion), announce it without disrupting the user's current context. Provide a keyboard shortcut to hear the suggestion read aloud and another to accept or dismiss it. Never auto-focus an AI suggestion in a way that pulls the screen reader away from where the user was working.

## Mobile Considerations

AI interfaces on mobile face three constraints that desktop does not: limited screen space, touch input, and variable bandwidth.

**Touch-friendly interactions.** Feedback buttons (thumbs up/down) need at minimum 44x44 pixel touch targets. Swipe gestures can supplement button taps — swipe right to accept a suggestion, swipe left to dismiss — but must never be the only input method.

**Smaller screens.** Progressive disclosure becomes even more important. On mobile, show only the summary by default. Collapse feedback controls behind a single "..." menu. For inline editing, switch to a full-screen editor rather than trying to fit a textarea into a compact card layout.

**Bandwidth constraints.** Streaming responses token by token works well on fast connections but can create a janky experience on slow mobile networks. Consider batching updates — instead of rendering each token as it arrives, buffer tokens and render in small chunks at regular intervals. This smooths out the visual experience and reduces render cycles.

For mobile-first AI interfaces, consider whether the AI interaction should be synchronous at all. On slow connections, an asynchronous pattern — "We're working on this, we'll notify you when it's ready" — can be a better experience than watching a slow stream.

---

These patterns are not a checklist to implement blindly. They are a vocabulary for making design decisions about AI interfaces. The right combination depends on your users, your domain, and the reliability of your model. What is universal: your users will encounter AI errors, and your interface must make those errors recoverable, not catastrophic. Design for the failure case first, and the happy path will take care of itself.
