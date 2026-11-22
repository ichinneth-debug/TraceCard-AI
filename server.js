'use strict';

require('dotenv').config();
const express = require('express');
const path = require('path');

// Node.js 18+ required for the built-in fetch API.
const app = express();
app.disable('x-powered-by');

const PORT = Number(process.env.PORT || 3000);
const PROVIDERS = [
  {
    id: 'gemini',
    key: process.env.GEMINI_API_KEY?.trim(),
    model: process.env.GEMINI_MODEL?.trim() || 'gemini-3.6-flash',
  },
  {
    id: 'groq',
    key: process.env.GROQ_API_KEY?.trim(),
    model: process.env.GROQ_MODEL?.trim() || 'openai/gpt-oss-20b',
  },
  {
    id: 'openrouter',
    key: process.env.OPENROUTER_API_KEY?.trim(),
    model: process.env.OPENROUTER_MODEL?.trim() || 'openrouter/free',
  },
];

function boundedInteger(name, fallback, minimum, maximum) {
  const value = Number(process.env[name]);
  if (!Number.isFinite(value)) return fallback;
  return Math.max(minimum, Math.min(maximum, Math.floor(value)));
}

const MAX_ACTIVE = boundedInteger('AI_CONCURRENCY', 3, 1, 20);
const MAX_QUEUE = boundedInteger('AI_QUEUE_LIMIT', 80, 1, 500);
const MAX_QUEUE_WAIT_MS = boundedInteger('AI_QUEUE_WAIT_MS', 180000, 5000, 600000);
const PROVIDER_TIMEOUT_MS = boundedInteger('AI_TIMEOUT_MS', 45000, 5000, 120000);
const MAX_PROMPT_CHARS = boundedInteger('MAX_PROMPT_CHARS', 80000, 1000, 200000);

const cooldownUntil = new Map();
const queue = [];
let active = 0;

app.use(express.json({ limit: '2mb' }));

class AIError extends Error {
  constructor(message, status = 503) {
    super(message);
    this.status = status;
  }
}

function queueTask(task) {
  return new Promise((resolve, reject) => {
    if (queue.length >= MAX_QUEUE) {
      reject(new AIError('AI server is busy. Please try again in a moment.', 503));
      return;
    }
    queue.push({ task, resolve, reject, addedAt: Date.now() });
    runQueue();
  });
}

function runQueue() {
  while (active < MAX_ACTIVE && queue.length) {
    const job = queue.shift();
    if (Date.now() - job.addedAt > MAX_QUEUE_WAIT_MS) {
      job.reject(new AIError('Generation queue took too long. Please try again.', 503));
      continue;
    }
    active++;
    Promise.resolve()
      .then(job.task)
      .then(job.resolve, job.reject)
      .finally(() => {
        active--;
        runQueue();
      });
  }
}

// Accept either a JSON array or { flashcards: [...] } from AI providers,
// but ALWAYS return a JSON array to the unchanged TraceCard frontend.
function parseFlashcards(raw) {
  if (typeof raw !== 'string' || !raw.trim()) {
    throw new AIError('AI response contained no text.', 502);
  }
  let text = raw.trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/, '')
    .trim();

  let value;
  try {
    value = JSON.parse(text);
  } catch {
    // Some free models add a sentence around valid JSON.
    const start = Math.min(...[text.indexOf('['), text.indexOf('{')].filter(i => i >= 0));
    const end = Math.max(text.lastIndexOf(']'), text.lastIndexOf('}'));
    if (!Number.isFinite(start) || end <= start) {
      throw new AIError('Provider did not return valid flashcard JSON.', 502);
    }
    try {
      value = JSON.parse(text.slice(start, end + 1));
    } catch {
      throw new AIError('Provider did not return valid flashcard JSON.', 502);
    }
  }

  const list = Array.isArray(value) ? value
    : Array.isArray(value?.flashcards) ? value.flashcards
    : Array.isArray(value?.cards) ? value.cards : null;

  if (!list || !list.length) {
    throw new AIError('Provider returned no flashcards.', 502);
  }

  // Prevent malformed AI data from breaking the existing UI.
  const valid = list.filter(card => {
    if (!card || typeof card !== 'object') return false;
    const type = ['mcq','identification','enumeration','fillblank'].includes(card.type) ? card.type : 'mcq';
    const question = String(card.question || '').trim();
    const answer = Array.isArray(card.answer) ? card.answer.map(x => String(x).trim()).filter(Boolean) : String(card.answer || '').trim();
    const excerpt = String(card.source_excerpt || '').trim();
    const page = Number(card.page);
    const choices = Array.isArray(card.choices) ? card.choices.map(x => String(x || '').trim()) : [];
    if (!question || !answer || excerpt.length < 10 || excerpt.length > 160 || !Number.isInteger(page) || page < 1) return false;
    if (type === 'fillblank') return (question.match(/____/g) || []).length === 1 && typeof answer === 'string' && answer.length <= 100;
    if (type === 'identification') return typeof answer === 'string';
    if (type === 'enumeration') return Array.isArray(answer) ? answer.length >= 2 : String(answer).split(/[;,\\n]/).filter(Boolean).length >= 2;
    return typeof answer === 'string' && choices.length === 4 && choices.every(Boolean) &&
      new Set(choices.map(x => x.toLowerCase())).size === 4 &&
      choices.some(x => x.toLowerCase() === answer.toLowerCase());
  });

  if (!valid.length) {
    throw new AIError('Provider returned flashcards that failed format validation.', 502);
  }

  return valid.map(card => ({...card, type: ['identification','enumeration'].includes(card.type) ? card.type : 'mcq', answer: Array.isArray(card.answer) ? card.answer.map(String).join('; ') : String(card.answer)}));
}

function extractGeminiText(body) {
  if (typeof body?.output_text === 'string' && body.output_text.trim()) {
    return body.output_text;
  }
  const steps = body?.steps || body?.interaction?.steps || [];
  const texts = [];
  for (const step of steps) {
    if (step?.type !== 'model_output') continue;
    for (const item of step.content || []) {
      if (item?.type === 'text' && typeof item.text === 'string') {
        texts.push(item.text);
      }
    }
  }
  return texts.join('').trim();
}

async function postJSON(provider, url, body, headers) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), PROVIDER_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    // Read the response while the timeout is still active.
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch { data = {}; }

    if (!response.ok) {
      const err = new AIError(`${provider.id} HTTP ${response.status}`, response.status);
      const retryAfter = response.headers.get('retry-after');
      const retrySeconds = Number(retryAfter);
      if (Number.isFinite(retrySeconds) && retrySeconds > 0) {
        err.retryAfterMs = retrySeconds * 1000;
      } else if (retryAfter) {
        const dateMs = Date.parse(retryAfter) - Date.now();
        if (Number.isFinite(dateMs) && dateMs > 0) err.retryAfterMs = dateMs;
      }
      throw err;
    }
    return data;
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new AIError(`${provider.id} timed out.`, 504);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function askGemini(provider, prompt) {
  const data = await postJSON(provider,
    'https://generativelanguage.googleapis.com/v1beta/interactions',
    {
      model: provider.model,
      input: prompt,
      generation_config: { thinking_level: 'low' },
    },
    { 'x-goog-api-key': provider.key },
  );
  return extractGeminiText(data);
}

// Groq and OpenRouter both accept OpenAI-compatible chat completion requests.
async function askChatCompletion(provider, prompt) {
  const url = provider.id === 'groq'
    ? 'https://api.groq.com/openai/v1/chat/completions'
    : 'https://openrouter.ai/api/v1/chat/completions';

  const jsonInstructions =
    'You generate accurate source-linked flashcards of the question type requested by the user. ' +
    'Return ONLY a valid JSON object with a "flashcards" array. ' +
    'Each card must contain type, question, answer (string, or array for enumeration), choices (4 distinct strings for MCQ only), ' +
    'page (integer), and source_excerpt (exact 10-160 character quote from the provided page). ' +
    'For fillblank, use exactly one ____ placeholder in the question. Follow the user requirements and use only the provided source text.';

  const body = {
    model: provider.model,
    messages: [
      { role: 'system', content: jsonInstructions },
      { role: 'user', content: prompt + '\n\nReturn the results as {"flashcards":[...]}, not a bare array.' },
    ],
    temperature: 0.2,
    max_tokens: 8192,
  };
  // Supported by the recommended GPT-OSS model on Groq.
  // OpenRouter's free router selects models dynamically, so omit JSON mode there.
  if (provider.id === 'groq') {
    body.response_format = { type: 'json_object' };
  }

  const data = await postJSON(provider, url, body, {
    Authorization: `Bearer ${provider.key}`,
    ...(provider.id === 'openrouter' ? { 'X-Title': 'TraceCard AI' } : {}),
  });
  const content = data?.choices?.[0]?.message?.content;
  return typeof content === 'string' ? content : '';
}

function recordFailure(provider, error) {
  const status = Number(error?.status);
  let cooldownMs = 0;
  if (status === 429) cooldownMs = Math.max(60000, error.retryAfterMs || 0);
  else if ([400, 401, 402, 403, 404].includes(status)) cooldownMs = 120000;
  else if ([500, 502, 503, 504].includes(status)) cooldownMs = 10000;
  if (cooldownMs) cooldownUntil.set(provider.id, Date.now() + Math.min(cooldownMs, 300000));
  // Do not log prompts, provider response bodies, or API keys.
  console.warn(`[TraceCard AI] ${provider.id} failed: ${status || 'network/format error'}`);
}

async function generateFlashcards(prompt) {
  let tried = 0;
  for (const provider of PROVIDERS) {
    if (!provider.key) continue;
    if ((cooldownUntil.get(provider.id) || 0) > Date.now()) continue;
    tried++;
    try {
      const raw = provider.id === 'gemini'
        ? await askGemini(provider, prompt)
        : await askChatCompletion(provider, prompt);
      const cards = parseFlashcards(raw);
      console.info(`[TraceCard AI] Generated ${cards.length} cards with ${provider.id}`);
      return cards;
    } catch (error) {
      recordFailure(provider, error);
      // Continue to next configured provider automatically.
    }
  }
  const configured = PROVIDERS.some(p => p.key);
  if (!configured) {
    throw new AIError('No AI provider configured. Add API keys to your server .env file.', 503);
  }
  if (!tried) {
    throw new AIError('All AI providers are temporarily cooling down. Try again shortly.', 503);
  }
  throw new AIError('All AI providers failed or are unavailable. Please try again later.', 503);
}

app.get('/api/health', (req, res) => {
  res.set('Cache-Control', 'no-store');
  res.json({
    ok: true,
    aiConfigured: PROVIDERS.some(p => Boolean(p.key)),
    providers: PROVIDERS.map(p => ({
      name: p.id,
      configured: Boolean(p.key),
      available: Boolean(p.key) && (cooldownUntil.get(p.id) || 0) <= Date.now(),
    })),
    concurrency: { active, queued: queue.length, maximum: MAX_ACTIVE },
  });
});

async function handleGenerate(req, res) {
  const prompt = req.body?.prompt;
  if (typeof prompt !== 'string' || !prompt.trim()) {
    return res.status(400).json({ error: 'Missing AI prompt.' });
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    return res.status(413).json({ error: 'AI prompt is too long. Try fewer PDF pages.' });
  }
  try {
    const cards = await queueTask(() => generateFlashcards(prompt.trim()));
    res.set('Cache-Control', 'no-store');
    res.json(cards);
  } catch (error) {
    res.status(error?.status === 503 ? 503 : 500).json({
      error: error?.message || 'Flashcard generation failed.',
    });
  }
}

// Keep the frontend's original URL working without editing index.html.
app.post('/api/gemini', handleGenerate);
app.post('/api/flashcards', handleGenerate);

// Serve the current standalone frontend; do NOT expose .env, server.js,
// package.json, node_modules, or other project files as public static files.
app.get(['/', '/index.html'], (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});
app.use('/assets', express.static(path.join(__dirname, 'assets'), { dotfiles: 'deny' }));
app.use((req, res) => res.status(404).json({ error: 'Not found.' }));

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`TraceCard AI running at http://localhost:${PORT}`);
    console.log('Configured AI providers:', PROVIDERS.filter(p => p.key).map(p => p.id).join(' -> ') || 'none');
  });
}

module.exports = { app, parseFlashcards, extractGeminiText, generateFlashcards };
