'use strict';

require('dotenv').config();
const express = require('express');
const path = require('path');

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

const CARD_TYPES = new Set(['mcq', 'identification', 'enumeration', 'fillblank']);
const FIELDS = ['type', 'question', 'answer', 'choices', 'page', 'source_excerpt'];
function requestedCardType(prompt, suppliedSchema) {
  const fromSchema = suppliedSchema?.properties?.flashcards?.items?.properties?.type?.enum;
  if (Array.isArray(fromSchema) && fromSchema.length === 1 && CARD_TYPES.has(fromSchema[0])) {
    return fromSchema[0];
  }
  const fromPrompt = String(prompt).match(/(?:^|\n)Type:\s*(mcq|identification|enumeration|fillblank)\b/i);
  return fromPrompt ? fromPrompt[1].toLowerCase() : 'mcq';
}
function buildResponseSchema(type) {
  if (!CARD_TYPES.has(type)) throw new AIError('Unsupported question type.', 400);
  const answer = type === 'enumeration'
    ? { type: 'array', items: { type: 'string' }, minItems: 2, maxItems: 5 }
    : { type: 'string' };
  const choices = type === 'mcq'
    ? { type: 'array', items: { type: 'string' }, minItems: 4, maxItems: 4 }
    : { type: 'array', items: { type: 'string' }, maxItems: 0 };
  return {
    type: 'object', additionalProperties: false,
    properties: {
      flashcards: {
        type: 'array', items: {
          type: 'object', additionalProperties: false,
          properties: {
            type: { type: 'string', enum: [type] },
            question: { type: 'string' }, answer, choices,
            page: { type: 'integer' }, source_excerpt: { type: 'string' },
          },
          required: FIELDS,
        },
      },
    },
    required: ['flashcards'],
  };
}


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
function quotedInSource(prompt, page, excerpt) {
  if (typeof prompt !== 'string') return true;
  const marker = /^PAGE (\d+):\s*/gm;
  const spans = [...prompt.matchAll(marker)];
  if (!spans.length) return true; // Legacy API calls without page-formatted source.
  const normalize = value => String(value).replace(/\s+/g, ' ').trim().toLowerCase();
  for (let i = 0; i < spans.length; i++) {
    if (Number(spans[i][1]) !== page) continue;
    const start = spans[i].index + spans[i][0].length;
    const end = i + 1 < spans.length ? spans[i + 1].index : prompt.length;
    if (normalize(prompt.slice(start, end)).includes(normalize(excerpt))) return true;
  }
  return false;
}

function parseFlashcards(raw, requestedType = null, prompt = null) {
  let value = raw;
  if (typeof value === 'string') {
    const text = value.trim().replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
    if (!text) throw new AIError('AI response contained no text.', 502);
    try {
      value = JSON.parse(text);
    } catch {

      const locations = [text.indexOf('['), text.indexOf('{')].filter(n => n >= 0);
      const start = locations.length ? Math.min(...locations) : -1;
      const end = Math.max(text.lastIndexOf(']'), text.lastIndexOf('}'));
      if (start < 0 || end <= start) throw new AIError('Provider did not return valid flashcard JSON.', 502);
      try { value = JSON.parse(text.slice(start, end + 1)); }
      catch { throw new AIError('Provider did not return valid flashcard JSON.', 502); }
    }
  }
  const list = Array.isArray(value) ? value
    : Array.isArray(value?.flashcards) ? value.flashcards
    : Array.isArray(value?.cards) ? value.cards : null;
  if (!list?.length) throw new AIError('Provider returned no flashcards.', 502);

  const reasons = { format: 0, type: 0, answer: 0, source: 0 };
  const valid = [];
  const aliases = {
    multiple_choice: 'mcq', multiplechoice: 'mcq',
    fill_in_the_blank: 'fillblank', fill_in_blank: 'fillblank',
  };
  for (const input of list) {
    if (!input || typeof input !== 'object' || Array.isArray(input)) { reasons.format++; continue; }
    const rawType = input.type || input.question_type || input.questionType || requestedType || 'mcq';
    const type = aliases[rawType] || rawType;
    if (!CARD_TYPES.has(type) || (requestedType && type !== requestedType)) { reasons.type++; continue; }
    const question = String(input.question ?? input.q ?? '').trim();
    const excerpt = String(input.source_excerpt ?? input.sourceExcerpt ?? input.excerpt ?? '').replace(/\s+/g,' ').trim();
    const page = Number(input.page ?? input.page_number ?? input.pageNumber);
    const choiceInput = input.choices ?? input.opts ?? input.options ?? [];
    const choices = Array.isArray(choiceInput) ? choiceInput.map(x => String(x ?? '').trim()) : null;
    if (!question || !Number.isInteger(page) || page < 1 || !Array.isArray(choices)) {
      reasons.format++; continue;
    }
    if (excerpt.length < 10 || excerpt.length > 160 || !quotedInSource(prompt, page, excerpt)) {
      reasons.source++; continue;
    }
    let answer = input.answer ?? input.a;
    if (type === 'enumeration') {
      if (typeof answer === 'string') answer = answer.split(/[;\n]+/).map(x => x.trim()).filter(Boolean);
      if (!Array.isArray(answer)) { reasons.answer++; continue; }
      answer = answer.map(x => String(x ?? '').trim()).filter(Boolean);
      if (answer.length < 2 || answer.length > 5 || new Set(answer.map(x => x.toLowerCase())).size !== answer.length || choices.length) {
        reasons.answer++; continue;
      }
    } else {
      if (typeof answer !== 'string' || !answer.trim()) { reasons.answer++; continue; }
      answer = answer.trim();
      if (type === 'mcq') {
        if (choices.length !== 4 || choices.some(x => !x) ||
            new Set(choices.map(x => x.toLowerCase())).size !== 4 ||
            !choices.some(x => x.toLowerCase() === answer.toLowerCase())) {
          reasons.answer++; continue;
        }
      } else if (choices.length || (type === 'fillblank' &&
                 ((question.match(/____/g) || []).length !== 1 || answer.length > 100))) {
        reasons.answer++; continue;
      }
    }
    valid.push({ type, question, answer, choices, page, source_excerpt: excerpt });
  }
  // Only aggregate diagnostics: never print student PDF text, questions, or keys.
  if (valid.length < list.length) {
    console.warn(`[TraceCard AI] Provider validation: ${valid.length}/${list.length} accepted; rejected=${JSON.stringify(reasons)}`);
  }
  if (!valid.length) throw new AIError('Provider returned flashcards that failed format validation.', 502);
  return valid;
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

async function askGemini(provider, prompt, schema) {
  const data = await postJSON(provider,
    'https://generativelanguage.googleapis.com/v1beta/interactions',
    {
      model: provider.model,
      input: prompt,
      generation_config: { thinking_level: 'low' },
      // Gemini Interactions structured output setting is TOP LEVEL.
      response_format: { type: 'text', mime_type: 'application/json', schema },
    },
    { 'x-goog-api-key': provider.key },
  );
  return extractGeminiText(data);
}

// Groq and OpenRouter both accept OpenAI-compatible chat completion requests.
async function askChatCompletion(provider, prompt, schema) {
  const url = provider.id === 'groq'
    ? 'https://api.groq.com/openai/v1/chat/completions'
    : 'https://openrouter.ai/api/v1/chat/completions';
  const jsonInstructions =
    'You generate accurate flashcards from user-supplied PDF text only. ' +
    'Return ONLY a valid JSON object with one "flashcards" array. ' +
    'Preserve the requested question type. Each card needs type, question, answer, choices, page, source_excerpt. ' +
    'For MCQ use exactly 4 distinct answer choices. For identification and fillblank use a string answer and empty choices. ' +
    'For enumeration return an ARRAY of 2-5 strings and empty choices. ' +
    'For fillblank use exactly one ____ in the question. ' +
    'Copy source_excerpt exactly from the referenced page (10-160 characters). ' +
    'Never invent source text. Output must match this JSON Schema: ' + JSON.stringify(schema);
  const body = {
    model: provider.model,
    messages: [
      { role: 'system', content: jsonInstructions },
      { role: 'user', content: prompt },
    ],
    temperature: 0.2,
    max_tokens: 8192,
  };

  if (provider.id === 'groq') {
    // Strict JSON Schema support depends on the model; GPT-OSS 20B/120B are supported.
    const strictSupported = /^openai\/gpt-oss-(?:20b|120b)$/i.test(provider.model);
    body.response_format = strictSupported
      ? { type: 'json_schema', json_schema: { name: 'tracecard_flashcards', strict: true, schema } }
      : { type: 'json_object' };
  } else if (provider.id === 'openrouter' &&
             process.env.OPENROUTER_STRUCTURED_OUTPUTS === 'true' &&
             provider.model !== 'openrouter/free') {
    // Explicit opt-in: not all OpenRouter model/provider endpoints support schemas.
    body.response_format = {
      type: 'json_schema',
      json_schema: { name: 'tracecard_flashcards', strict: true, schema },
    };
  }
  const data = await postJSON(provider, url, body, {
    Authorization: `Bearer ${provider.key}`,
    ...(provider.id === 'openrouter' ? { 'X-Title': 'TraceCard AI' } : {}),
  });
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.filter(x => x?.type === 'text').map(x => x.text).join('');
  return '';
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

async function generateFlashcards(prompt, requestedType = 'mcq') {
  const schema = buildResponseSchema(requestedType);
  let tried = 0;
  for (const provider of PROVIDERS) {
    if (!provider.key) continue;
    if ((cooldownUntil.get(provider.id) || 0) > Date.now()) continue;
    tried++;
    try {
      const raw = provider.id === 'gemini'
        ? await askGemini(provider, prompt, schema)
        : await askChatCompletion(provider, prompt, schema);
      const cards = parseFlashcards(raw, requestedType, prompt);
      console.info(`[TraceCard AI] Generated ${cards.length} valid ${requestedType} cards with ${provider.id}`);
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
    const requestedType = requestedCardType(prompt, req.body?.responseSchema);
    const cards = await queueTask(() => generateFlashcards(prompt.trim(), requestedType));
    res.set('Cache-Control', 'no-store');
    res.json(cards);
  } catch (error) {
    const status = [400, 413, 429, 502, 503, 504].includes(error?.status) ? error.status : 500;
    res.status(status).json({
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

module.exports = { app, parseFlashcards, extractGeminiText, generateFlashcards, buildResponseSchema, requestedCardType };
