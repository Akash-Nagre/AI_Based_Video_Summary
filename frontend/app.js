const $ = (id) => document.getElementById(id);
const input = $('videoInput');
const dropzone = $('dropzone');
const startButton = $('startButton');
let selectedFile = null;
let toastTimer;
let currentJobId = null;

function showToast(message) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 5000);
}

function prettySize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setFile(file) {
  if (!file) return;
  const allowed = /\.(mp4|mov|avi|mkv|webm)$/i;
  if (!allowed.test(file.name)) {
    showToast('Please choose an MP4, MOV, AVI, MKV, or WEBM video.');
    return;
  }
  selectedFile = file;
  $('fileName').textContent = file.name;
  $('fileSize').textContent = prettySize(file.size);
  $('fileDetail').classList.remove('hidden');
  $('filePrompt').textContent = 'Video ready to summarize';
  startButton.disabled = false;
  $('startHint').textContent = 'Your video is ready. Choose “Make my summary” when you’re ready.';
}

input.addEventListener('change', () => setFile(input.files[0]));
$('removeFile').addEventListener('click', () => {
  selectedFile = null;
  input.value = '';
  $('fileDetail').classList.add('hidden');
  $('filePrompt').textContent = 'Drop your video here';
  startButton.disabled = true;
  $('startHint').textContent = 'Choose a video to enable summary generation.';
});

for (const eventName of ['dragenter', 'dragover']) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add('dragging'); });
}
for (const eventName of ['dragleave', 'drop']) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); });
}
dropzone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));

function setRange(value) {
  const min = 15, max = 40;
  const percent = ((value - min) / (max - min)) * 100;
  $('rangeValue').textContent = value;
  $('summaryLength').style.background = `linear-gradient(to right,#a8c84b 0%,#a8c84b ${percent}%,#e8eadf ${percent}%,#e8eadf 100%)`;
}
$('summaryLength').addEventListener('input', (event) => setRange(Number(event.target.value)));
setRange(25);

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function showProgress(filename) {
  $('uploadPanel').classList.add('hidden');
  document.querySelector('.side-column').classList.add('hidden');
  $('progressCard').classList.remove('hidden');
  $('progressFilename').textContent = filename;
  $('stageText').textContent = 'Uploading your video…';
  $('progressTitle').textContent = 'Working on your summary';
  $('progressBar').style.width = '4%';
  $('progressPercent').textContent = '4%';
  let displayed = 4;
  const timer = setInterval(() => {
    if ($('progressCard').classList.contains('hidden')) return clearInterval(timer);
    if (displayed < 90) displayed += Math.max(1, Math.round((90 - displayed) * 0.045));
    $('progressBar').style.width = `${displayed}%`;
    $('progressPercent').textContent = `${displayed}%`;
  }, 1200);
  return timer;
}

function formatDuration(seconds) {
  if (!Number.isFinite(Number(seconds))) return '—';
  const total = Math.round(Number(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function formatTimestamp(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remainder = value % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
    : `${minutes}:${String(remainder).padStart(2, '0')}`;
}

function renderTextSummary(payload) {
  const content = $('textSummaryContent');
  content.replaceChildren();
  const summaries = Array.isArray(payload.summaries) ? payload.summaries : [];
  if (!summaries.length) {
    const empty = document.createElement('p');
    empty.className = 'text-summary-empty';
    empty.textContent = 'No text summary was returned for this video.';
    content.append(empty);
    return;
  }

  for (const summary of summaries) {
    const section = document.createElement('article');
    section.className = 'text-summary-item';
    const heading = document.createElement('div');
    heading.className = 'text-summary-item-heading';
    const title = document.createElement('span');
    title.className = 'text-summary-topic';
    title.textContent = summary.label || summary.language || 'Summary';
    heading.append(title);
    const paragraph = document.createElement('p');
    paragraph.className = 'generated-summary-text';
    paragraph.textContent = summary.text || '';
    section.append(heading, paragraph);
    content.append(section);
  }
}

function renderResult(payload, filename, jobId) {
  const result = payload.result || {};
  const report = payload.coverage_report || result.coverage_report || {};
  currentJobId = jobId;
  $('progressCard').classList.add('hidden');
  $('resultCard').classList.remove('hidden');
  $('resultName').textContent = filename;
  $('progressBar').style.width = '100%';
  $('progressPercent').textContent = '100%';
  $('summaryVideo').src = `/download/${encodeURIComponent(jobId)}`;
  $('downloadLink').href = `/download/${encodeURIComponent(jobId)}`;
  const ratio = Number(report.achieved_compression_ratio);
  const selected = result.selected_segments || report.selected_clips || [];
  const actual = report.actual_summary_duration_seconds;
  $('resultStats').textContent = `${selected.length} HIGHLIGHTS  ·  ${formatDuration(actual)} SUMMARY${Number.isFinite(ratio) ? `  ·  ${Math.round(ratio * 100)}% OF ORIGINAL` : ''}`;

  const highlights = $('highlightsList');
  highlights.replaceChildren();
  let summaryOffset = 0;
  for (const [index, clip] of selected.entries()) {
    const start = Number(clip.start) || 0;
    const end = Number(clip.end) || start;
    const duration = Math.max(0, end - start);
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'highlight-item';
    item.setAttribute('aria-label', `Jump to highlight ${index + 1}, from ${formatTimestamp(start)} to ${formatTimestamp(end)}`);
    item.dataset.offset = String(summaryOffset);
    const number = document.createElement('span');
    number.className = 'highlight-number';
    number.textContent = String(index + 1).padStart(2, '0');
    const content = document.createElement('span');
    content.className = 'highlight-content';
    const time = document.createElement('span');
    time.className = 'highlight-time';
    time.textContent = `${formatTimestamp(start)} – ${formatTimestamp(end)}`;
    const excerpt = document.createElement('span');
    excerpt.className = 'highlight-excerpt';
    excerpt.textContent = clip.text || 'Selected highlight';
    content.append(time, excerpt);
    const arrow = document.createElement('span');
    arrow.className = 'highlight-arrow';
    arrow.textContent = '↗';
    item.append(number, content, arrow);
    item.addEventListener('click', () => {
      const video = $('summaryVideo');
      video.currentTime = Number(item.dataset.offset);
      video.play().catch(() => {});
    });
    highlights.append(item);
    summaryOffset += duration;
  }
  if (!selected.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-highlights';
    empty.textContent = 'No individual timestamps were included in this result.';
    highlights.append(empty);
  }

  const topics = Array.isArray(report.topics) ? report.topics : [];
  $('topicsList').replaceChildren();
  $('topicsSection').classList.toggle('hidden', topics.length === 0);
  for (const topic of topics) {
    const pill = document.createElement('span');
    pill.className = 'topic-pill';
    const label = topic.label || topic.topic_id || 'Topic';
    pill.append(document.createTextNode(label));
    const state = document.createElement('small');
    state.textContent = topic.coverage_status || '';
    pill.append(state);
    $('topicsList').append(pill);
  }
  $('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function runJob(file) {
  startButton.disabled = true;
  $('startHint').textContent = 'Preparing your summary…';
  const progressTimer = showProgress(file.name);
  try {
    $('stageText').textContent = 'Uploading your video…';
    const form = new FormData();
    form.append('file', file);
    const uploaded = await request('/upload', { method: 'POST', body: form });
    $('stageText').textContent = 'Finding the key moments in your video…';
    const fraction = Number($('summaryLength').value) / 100;
    await request(`/summarize/${encodeURIComponent(uploaded.job_id)}?target_summary_fraction=${fraction}`, { method: 'POST' });

    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 2200));
      const status = await request(`/status/${encodeURIComponent(uploaded.job_id)}`);
      if (status.status === 'failed') throw new Error(status.error || 'Video processing failed.');
      if (status.status === 'completed') break;
      $('stageText').textContent = status.stage && status.stage !== 'starting' ? `Processing: ${status.stage.replaceAll('_', ' ')}…` : 'Transcribing and selecting your highlights…';
    }
    const result = await request(`/result/${encodeURIComponent(uploaded.job_id)}`);
    clearInterval(progressTimer);
    renderResult(result, file.name, uploaded.job_id);
  } catch (error) {
    clearInterval(progressTimer);
    $('progressCard').classList.add('hidden');
    $('uploadPanel').classList.remove('hidden');
    document.querySelector('.side-column').classList.remove('hidden');
    startButton.disabled = !selectedFile;
    $('startHint').textContent = selectedFile
      ? 'Your video is ready. You can try generating the summary again.'
      : 'Choose a video to enable summary generation.';
    showToast(error.message || 'Something went wrong. Please try again.');
  }
}

startButton.addEventListener('click', () => { if (selectedFile) runJob(selectedFile); });
$('newVideo').addEventListener('click', () => {
  $('resultCard').classList.add('hidden');
  $('uploadPanel').classList.remove('hidden');
  document.querySelector('.side-column').classList.remove('hidden');
  $('summaryVideo').pause();
  $('summaryVideo').removeAttribute('src');
  $('summaryVideo').load();
  selectedFile = null;
  input.value = '';
  $('fileDetail').classList.add('hidden');
  $('filePrompt').textContent = 'Drop your video here';
  startButton.disabled = true;
  $('startHint').textContent = 'Choose a video to enable summary generation.';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

$('openTextSummary').addEventListener('click', async () => {
  const dialog = $('textSummaryDialog');
  const content = $('textSummaryContent');
  const button = $('openTextSummary');
  if (!currentJobId) return;
  dialog.showModal();
  content.replaceChildren();
  const loading = document.createElement('p');
  loading.className = 'text-summary-empty';
  loading.textContent = 'Collecting the complete transcript of the clips in your generated video.';
  content.append(loading);
  button.disabled = true;
  try {
    const summary = await request(`/text-summary/${encodeURIComponent(currentJobId)}`, { method: 'POST' });
    renderTextSummary(summary);
  } catch (error) {
    content.replaceChildren();
    const message = document.createElement('p');
    message.className = 'text-summary-empty';
    message.textContent = error.message || 'Could not generate a text summary.';
    content.append(message);
  } finally {
    button.disabled = false;
  }
});
$('closeTextSummary').addEventListener('click', () => $('textSummaryDialog').close());
$('textSummaryDialog').addEventListener('click', (event) => {
  if (event.target === $('textSummaryDialog')) $('textSummaryDialog').close();
});
