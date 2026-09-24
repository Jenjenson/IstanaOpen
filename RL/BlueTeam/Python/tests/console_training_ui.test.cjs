// Behavioral checks for replay selection and evidence labels, without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(text = '') { this.textContent = text; this.children = []; this.dataset = {}; this.style = {}; this.value = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
}

async function main() {
  const source = fs.readFileSync(path.join(__dirname, '../console/app.js'), 'utf8');
  const nodes = new Map(), requests = [], shown = [], errors = [];
  const context = {
    $: id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
    element: (_tag, text) => new Element(text),
    fmt: (value, places = 1) => Number.isFinite(value) ? value.toFixed(places) : '—',
    pct: value => Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—',
    mode: 'training', playing: false, view: null, elapsed: 0, lastWall: 0,
    training: { status: {}, replaySelection: 'latest', replaySequence: 0, replayLoading: false, replayViewer: null, historySignature: '', lastViewerEpisode: -1 },
    controls() {}, connectionStatus() {}, drawTrainingCharts() {}, drawTrainingRewardLog() {},
    showError: error => { if (error) errors.push(error); },
    pause: () => { context.playing = false; },
    setView: viewer => { shown.push(viewer); context.view = viewer; },
    api: async url => { requests.push(url); return { viewer: { mode: 'training', label: 'Episode 500', frames: [{ time: 0 }] }, label: 'Sampled episode 500' }; },
  };
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf('function trainingReplayLabel('), source.indexOf('async function refreshTraining(')), context);
  const latestViewer = { mode: 'training', frames: [{ time: 0 }] };
  const status = {
    outputDirectory: '/run/one', phase: 'training', running: true, episode: 1000, totalEpisodes: 1500,
    baselineEvaluation: { meanWarningSeconds: 4 }, initialEvaluation: { meanWarningSeconds: 4 },
    bestValidationWarningSeconds: 4, bestEpisode: 0, validationCases: 5,
    exploration: { uniqueLayouts: 987, sensorsChangedFromInitial: 2, meanNormalizedEntropy: 0.7, explorationProbability: 0.2 },
    baselineProbe: { candidates: 4, bestDeltaSeconds: -0.2, improvesBaseline: false },
    checkpoints: [0, 500, 1000], viewer: latestViewer,
    history: Array.from({ length: 1000 }, (_, index) => ({ episode: index + 1, meanWarningSeconds: 9, detectedFraction: 1, sensorsChangedFromInitial: 2, validationWarningSeconds: null })),
  };
  context.applyTrainingStatus(status);
  assert.equal(nodes.get('training-best-warning').textContent, '4.00 s');
  assert.equal(nodes.get('training-validation-delta').textContent, '0.00 s vs contractor · episode 0');
  assert.equal(nodes.get('training-sample-warning').textContent, '9.00 s');
  assert.match(nodes.get('training-sample-note').textContent, /one sampled scenario/);
  assert.match(nodes.get('training-probe-summary').textContent, /did not improve/);
  assert.equal(nodes.get('training-history-rows').children.length, 100);
  assert.match(nodes.get('training-history-count').textContent, /1000 episodes · latest 100 shown/);
  assert.equal(nodes.get('training-history-rows').children[0].children[4].textContent, 'Not evaluated');
  assert.match(nodes.get('training-exploration-summary').textContent, /placement order ignored/);
  assert.ok(nodes.get('training-replay').children.some(option => option.value === 'checkpoint-500'));
  assert.ok(nodes.get('training-replay').children.some(option => option.value === '500'));

  await context.selectTrainingReplay('checkpoint-500');
  assert.equal(requests.at(-1), '/api/training/replay/checkpoint-500');
  assert.equal(context.training.replaySelection, 'checkpoint-500');
  assert.match(nodes.get('training-replay-note').textContent, /cards show scenario averages/);

  await context.selectTrainingReplay('500');
  assert.equal(requests.at(-1), '/api/training/replay/500');
  const pinnedViewer = context.view;
  context.applyTrainingStatus({ ...status, episode: 1001, viewer: { mode: 'training', label: 'New latest' } });
  assert.equal(context.view, pinnedViewer, 'polling must not replace a selected earlier episode');
  assert.match(nodes.get('training-replay-note').textContent, /stays selected/);
  assert.equal(nodes.get('training-replay').value, '500');

  const countBeforeInvalid = requests.length;
  await context.selectTrainingReplay('1500');
  assert.equal(requests.length, countBeforeInvalid, 'incomplete episode must not be requested');
  assert.match(errors.at(-1), /completed episode/);

  context.api = async () => { throw new Error('Replay unavailable'); };
  await context.selectTrainingReplay('999');
  assert.equal(context.view, pinnedViewer, 'failed fetch preserves the selected replay');
  assert.equal(context.training.replaySelection, '500');
  await context.selectTrainingReplay('latest');
  assert.equal(context.training.replaySelection, 'latest');
  assert.equal(context.view.label, 'New latest');

  context.mode = 'recorded';
  const callsBeforeOtherMode = shown.length;
  context.applyTrainingStatus({ ...status, episode: 1002 });
  assert.equal(shown.length, callsBeforeOtherMode, 'training status must not overwrite another mode');

  context.applyTrainingStatus({ ...status, initialEvaluation: null, baselineEvaluation: null, bestValidationWarningSeconds: null });
  assert.equal(nodes.get('training-baseline-warning').textContent, 'Awaiting evaluation');
  assert.equal(nodes.get('training-best-warning').textContent, 'Awaiting evaluation');
  assert.equal(nodes.get('training-test-warning').textContent, 'Not tested yet');
  context.applyTrainingStatus({ phase: 'idle', initializationLabel: 'Untrained random policy', history: [] });
  assert.equal(nodes.get('training-status').textContent, 'Ready to train. Choose a starting placement and settings above.');
  assert.doesNotMatch(nodes.get('training-status').textContent, /Untrained random policy/);
  process.stdout.write('Training evidence and replay behavior checks passed.\n');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
