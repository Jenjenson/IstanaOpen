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
  get options() { return this.children.flatMap(child => child.children?.length ? child.options : [child]); }
  set options(rows) { this.children = rows; }
  remove() {}
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

  context.applyTrainingStatus({ ...status, algorithm: 'local_ppo', rewardMode: 'paired_contractor_delta',
    exploration: { ...status.exploration, localEdits: true, explorationProbability: null },
    history: [{ episode: 1000, meanWarningSeconds: 9, trainingReward: -2,
      pairedTraining: { contractorWarningSeconds: 11 } }] });
  assert.equal(nodes.get('training-sample-warning').textContent, '9.00 s');
  assert.match(nodes.get('training-sample-note').textContent, /contractor 11.00 s · gain -2.00 s/);
  assert.match(nodes.get('training-exploration-summary').textContent, /Local edits start with equal probabilities/);
  assert.doesNotMatch(nodes.get('training-exploration-summary').textContent, /Initial exploration target/);
  assert.match(nodes.get('training-initialization-help').textContent, /no probability advantage/);
  context.applyTrainingStatus(status);

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
  context.$('training-algorithm').value = 'local_ppo';
  context.$('training-validation').value = '8';
  context.applyTrainingStatus({ phase: 'idle', algorithm: 'reinforce', validationCases: 5, history: [] });
  assert.match(nodes.get('training-initialization-help').textContent, /no probability advantage/);
  assert.match(nodes.get('training-evaluation-note').textContent, /8 fixed validation scenarios/);
  context.$('training-algorithm').value = 'ppo';
  context.updateTrainingInitializationHelp();
  assert.match(nodes.get('training-initialization-help').textContent, /starting preference/);
  vm.runInContext(source.slice(source.indexOf('function isSavedLayout('), source.indexOf('function syncModelCatalog(')), context);
  context.comparison = { selectedPolicy: null, episodes: [
    { policy: '406' }, { policy: 'trained-new', trainedModel: true },
    { policy: 'observed-trained-new', trainedModel: true, bestObservedEpisode: true },
  ] };
  const policySelect = context.$('policy');
  policySelect.options = ['406', '407', 'trained-new', 'trained-ineligible', 'observed-trained-new'].map(value => ({ value }));
  context.mode = 'comparison';
  context.syncComparisonPolicyOptions();
  assert.equal(policySelect.value, 'trained-new', 'a valid trained policy is preferred over historical checkpoints');
  assert.equal(policySelect.options.find(option => option.value === 'trained-ineligible').hidden, true);
  assert.equal(policySelect.options.find(option => option.value === '407').hidden, true);
  context.comparison.selectedPolicy = '406';
  context.syncComparisonPolicyOptions();
  assert.equal(policySelect.value, '406', 'an explicit eligible comparison choice is retained');
  context.mode = 'recorded';
  context.syncComparisonPolicyOptions();
  assert.equal(policySelect.options.find(option => option.value === '407').hidden, false, 'historical recorded policies remain available');
  context.mode = 'comparison';
  context.comparison.episodes = [];
  context.syncComparisonPolicyOptions();
  assert.equal(policySelect.value, '', 'no eligible comparison must not fall back to an old omni layout');
  context.document = { createElement: () => new Element() };
  vm.runInContext(source.slice(source.indexOf('function fillEpisodeOptions('), source.indexOf('function isSavedLayout(')), context);
  vm.runInContext(source.slice(source.indexOf('function syncModelCatalog('), source.indexOf('function comparisonControls(')), context);
  context.mode = 'training';
  context.comparison.layouts = [];
  context.syncModelCatalog({ comparisonEpisodes: [], comparisonLayouts: [] });
  const layoutSelect = context.$('comparison-layout');
  assert.equal(layoutSelect.options.length, 0);
  const directionalLayout = { id: 'directional_balanced_8', label: 'Directional workbench' };
  const matchedLayout = { id: 'matched_common_sense', label: 'Matched directional comparison' };
  const newEpisode = { id: 'trained-new', policy: 'trained-new', case: 1, label: 'New trained policy', trainedModel: true,
    defaultLayout: directionalLayout.id, availableLayouts: [directionalLayout.id] };
  const newSession = { comparisonEpisodes: [newEpisode], comparisonLayouts: [directionalLayout],
    trainedModels: [{ id: 'trained-new', label: 'New trained policy', bestEpisode: 8 }] };
  context.syncModelCatalog(newSession);
  assert.equal(layoutSelect.options.length, 1, 'completion of the first eligible model refreshes an empty layout selector');
  assert.equal(layoutSelect.value, directionalLayout.id);
  assert.equal(context.comparison.layouts[0].id, directionalLayout.id);
  context.syncModelCatalog({ ...newSession, comparisonLayouts: [matchedLayout, directionalLayout] });
  assert.equal(layoutSelect.value, directionalLayout.id, 'refresh preserves an eligible existing layout selection');
  layoutSelect.value = matchedLayout.id;
  context.syncModelCatalog(newSession);
  assert.equal(layoutSelect.value, directionalLayout.id, 'a removed stale layout is replaced after a completed run');
  context.mode = 'comparison';
  context.comparison.selectedPolicy = 'trained-new';
  context.$('case').value = 'trained-new';
  layoutSelect.value = matchedLayout.id;
  context.syncModelCatalog({ ...newSession, comparisonLayouts: [matchedLayout, directionalLayout] });
  assert.equal(layoutSelect.value, directionalLayout.id, 'a globally available layout must still be eligible for the current model');
  assert.equal(layoutSelect.options.find(option => option.value === matchedLayout.id).hidden, true);
  context.syncModelCatalog({ comparisonEpisodes: [], comparisonLayouts: [] });
  assert.equal(layoutSelect.options.length, 0, 'removed comparison layouts do not linger in the selector');
  process.stdout.write('Training evidence and replay behavior checks passed.\n');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
