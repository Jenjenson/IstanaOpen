const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../console/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../console/index.html'), 'utf8');
const nodes = new Map();
const context = {
  $: id => { if (!nodes.has(id)) nodes.set(id, { textContent: '', hidden: false }); return nodes.get(id); },
  fmt: (value, places = 1) => Number.isFinite(value) ? value.toFixed(places) : '—',
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function signed('), source.indexOf('function comparisonResults(')), context);
vm.runInContext(source.slice(source.indexOf('function comparisonClearResults('), source.indexOf('function comparisonCost(')), context);
const zero = { source: 'testEvaluation', caseCount: 64, pairedCaseCount: 64,
  meanWarningSeconds: 27, baselineMeanWarningSeconds: 27, deltaSeconds: 0,
  interval95: { lowerSeconds: 0, upperSeconds: 0, standardErrorSeconds: 0, method: 'normal_approximation' } };
const result = { trainedModel: true, heldOutSummary: zero, metrics: { rl: { mean_warning_s: 99 } } };
context.renderComparisonTestSummary(result);
assert.equal(context.$('comparison-held-out').hidden, false);
assert.equal(context.$('comparison-test-count').textContent, '64');
assert.equal(context.$('comparison-test-rl').textContent, '27.00 s');
assert.equal(context.$('comparison-test-gain').textContent, '0.00 s · no measured gain');
assert.match(context.$('comparison-test-uncertainty').textContent, /Approximate 95% interval.*0.00 s to 0.00 s.*paired mean ± 1.96 standard errors/);
assert.doesNotMatch(context.$('comparison-test-uncertainty').textContent, /200|confirmation/);
context.renderComparisonTestSummary({ ...result, heldOutSummary: { ...zero, meanWarningSeconds: 26, deltaSeconds: -1,
  interval95: { ...zero.interval95, lowerSeconds: -2, upperSeconds: 0 } } });
assert.equal(context.$('comparison-test-gain').textContent, '-1.00 s');
for (const hidden of [null, {}, { trainedModel: true }, { ...result, bestObservedEpisode: true },
  { ...result, trainedModel: false }, { ...result, layoutOnly: true },
  { ...result, heldOutSummary: { ...zero, caseCount: 0 } },
  { ...result, heldOutSummary: { ...zero, meanWarningSeconds: NaN } }]) {
  context.renderComparisonTestSummary(result);
  context.renderComparisonTestSummary(hidden);
  assert.equal(context.$('comparison-held-out').hidden, true, 'switching source must remove stale aggregate');
  assert.equal(context.$('comparison-test-rl').textContent, '—');
  assert.equal(context.$('comparison-test-uncertainty').textContent, '');
}
for (const summary of [{ ...zero, interval95: null, pairedCaseCount: 0 },
  { ...zero, pairedCaseCount: 2 }, { ...zero, caseCount: 1, pairedCaseCount: 1 }]) {
  context.renderComparisonTestSummary({ ...result, heldOutSummary: summary });
  assert.equal(context.$('comparison-held-out').hidden, false);
  assert.match(context.$('comparison-test-uncertainty').textContent, /Uncertainty unavailable/);
}
context.renderComparisonTestSummary(result);
context.comparisonClearResults();
assert.equal(context.$('comparison-held-out').hidden, true);
assert.ok(html.indexOf('id="comparison-held-out"') < html.indexOf('class="comparison-map-grid"'));
assert.match(html, /single-episode replay below is illustrative/);
assert.match(html, /Recorded internal test of the validation-selected policy/);
process.stdout.write('Comparison held-out summary behavior checks passed.\n');
