#!/usr/bin/env node

/**
 * Model Performance Testing CLI Tool
 *
 * Compares inference performance across multiple vLLM models:
 * - Qwen3-32B-FP8 (existing baseline)
 * - Qwen3-30B-A3B-FP8 (MoE model)
 * - Llama 3.3 70B-FP8 (large dense model)
 *
 * Tests:
 * 1. Single-request latency (fixed 500 tokens output)
 * 2. Long-context handling (varying input: 1K, 8K, 16K, 32K tokens)
 *
 * Usage: ./perftest-models.js [options]
 * Example: ./perftest-models.js --iterations 3 --skip qwen3-32b-fp8
 */

import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  VLLM_PORT: 8000,
  HEALTH_CHECK_TIMEOUT: 1800000, // 30 minutes for first load (large models with downloads)
  HEALTH_CHECK_INTERVAL: 3000,  // 3 seconds between checks
  WARMUP_TOKENS: 50,
  LATENCY_TEST_TOKENS: 500,
  ITERATIONS: 3,
  DOCKER_COMPOSE_PATH: '/opt/inference/docker-compose.yml',
};

const MODELS = [
  {
    name: 'qwen3-32b-fp8',
    serviceName: 'vllm-qwen3-32b-fp8',
    modelId: 'Qwen/Qwen3-32B-FP8',
    description: 'Dense 32B (Baseline)',
    maxContext: 32000,
  },
  {
    name: 'qwen3-30b-a3b-fp8',
    serviceName: 'vllm-qwen3-30b-a3b-fp8',
    modelId: 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
    description: 'MoE 30B (3B active)',
    maxContext: 32768,
  },
  {
    name: 'llama33-70b-fp8',
    serviceName: 'vllm-llama33-70b-fp8',
    modelId: 'nvidia/Llama-3.3-70B-Instruct-FP8',
    description: 'Dense 70B (Long-context)',
    maxContext: 65536,
  },
];

// Latency test: Fixed output, minimal input
const LATENCY_TEST = {
  name: 'Single-Request Latency',
  prompt: 'Write a detailed technical explanation of how neural network backpropagation works. Include mathematical formulas, examples, and practical considerations for implementation.',
  maxTokens: CONFIG.LATENCY_TEST_TOKENS,
  type: 'latency',
};

// Long-context tests: Varying input size
const CONTEXT_TESTS = [
  {
    name: 'Short Context (1K tokens)',
    contextSize: 1000,
    type: 'context-1k',
  },
  {
    name: 'Medium Context (8K tokens)',
    contextSize: 8000,
    type: 'context-8k',
  },
  {
    name: 'Long Context (16K tokens)',
    contextSize: 16000,
    type: 'context-16k',
  },
  {
    name: 'Very Long Context (32K tokens)',
    contextSize: 32000,
    type: 'context-32k',
  },
];

// ============================================================================
// Helper Functions
// ============================================================================

function execCommand(command, description) {
  try {
    const output = execSync(command, {
      encoding: 'utf8',
      cwd: '/opt/inference',
      stdio: ['pipe', 'pipe', 'pipe']
    });
    return output;
  } catch (error) {
    console.error(`[ERROR] ${description} failed:`, error.message);
    throw error;
  }
}

function generateContextPrompt(tokenCount) {
  // Generate approximately tokenCount tokens of text (rough estimate: 4 chars per token)
  const charsNeeded = tokenCount * 4;
  const paragraph = "In computer science and software engineering, artificial intelligence and machine learning have revolutionized how we approach complex problems. Neural networks, deep learning architectures, and transformer models have enabled unprecedented capabilities in natural language processing, computer vision, and reinforcement learning. These technologies continue to evolve rapidly, with new architectures and training techniques emerging regularly. ";

  const repetitions = Math.ceil(charsNeeded / paragraph.length);
  const context = paragraph.repeat(repetitions).substring(0, charsNeeded);

  return `Given the following context:\n\n${context}\n\nBased on the above context, provide a brief summary of the key points in exactly 100 words.`;
}

// ============================================================================
// Service Discovery
// ============================================================================

function parseDockerCompose() {
  try {
    const content = readFileSync(CONFIG.DOCKER_COMPOSE_PATH, 'utf8');
    const services = {};

    for (const model of MODELS) {
      const pattern = new RegExp(`^\\s+(${model.serviceName}):\\s*$`, 'm');
      const match = content.match(pattern);
      if (match) {
        services[model.name] = model.serviceName;
      }
    }

    return services;
  } catch (error) {
    throw new Error(`Failed to read docker-compose.yml: ${error.message}`);
  }
}

function validateServices(services, skipModels) {
  const errors = [];

  for (const model of MODELS) {
    if (!skipModels.includes(model.name) && !services[model.name]) {
      errors.push(`Service ${model.serviceName} not found`);
    }
  }

  if (errors.length > 0) {
    throw new Error(`Service validation failed:\n  - ${errors.join('\n  - ')}`);
  }
}

// ============================================================================
// Container Management
// ============================================================================

function stopAllContainers() {
  console.log('🛑 Stopping all inference containers...');
  try {
    execCommand('docker compose stop', 'Stop containers');
    console.log('   ✓ All containers stopped\n');
  } catch (error) {
    console.warn('   ⚠ Warning: Failed to stop containers gracefully');
  }
}

function startService(serviceName) {
  console.log(`🚀 Starting ${serviceName}...`);
  execCommand(`docker compose up -d ${serviceName}`, `Start ${serviceName}`);
  console.log(`   ✓ ${serviceName} started`);
}

function stopService(serviceName) {
  console.log(`\n🛑 Stopping ${serviceName}...`);
  execCommand(`docker compose stop ${serviceName}`, `Stop ${serviceName}`);
  console.log(`   ✓ ${serviceName} stopped`);
}

async function waitForHealth(modelName, timeout = CONFIG.HEALTH_CHECK_TIMEOUT) {
  const startTime = Date.now();
  const endpoint = '/health';

  console.log(`⏳ Waiting for ${modelName} to be ready (timeout: ${timeout/1000}s)...`);

  while (Date.now() - startTime < timeout) {
    try {
      const response = await fetch(`http://localhost:${CONFIG.VLLM_PORT}${endpoint}`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`   ✓ ${modelName} ready (${elapsed}s)\n`);
        return true;
      }
    } catch (error) {
      // Service not ready yet, continue waiting
    }

    await new Promise(resolve => setTimeout(resolve, CONFIG.HEALTH_CHECK_INTERVAL));
  }

  throw new Error(`${modelName} health check timeout after ${timeout/1000}s`);
}

// ============================================================================
// API Client
// ============================================================================

async function runVllmChatCompletion(modelId, prompt, maxTokens, temperature = 0.7) {
  const url = `http://localhost:${CONFIG.VLLM_PORT}/v1/chat/completions`;

  const body = {
    model: modelId,
    messages: [
      { role: 'user', content: prompt }
    ],
    max_tokens: maxTokens,
    temperature: temperature,
    stream: true,
    stream_options: { include_usage: true },
  };

  const metrics = {
    ttft: null,
    totalTime: null,
    tokensIn: 0,
    tokensOut: 0,
    tps: null,
    chunks: 0,
    content: '',
    error: null,
  };

  const startTime = performance.now();
  let firstTokenTime = null;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(600000),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim() || line.trim() === 'data: [DONE]') continue;
        if (!line.startsWith('data: ')) continue;

        try {
          const data = JSON.parse(line.slice(6));

          // Record TTFT on first content chunk
          if (firstTokenTime === null && data.choices?.[0]?.delta?.content) {
            firstTokenTime = performance.now();
            metrics.ttft = firstTokenTime - startTime;
          }

          // Accumulate content
          if (data.choices?.[0]?.delta?.content) {
            metrics.content += data.choices[0].delta.content;
            metrics.chunks++;
          }

          // Capture usage stats from final chunk (if available)
          if (data.usage) {
            metrics.tokensIn = data.usage.prompt_tokens || 0;
            metrics.tokensOut = data.usage.completion_tokens || 0;
          }
        } catch (parseError) {
          // Skip malformed JSON lines
        }
      }
    }

    const endTime = performance.now();
    metrics.totalTime = endTime - startTime;

    // If no usage stats from API, estimate from content
    if (metrics.tokensOut === 0) {
      // Rough estimate: ~4 chars per token
      metrics.tokensOut = Math.round(metrics.content.length / 4);
    }

    // Calculate TPS: exclude TTFT from generation time
    if (metrics.ttft && metrics.tokensOut > 0) {
      const generationTime = (metrics.totalTime - metrics.ttft) / 1000; // Convert to seconds
      metrics.tps = metrics.tokensOut / generationTime;
    }

  } catch (error) {
    metrics.error = error.message;
  }

  return metrics;
}

// ============================================================================
// Test Execution
// ============================================================================

async function runWarmup(modelName, modelId) {
  console.log(`   🔥 Pre-warming ${modelName}...`);

  const warmupPrompt = 'Say hello in exactly 5 words.';
  const result = await runVllmChatCompletion(modelId, warmupPrompt, CONFIG.WARMUP_TOKENS, 0.1);

  if (result.error) {
    throw new Error(`Warmup failed: ${result.error}`);
  }

  const tpsDisplay = result.tps ? `, ${result.tps.toFixed(1)} tok/s` : '';
  console.log(`      ✓ Warmup complete (${result.tokensOut} tokens, ${result.totalTime.toFixed(0)}ms${tpsDisplay})`);
}

async function runLatencyTest(modelName, modelId, iteration) {
  const label = `${modelName} - Latency - Run ${iteration + 1}`;
  process.stdout.write(`   ⚙️  ${label}... `);

  const result = await runVllmChatCompletion(
    modelId,
    LATENCY_TEST.prompt,
    LATENCY_TEST.maxTokens,
    0.7
  );

  if (result.error) {
    console.log(`❌ FAILED`);
    console.error(`      Error: ${result.error}`);
    return null;
  }

  const tpsDisplay = result.tps ? `${result.tps.toFixed(2)} tok/s` : 'N/A';
  const ttftDisplay = result.ttft ? `${result.ttft.toFixed(0)}ms` : 'N/A';
  console.log(`✓ (TTFT: ${ttftDisplay}, ${result.tokensOut} tokens, ${tpsDisplay})`);
  return result;
}

async function runContextTest(modelName, modelId, contextTest, iteration) {
  const label = `${modelName} - ${contextTest.name} - Run ${iteration + 1}`;
  process.stdout.write(`   ⚙️  ${label}... `);

  const prompt = generateContextPrompt(contextTest.contextSize);
  const result = await runVllmChatCompletion(modelId, prompt, 100, 0.7); // Short output for context tests

  if (result.error) {
    console.log(`❌ FAILED`);
    console.error(`      Error: ${result.error}`);
    return null;
  }

  const ttftDisplay = result.ttft ? `${result.ttft.toFixed(0)}ms` : 'N/A';
  const totalDisplay = result.totalTime ? `${(result.totalTime / 1000).toFixed(1)}s` : 'N/A';
  console.log(`✓ (TTFT: ${ttftDisplay}, Total: ${totalDisplay}, ${result.tokensIn} tokens in)`);
  return result;
}

async function runAllTests(model) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📊 Testing ${model.description.toUpperCase()}: ${model.serviceName}`);
  console.log(`${'='.repeat(70)}`);

  startService(model.serviceName);
  await waitForHealth(model.name);

  // Warmup
  await runWarmup(model.name, model.modelId);

  const results = {
    latency: [],
    contexts: {}
  };

  // Run latency tests
  console.log(`\n   📝 Test: ${LATENCY_TEST.name}`);
  for (let i = 0; i < CONFIG.ITERATIONS; i++) {
    const result = await runLatencyTest(model.name, model.modelId, i);
    if (result) {
      results.latency.push(result);
    }

    // Brief pause between iterations
    if (i < CONFIG.ITERATIONS - 1) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  // Run context tests
  for (const contextTest of CONTEXT_TESTS) {
    // Skip tests that exceed model's max context
    if (contextTest.contextSize > model.maxContext) {
      console.log(`\n   ⏭️  Skipping ${contextTest.name} (exceeds ${model.maxContext} limit)`);
      continue;
    }

    console.log(`\n   📝 Test: ${contextTest.name}`);
    results.contexts[contextTest.type] = [];

    for (let i = 0; i < CONFIG.ITERATIONS; i++) {
      const result = await runContextTest(model.name, model.modelId, contextTest, i);
      if (result) {
        results.contexts[contextTest.type].push(result);
      }

      // Brief pause between iterations
      if (i < CONFIG.ITERATIONS - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }

  stopService(model.serviceName);

  return results;
}

// ============================================================================
// Statistics
// ============================================================================

function calculateStats(values) {
  if (values.length === 0) return { mean: 0, stddev: 0, min: 0, max: 0 };

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  const stddev = Math.sqrt(variance);
  const min = Math.min(...values);
  const max = Math.max(...values);

  return { mean, stddev, min, max };
}

function aggregateResults(results) {
  const aggregated = {
    latency: null,
    contexts: {}
  };

  // Aggregate latency tests
  if (results.latency.length > 0) {
    const ttfts = results.latency.map(r => r.ttft).filter(v => v !== null);
    const tpss = results.latency.map(r => r.tps).filter(v => v !== null);
    const tokensIns = results.latency.map(r => r.tokensIn).filter(v => v > 0);
    const tokensOuts = results.latency.map(r => r.tokensOut).filter(v => v > 0);
    const totalTimes = results.latency.map(r => r.totalTime / 1000).filter(v => v > 0);

    aggregated.latency = {
      ttft: calculateStats(ttfts),
      tps: calculateStats(tpss),
      tokensIn: tokensIns.length > 0 ? Math.round(tokensIns.reduce((a, b) => a + b) / tokensIns.length) : 0,
      tokensOut: tokensOuts.length > 0 ? Math.round(tokensOuts.reduce((a, b) => a + b) / tokensOuts.length) : 0,
      totalTime: calculateStats(totalTimes),
      successCount: results.latency.length,
    };
  }

  // Aggregate context tests
  for (const [contextType, runs] of Object.entries(results.contexts)) {
    if (runs.length === 0) continue;

    const ttfts = runs.map(r => r.ttft).filter(v => v !== null);
    const tokensIns = runs.map(r => r.tokensIn).filter(v => v > 0);
    const totalTimes = runs.map(r => r.totalTime / 1000).filter(v => v > 0);

    aggregated.contexts[contextType] = {
      ttft: calculateStats(ttfts),
      tokensIn: tokensIns.length > 0 ? Math.round(tokensIns.reduce((a, b) => a + b) / tokensIns.length) : 0,
      totalTime: calculateStats(totalTimes),
      successCount: runs.length,
    };
  }

  return aggregated;
}

// ============================================================================
// Results Formatting
// ============================================================================

function formatValue(value, decimals = 1, showStddev = true) {
  if (typeof value === 'object' && value.mean !== undefined) {
    const mean = value.mean.toFixed(decimals);
    const stddev = value.stddev.toFixed(decimals);
    return showStddev ? `${mean} ± ${stddev}` : mean;
  }
  return value.toFixed(decimals);
}

function determineWinner(values, lowerIsBetter = false) {
  const means = values.map(v => typeof v === 'object' ? v.mean : v);

  if (means.some(m => m === 0 || m === null)) return '-';

  const bestIndex = lowerIsBetter
    ? means.indexOf(Math.min(...means))
    : means.indexOf(Math.max(...means));

  return bestIndex;
}

function printLatencyResults(modelResults, models) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`📈 LATENCY TEST RESULTS`);
  console.log(`${'='.repeat(80)}`);
  console.log(`Test: Single-request inference | Iterations: ${CONFIG.ITERATIONS} | Max Tokens: ${CONFIG.LATENCY_TEST_TOKENS}`);
  console.log();

  const hasData = modelResults.every(r => r.latency !== null);
  if (!hasData) {
    console.log('⚠️  Insufficient data for latency comparison\n');
    return;
  }

  // Build table header
  const colWidth = 16;
  let header = '┌─────────────────┬';
  let labels = '│ Metric          │';
  let separator = '├─────────────────┼';

  for (const model of models) {
    header += `${'─'.repeat(colWidth)}┬`;
    labels += ` ${model.description.substring(0, colWidth - 2).padEnd(colWidth - 2)} │`;
    separator += `${'─'.repeat(colWidth)}┼`;
  }
  header += '─────────────┐';
  labels += ' Winner      │';
  separator += '─────────────┤';

  console.log(header);
  console.log(labels);
  console.log(separator);

  // TTFT row
  process.stdout.write('│ TTFT (ms)       │');
  const ttfts = modelResults.map(r => r.latency.ttft);
  for (const ttft of ttfts) {
    process.stdout.write(` ${formatValue(ttft, 0).padEnd(colWidth - 2)} │`);
  }
  const ttftWinner = determineWinner(ttfts, true);
  console.log(` ${ttftWinner >= 0 ? `🏆 ${models[ttftWinner].name}` : '-'.padEnd(11)} │`);

  // TPS row
  process.stdout.write('│ TPS (tok/s)     │');
  const tpss = modelResults.map(r => r.latency.tps);
  for (const tps of tpss) {
    process.stdout.write(` ${formatValue(tps, 2).padEnd(colWidth - 2)} │`);
  }
  const tpsWinner = determineWinner(tpss, false);
  console.log(` ${tpsWinner >= 0 ? `🏆 ${models[tpsWinner].name}` : '-'.padEnd(11)} │`);

  // Tokens In
  process.stdout.write('│ Tokens In       │');
  for (const result of modelResults) {
    process.stdout.write(` ${String(result.latency.tokensIn).padEnd(colWidth - 2)} │`);
  }
  console.log(' ─           │');

  // Tokens Out
  process.stdout.write('│ Tokens Out      │');
  for (const result of modelResults) {
    process.stdout.write(` ${String(result.latency.tokensOut).padEnd(colWidth - 2)} │`);
  }
  console.log(' ─           │');

  // Total Time
  process.stdout.write('│ Total Time (s)  │');
  const totalTimes = modelResults.map(r => r.latency.totalTime);
  for (const time of totalTimes) {
    process.stdout.write(` ${formatValue(time, 1).padEnd(colWidth - 2)} │`);
  }
  const timeWinner = determineWinner(totalTimes, true);
  console.log(` ${timeWinner >= 0 ? `🏆 ${models[timeWinner].name}` : '-'.padEnd(11)} │`);

  console.log('└─────────────────┴' + '─'.repeat(colWidth) + '┴'.repeat(models.length - 1) + '─'.repeat(colWidth) + '┴─────────────┘');
}

function printContextResults(modelResults, models) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`📈 LONG-CONTEXT TEST RESULTS`);
  console.log(`${'='.repeat(80)}`);
  console.log(`Test: Context length handling | Iterations: ${CONFIG.ITERATIONS} | Output: 100 tokens`);
  console.log();

  for (const contextTest of CONTEXT_TESTS) {
    const contextType = contextTest.type;

    // Check if all models have data for this context size
    const hasData = modelResults.every(r =>
      r.contexts[contextType] && r.contexts[contextType].successCount > 0
    );

    if (!hasData) {
      console.log(`\n${contextTest.name}: ⏭️  Skipped (exceeds model limits)\n`);
      continue;
    }

    console.log(`\n${contextTest.name.toUpperCase()}`);
    console.log('─'.repeat(80));

    const colWidth = 16;
    let header = '┌─────────────────┬';
    let labels = '│ Metric          │';
    let separator = '├─────────────────┼';

    for (const model of models) {
      header += `${'─'.repeat(colWidth)}┬`;
      labels += ` ${model.description.substring(0, colWidth - 2).padEnd(colWidth - 2)} │`;
      separator += `${'─'.repeat(colWidth)}┼`;
    }
    header += '─────────────┐';
    labels += ' Winner      │';
    separator += '─────────────┤';

    console.log(header);
    console.log(labels);
    console.log(separator);

    // TTFT row
    process.stdout.write('│ TTFT (ms)       │');
    const ttfts = modelResults.map(r => r.contexts[contextType]?.ttft || { mean: 0 });
    for (const ttft of ttfts) {
      process.stdout.write(` ${formatValue(ttft, 0).padEnd(colWidth - 2)} │`);
    }
    const ttftWinner = determineWinner(ttfts, true);
    console.log(` ${ttftWinner >= 0 ? `🏆 ${models[ttftWinner].name}` : '-'.padEnd(11)} │`);

    // Tokens In
    process.stdout.write('│ Tokens In       │');
    for (const result of modelResults) {
      const tokensIn = result.contexts[contextType]?.tokensIn || 0;
      process.stdout.write(` ${String(tokensIn).padEnd(colWidth - 2)} │`);
    }
    console.log(' ─           │');

    // Total Time
    process.stdout.write('│ Total Time (s)  │');
    const totalTimes = modelResults.map(r => r.contexts[contextType]?.totalTime || { mean: 0 });
    for (const time of totalTimes) {
      process.stdout.write(` ${formatValue(time, 1).padEnd(colWidth - 2)} │`);
    }
    const timeWinner = determineWinner(totalTimes, true);
    console.log(` ${timeWinner >= 0 ? `🏆 ${models[timeWinner].name}` : '-'.padEnd(11)} │`);

    console.log('└─────────────────┴' + '─'.repeat(colWidth) + '┴'.repeat(models.length - 1) + '─'.repeat(colWidth) + '┴─────────────┘');
  }
}

// ============================================================================
// CLI Interface
// ============================================================================

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    skipModels: [],
    iterations: CONFIG.ITERATIONS,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg.startsWith('--')) {
      const flag = arg.slice(2);

      switch (flag) {
        case 'skip':
          config.skipModels.push(args[++i]);
          break;
        case 'iterations':
          config.iterations = parseInt(args[++i], 10);
          break;
        case 'help':
          printHelp();
          process.exit(0);
        default:
          console.error(`Unknown option: --${flag}`);
          process.exit(1);
      }
    }
  }

  return config;
}

function printHelp() {
  console.log(`
Model Performance Testing CLI Tool

Compares inference performance across multiple vLLM models for NVIDIA DGX Spark.

Usage: ./perftest-models.js [options]

Options:
  --iterations <n>     Number of test runs per test (default: ${CONFIG.ITERATIONS})
  --skip <model-name>  Skip testing a specific model (can be used multiple times)
  --help               Show this help message

Available Models:
${MODELS.map(m => `  - ${m.name.padEnd(20)} ${m.description}`).join('\n')}

Examples:
  ./perftest-models.js                              # Test all models
  ./perftest-models.js --iterations 5               # Run 5 iterations per test
  ./perftest-models.js --skip qwen3-32b-fp8         # Skip baseline model
  ./perftest-models.js --skip llama33-70b-fp8       # Skip 70B model

Tests:
  1. Single-Request Latency - Fixed 500 token output with minimal input
  2. Long-Context Handling - Varying input sizes (1K, 8K, 16K, 32K tokens)

Metrics:
  • TTFT - Time to First Token (latency)
  • TPS - Tokens Per Second (throughput, latency test only)
  • Tokens In/Out - Input and output token counts
  • Total Time - Full request duration
`);
}

// ============================================================================
// Main
// ============================================================================

(async () => {
  try {
    // Parse CLI arguments
    const cliConfig = parseArgs();

    // Update global config
    CONFIG.ITERATIONS = cliConfig.iterations;

    console.log(`\n🔍 Model Performance Comparison`);
    console.log(`${'='.repeat(70)}`);

    // Discover services
    console.log('\n📋 Discovering services...');
    const services = parseDockerCompose();

    for (const model of MODELS) {
      const status = services[model.name] ? '✓' : '❌ Not found';
      const skip = cliConfig.skipModels.includes(model.name) ? '(skipped)' : '';
      console.log(`   ${model.name.padEnd(20)} ${status} ${skip}`);
    }

    validateServices(services, cliConfig.skipModels);

    // Stop all containers
    stopAllContainers();

    // Test each model
    const allResults = [];
    const testedModels = [];

    for (const model of MODELS) {
      if (cliConfig.skipModels.includes(model.name)) {
        continue;
      }

      const results = await runAllTests(model);
      const aggregated = aggregateResults(results);
      allResults.push(aggregated);
      testedModels.push(model);
    }

    // Display results
    if (allResults.length > 0) {
      printLatencyResults(allResults, testedModels);
      printContextResults(allResults, testedModels);
    } else {
      console.log('\n⚠️  No tests were run\n');
    }

    console.log(`\n${'='.repeat(80)}`);
    console.log('✅ Performance testing complete!\n');

  } catch (error) {
    console.error(`\n❌ Error: ${error.message}\n`);
    process.exit(1);
  }
})();
