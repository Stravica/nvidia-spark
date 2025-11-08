#!/usr/bin/env node

/**
 * Performance Testing CLI Tool for Ollama vs vLLM
 *
 * Compares inference performance between Ollama and vLLM for the same model.
 * - Ollama: Native API (/api/chat) with keep_alive: -1
 * - vLLM: OpenAI-compatible API (/v1/chat/completions)
 *
 * Usage: ./perftest.js <model-name> [options]
 * Example: ./perftest.js qwen3-32b-fp8
 */

import { spawn, execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  OLLAMA_PORT: 11434,
  VLLM_PORT: 8000,
  HEALTH_CHECK_TIMEOUT: 300000, // 5 minutes for first load
  HEALTH_CHECK_INTERVAL: 2000,  // 2 seconds between checks
  WARMUP_TOKENS: 50,
  TEST_TOKENS: 500,
  ITERATIONS: 3,
  DOCKER_COMPOSE_PATH: '/opt/inference/docker-compose.yml',
};

const TESTS = [
  {
    name: 'Creative Writing (Non-Reasoning)',
    prompt: 'Write a detailed technical blog post about quantum computing. Include an introduction, 3 main sections covering quantum superposition, quantum entanglement, and quantum gates, and a conclusion. Make it informative and engaging.',
    type: 'creative',
  },
  {
    name: 'Analytical Reasoning (Code Analysis)',
    prompt: 'Analyze the time and space complexity of the quicksort algorithm. Explain the best case, average case, and worst case scenarios with detailed examples. Include the recurrence relations and explain why the pivot selection matters.',
    type: 'reasoning',
  },
];

// ============================================================================
// Service Discovery
// ============================================================================

function parseDockerCompose(modelName) {
  try {
    const content = readFileSync(CONFIG.DOCKER_COMPOSE_PATH, 'utf8');
    const services = { ollama: null, vllm: null };

    // Look for service names matching pattern: {provider}-{model}
    const ollamaPattern = new RegExp(`^\\s+(ollama-${modelName}):\\s*$`, 'm');
    const vllmPattern = new RegExp(`^\\s+(vllm-${modelName}):\\s*$`, 'm');

    const ollamaMatch = content.match(ollamaPattern);
    const vllmMatch = content.match(vllmPattern);

    if (ollamaMatch) services.ollama = ollamaMatch[1];
    if (vllmMatch) services.vllm = vllmMatch[1];

    return services;
  } catch (error) {
    throw new Error(`Failed to read docker-compose.yml: ${error.message}`);
  }
}

function validateServices(services, skipOllama, skipVllm) {
  const errors = [];

  if (!skipOllama && !services.ollama) {
    errors.push('Ollama service not found');
  }
  if (!skipVllm && !services.vllm) {
    errors.push('vLLM service not found');
  }

  if (errors.length > 0) {
    throw new Error(`Service validation failed:\n  - ${errors.join('\n  - ')}`);
  }
}

// ============================================================================
// Container Management
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

async function waitForHealth(provider, port, timeout = CONFIG.HEALTH_CHECK_TIMEOUT) {
  const startTime = Date.now();
  const endpoint = provider === 'ollama' ? '/v1/models' : '/health';

  console.log(`⏳ Waiting for ${provider} to be ready (timeout: ${timeout/1000}s)...`);

  while (Date.now() - startTime < timeout) {
    try {
      const response = await fetch(`http://localhost:${port}${endpoint}`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`   ✓ ${provider} ready (${elapsed}s)\n`);
        return true;
      }
    } catch (error) {
      // Service not ready yet, continue waiting
    }

    await new Promise(resolve => setTimeout(resolve, CONFIG.HEALTH_CHECK_INTERVAL));
  }

  throw new Error(`${provider} health check timeout after ${timeout/1000}s`);
}

// ============================================================================
// API Client (Native APIs for both providers)
// ============================================================================

async function runOllamaChatCompletion(port, modelId, prompt, maxTokens, temperature = 0.7) {
  const url = `http://localhost:${port}/api/chat`;

  const body = {
    model: modelId,
    messages: [
      { role: 'user', content: prompt }
    ],
    options: {
      num_predict: maxTokens,
      temperature: temperature,
    },
    stream: true,
    keep_alive: -1,  // Keep model loaded in memory
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
        if (!line.trim()) continue;

        try {
          const data = JSON.parse(line);

          // Record TTFT on first chunk with actual content or thinking tokens
          // (Qwen3 models use "thinking" field for CoT reasoning)
          if (firstTokenTime === null && (data.message?.content || data.message?.thinking)) {
            firstTokenTime = performance.now();
            metrics.ttft = firstTokenTime - startTime;
          }

          // Accumulate content (not thinking tokens)
          if (data.message?.content) {
            metrics.content += data.message.content;
            metrics.chunks++;
          }

          // Capture usage stats from final chunk
          if (data.done && data.prompt_eval_count !== undefined) {
            metrics.tokensIn = data.prompt_eval_count || 0;
            metrics.tokensOut = data.eval_count || 0;
          }
        } catch (parseError) {
          // Skip malformed JSON lines
        }
      }
    }

    const endTime = performance.now();
    metrics.totalTime = endTime - startTime;

    // Estimate tokens if not provided
    if (metrics.tokensOut === 0 && metrics.content.length > 0) {
      metrics.tokensOut = Math.round(metrics.content.length / 4);
    }

    // Calculate TPS
    if (metrics.ttft && metrics.tokensOut > 0) {
      const generationTime = (metrics.totalTime - metrics.ttft) / 1000;
      metrics.tps = metrics.tokensOut / generationTime;
    }

  } catch (error) {
    metrics.error = error.message;
  }

  return metrics;
}

async function runVllmChatCompletion(port, modelId, prompt, maxTokens, temperature = 0.7) {
  const url = `http://localhost:${port}/v1/chat/completions`;

  const body = {
    model: modelId,
    messages: [
      { role: 'user', content: prompt }
    ],
    max_tokens: maxTokens,
    temperature: temperature,
    stream: true,
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

async function runChatCompletion(provider, port, modelId, prompt, maxTokens, temperature = 0.7) {
  if (provider === 'ollama') {
    return runOllamaChatCompletion(port, modelId, prompt, maxTokens, temperature);
  } else {
    return runVllmChatCompletion(port, modelId, prompt, maxTokens, temperature);
  }
}

async function runWarmup(provider, port, modelId) {
  console.log(`   🔥 Pre-warming ${provider}...`);

  const warmupPrompt = 'Say hello in exactly 5 words.';
  const result = await runChatCompletion(provider, port, modelId, warmupPrompt, CONFIG.WARMUP_TOKENS, 0.1);

  if (result.error) {
    throw new Error(`Warmup failed: ${result.error}`);
  }

  const tpsDisplay = result.tps ? `, ${result.tps.toFixed(1)} tok/s` : '';
  console.log(`      ✓ Warmup complete (${result.tokensOut} tokens, ${result.totalTime.toFixed(0)}ms${tpsDisplay})`);
}

async function runTestSuite(provider, port, modelId, test, iteration) {
  const label = `${provider.toUpperCase()} - ${test.type} - Run ${iteration + 1}`;
  process.stdout.write(`   ⚙️  ${label}... `);

  const result = await runChatCompletion(provider, port, modelId, test.prompt, CONFIG.TEST_TOKENS);

  if (result.error) {
    console.log(`❌ FAILED`);
    console.error(`      Error: ${result.error}`);
    return null;
  }

  const tpsDisplay = result.tps ? `${result.tps.toFixed(1)} tok/s` : 'N/A';
  console.log(`✓ (${result.tokensOut} tokens, ${tpsDisplay})`);
  return result;
}

async function runAllTests(provider, serviceName, port, modelId) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📊 Testing ${provider.toUpperCase()}: ${serviceName}`);
  console.log(`${'='.repeat(70)}`);

  startService(serviceName);
  await waitForHealth(provider, port);

  // Warmup (Ollama native API uses keep_alive: -1 to keep model loaded)
  await runWarmup(provider, port, modelId);

  const results = {};

  // Run both tests
  for (const test of TESTS) {
    console.log(`\n   📝 Test: ${test.name}`);
    results[test.type] = [];

    for (let i = 0; i < CONFIG.ITERATIONS; i++) {
      const result = await runTestSuite(provider, port, modelId, test, i);
      if (result) {
        results[test.type].push(result);
      }

      // Brief pause between iterations
      if (i < CONFIG.ITERATIONS - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }

  stopService(serviceName);

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
  const aggregated = {};

  for (const [testType, runs] of Object.entries(results)) {
    const ttfts = runs.map(r => r.ttft).filter(v => v !== null);
    const tpss = runs.map(r => r.tps).filter(v => v !== null);
    const tokensIns = runs.map(r => r.tokensIn).filter(v => v > 0);
    const tokensOuts = runs.map(r => r.tokensOut).filter(v => v > 0);
    const totalTimes = runs.map(r => r.totalTime / 1000).filter(v => v > 0); // Convert to seconds

    aggregated[testType] = {
      ttft: calculateStats(ttfts),
      tps: calculateStats(tpss),
      tokensIn: tokensIns.length > 0 ? Math.round(tokensIns.reduce((a, b) => a + b) / tokensIns.length) : 0,
      tokensOut: tokensOuts.length > 0 ? Math.round(tokensOuts.reduce((a, b) => a + b) / tokensOuts.length) : 0,
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

function determineWinner(ollamaValue, vllmValue, lowerIsBetter = false) {
  const ollama = typeof ollamaValue === 'object' ? ollamaValue.mean : ollamaValue;
  const vllm = typeof vllmValue === 'object' ? vllmValue.mean : vllmValue;

  if (ollama === 0 || vllm === 0) return '-';

  if (lowerIsBetter) {
    return ollama < vllm ? '🏆 Ollama' : '🏆 vLLM';
  } else {
    return ollama > vllm ? '🏆 Ollama' : '🏆 vLLM';
  }
}

function printResults(ollamaResults, vllmResults, modelName) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📈 PERFORMANCE COMPARISON RESULTS`);
  console.log(`${'='.repeat(70)}`);
  console.log(`Model: ${modelName} | Iterations: ${CONFIG.ITERATIONS} per test | Max Tokens: ${CONFIG.TEST_TOKENS}`);
  console.log();

  for (const test of TESTS) {
    const testType = test.type;
    const ollama = ollamaResults[testType];
    const vllm = vllmResults[testType];

    if (!ollama || !vllm) {
      console.log(`⚠️  ${test.name}: Insufficient data\n`);
      continue;
    }

    console.log(`\n📝 ${test.name.toUpperCase()}`);
    console.log('─'.repeat(70));
    console.log(`┌─────────────────┬────────────────┬────────────────┬─────────────┐`);
    console.log(`│ Metric          │ Ollama         │ vLLM           │ Winner      │`);
    console.log(`├─────────────────┼────────────────┼────────────────┼─────────────┤`);

    // TTFT (lower is better)
    console.log(`│ TTFT (ms)       │ ${formatValue(ollama.ttft, 0).padEnd(14)} │ ${formatValue(vllm.ttft, 0).padEnd(14)} │ ${determineWinner(ollama.ttft, vllm.ttft, true).padEnd(11)} │`);

    // TPS (higher is better)
    console.log(`│ TPS (tok/s)     │ ${formatValue(ollama.tps, 2).padEnd(14)} │ ${formatValue(vllm.tps, 2).padEnd(14)} │ ${determineWinner(ollama.tps, vllm.tps, false).padEnd(11)} │`);

    // Tokens In (no winner)
    console.log(`│ Tokens In       │ ${String(ollama.tokensIn).padEnd(14)} │ ${String(vllm.tokensIn).padEnd(14)} │ ${'─'.padEnd(11)} │`);

    // Tokens Out (no winner)
    console.log(`│ Tokens Out      │ ${String(ollama.tokensOut).padEnd(14)} │ ${String(vllm.tokensOut).padEnd(14)} │ ${'─'.padEnd(11)} │`);

    // Total Time (lower is better)
    console.log(`│ Total Time (s)  │ ${formatValue(ollama.totalTime, 1).padEnd(14)} │ ${formatValue(vllm.totalTime, 1).padEnd(14)} │ ${determineWinner(ollama.totalTime, vllm.totalTime, true).padEnd(11)} │`);

    console.log(`└─────────────────┴────────────────┴────────────────┴─────────────┘`);

    // Calculate percentage differences
    const ttftDiff = ((ollama.ttft.mean - vllm.ttft.mean) / ollama.ttft.mean * 100).toFixed(1);
    const tpsDiff = ((vllm.tps.mean - ollama.tps.mean) / ollama.tps.mean * 100).toFixed(1);

    console.log();
    console.log(`📊 Summary for ${test.name}:`);
    if (parseFloat(ttftDiff) > 0) {
      console.log(`   • vLLM has ${Math.abs(parseFloat(ttftDiff))}% faster TTFT`);
    } else {
      console.log(`   • Ollama has ${Math.abs(parseFloat(ttftDiff))}% faster TTFT`);
    }

    if (parseFloat(tpsDiff) > 0) {
      console.log(`   • vLLM has ${Math.abs(parseFloat(tpsDiff))}% higher TPS`);
    } else {
      console.log(`   • Ollama has ${Math.abs(parseFloat(tpsDiff))}% higher TPS`);
    }
  }

  console.log(`\n${'='.repeat(70)}\n`);
}

// ============================================================================
// CLI Interface
// ============================================================================

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    modelName: null,
    mode: 'both', // default: test both providers
    skipOllama: false,
    skipVllm: false,
    iterations: CONFIG.ITERATIONS,
    warmupTokens: CONFIG.WARMUP_TOKENS,
    testTokens: CONFIG.TEST_TOKENS,
  };

  let positionalArgs = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg.startsWith('--')) {
      const flag = arg.slice(2);

      switch (flag) {
        case 'skip-ollama':
          config.skipOllama = true;
          break;
        case 'skip-vllm':
          config.skipVllm = true;
          break;
        case 'iterations':
          config.iterations = parseInt(args[++i], 10);
          break;
        case 'warmup-tokens':
          config.warmupTokens = parseInt(args[++i], 10);
          break;
        case 'test-tokens':
          config.testTokens = parseInt(args[++i], 10);
          break;
        case 'help':
          printHelp();
          process.exit(0);
        default:
          console.error(`Unknown option: --${flag}`);
          process.exit(1);
      }
    } else {
      positionalArgs.push(arg);
    }
  }

  // Parse positional arguments: <model-name> [mode]
  if (positionalArgs.length > 0) {
    config.modelName = positionalArgs[0];
  }

  if (positionalArgs.length > 1) {
    const mode = positionalArgs[1].toLowerCase();
    if (!['ollama', 'vllm', 'both'].includes(mode)) {
      console.error(`Error: Invalid mode '${positionalArgs[1]}'. Valid modes: ollama, vllm, both`);
      process.exit(1);
    }
    config.mode = mode;
  }

  // Map mode to skip flags (skip flags take precedence if explicitly set)
  if (!args.includes('--skip-ollama') && !args.includes('--skip-vllm')) {
    if (config.mode === 'ollama') {
      config.skipVllm = true;
    } else if (config.mode === 'vllm') {
      config.skipOllama = true;
    }
  }

  return config;
}

function printHelp() {
  console.log(`
Performance Testing CLI Tool for Ollama vs vLLM

Usage: ./perftest.js <model-name> [mode] [options]

Arguments:
  <model-name>         Model name (e.g., qwen3-32b-fp8)
  [mode]               Test mode: ollama, vllm, or both (default: both)

Options:
  --iterations <n>     Number of test runs per test (default: ${CONFIG.ITERATIONS})
  --warmup-tokens <n>  Warmup max_tokens (default: ${CONFIG.WARMUP_TOKENS})
  --test-tokens <n>    Test max_tokens (default: ${CONFIG.TEST_TOKENS})
  --skip-ollama        Skip Ollama testing (overrides mode)
  --skip-vllm          Skip vLLM testing (overrides mode)
  --help               Show this help message

Examples:
  ./perftest.js qwen3-32b-fp8              # Test both providers
  ./perftest.js qwen3-32b-fp8 ollama       # Test Ollama only
  ./perftest.js qwen3-32b-fp8 vllm         # Test vLLM only
  ./perftest.js qwen3-32b-fp8 both         # Test both providers (explicit)
  ./perftest.js qwen3-32b-fp8 --iterations 5
  ./perftest.js qwen3-32b-fp8 ollama --test-tokens 1000

Tests:
  1. Creative Writing (Non-Reasoning) - Sustained generation
  2. Analytical Reasoning (Code Analysis) - Structured reasoning

Metrics:
  • TTFT - Time to First Token (latency)
  • TPS - Tokens Per Second (throughput)
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

    if (!cliConfig.modelName) {
      console.error('Error: Model name is required\n');
      printHelp();
      process.exit(1);
    }

    // Update global config
    CONFIG.ITERATIONS = cliConfig.iterations;
    CONFIG.WARMUP_TOKENS = cliConfig.warmupTokens;
    CONFIG.TEST_TOKENS = cliConfig.testTokens;

    console.log(`\n🔍 Performance Test: ${cliConfig.modelName}`);
    console.log(`${'='.repeat(70)}`);

    // Discover services
    console.log('\n📋 Discovering services...');
    const services = parseDockerCompose(cliConfig.modelName);

    console.log(`   Ollama: ${services.ollama || '❌ Not found'}`);
    console.log(`   vLLM:   ${services.vllm || '❌ Not found'}`);

    validateServices(services, cliConfig.skipOllama, cliConfig.skipVllm);

    // Stop all containers
    stopAllContainers();

    let ollamaResults = null;
    let vllmResults = null;

    // Test Ollama
    if (!cliConfig.skipOllama && services.ollama) {
      ollamaResults = await runAllTests(
        'ollama',
        services.ollama,
        CONFIG.OLLAMA_PORT,
        cliConfig.modelName
      );
      ollamaResults = aggregateResults(ollamaResults);
    }

    // Test vLLM
    if (!cliConfig.skipVllm && services.vllm) {
      vllmResults = await runAllTests(
        'vllm',
        services.vllm,
        CONFIG.VLLM_PORT,
        `Qwen/Qwen3-32B-FP8` // vLLM uses full HF model ID
      );
      vllmResults = aggregateResults(vllmResults);
    }

    // Display results
    if (ollamaResults && vllmResults) {
      printResults(ollamaResults, vllmResults, cliConfig.modelName);
    } else {
      console.log('\n⚠️  Insufficient test data for comparison');
    }

    console.log('✅ Performance testing complete!\n');

  } catch (error) {
    console.error(`\n❌ Error: ${error.message}\n`);
    process.exit(1);
  }
})();
