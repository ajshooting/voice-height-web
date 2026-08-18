#!/usr/bin/env node

/** Dependency-free CDP smoke test for a separately launched local Chrome. */

import { writeFile } from 'node:fs/promises'

const targetUrl = process.argv[2]
const debuggingPort = Number(process.argv[3] ?? 9223)
const expectedHeight = Number(process.argv[4] ?? 167.15265498384767)
const toleranceCm = Number(process.argv[5] ?? 0.1)
const screenshotPath = process.argv[6]
const viewportWidth = Number(process.argv[7] ?? 1280)
const viewportHeight = Number(process.argv[8] ?? 900)
const microphoneMode = new URL(targetUrl).searchParams.has('microphoneSmoke')
const targetOrigin = new URL(targetUrl).origin

if (!targetUrl || !Number.isInteger(debuggingPort)) {
  throw new Error(
    'Usage: node tools/browser_smoke.mjs <url> [debug-port] [expected-cm] [tolerance-cm]',
  )
}

const endpoint = `http://127.0.0.1:${debuggingPort}`
const target = await waitForTarget(endpoint)
const socket = new WebSocket(target.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true })
  socket.addEventListener('error', reject, { once: true })
})

let nextId = 1
const pending = new Map()
const requests = []
const loadingFailures = []
const exceptions = []
const sessions = new Map()

socket.addEventListener('message', async (event) => {
  const message = JSON.parse(String(event.data))
  if (typeof message.id === 'number') {
    const operation = pending.get(message.id)
    if (!operation) return
    pending.delete(message.id)
    if (message.error) operation.reject(new Error(message.error.message))
    else operation.resolve(message.result)
    return
  }

  if (message.method === 'Network.requestWillBeSent') {
    const request = message.params.request
    if (/^https?:/.test(request.url)) {
      requests.push({
        method: request.method,
        url: request.url,
        hasPostData: request.hasPostData === true,
        targetType: message.sessionId
          ? (sessions.get(message.sessionId) ?? 'attached')
          : 'page',
      })
    }
  } else if (message.method === 'Network.loadingFailed') {
    loadingFailures.push({
      url: message.params.blockedReason ?? message.params.errorText,
      canceled: message.params.canceled === true,
    })
  } else if (message.method === 'Runtime.exceptionThrown') {
    exceptions.push(message.params.exceptionDetails.text)
  } else if (message.method === 'Target.attachedToTarget') {
    const sessionId = message.params.sessionId
    sessions.set(sessionId, message.params.targetInfo.type)
    try {
      await send('Network.enable', {}, sessionId)
      await send('Runtime.enable', {}, sessionId)
      await send('Runtime.runIfWaitingForDebugger', {}, sessionId)
    } catch (error) {
      exceptions.push(
        error instanceof Error ? error.message : String(error),
      )
    }
  }
})

function send(method, params = {}, sessionId) {
  const id = nextId
  nextId += 1
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    socket.send(
      JSON.stringify({
        id,
        method,
        params,
        ...(sessionId ? { sessionId } : {}),
      }),
    )
  })
}

await Promise.all([
  send('Network.enable'),
  send('Runtime.enable'),
  send('Page.enable'),
])
await send('Target.setAutoAttach', {
  autoAttach: true,
  flatten: true,
  waitForDebuggerOnStart: true,
})
await send('Page.navigate', { url: targetUrl })

const startedAt = Date.now()
let pageState
let prepareClicked = false
let startClicked = false
let recordingStartedAt
let requestsAtRecordingStart
let stopClicked = false
while (Date.now() - startedAt < 180_000) {
  const evaluation = await send('Runtime.evaluate', {
    expression: `(() => {
      const root = document.querySelector('#app')
      return {
        state: root?.dataset.state ?? null,
        heightCm:
          root?.dataset.referenceHeightCm ??
          root?.dataset.resultHeightCm ??
          document.querySelector('[data-ui="result-value"]')?.textContent ??
          null,
        provider:
          root?.dataset.referenceProvider ??
          root?.dataset.resultProvider ??
          (document.querySelector('[data-ui="detail"]')?.textContent?.includes('WebGPU')
            ? 'webgpu'
            : document.querySelector('[data-ui="detail"]')?.textContent?.includes('WASM')
              ? 'wasm'
              : null),
        recordingDurationSeconds: root?.dataset.recordingDurationSeconds ?? null,
        status: document.querySelector('[data-ui="status"]')?.textContent ?? null,
        detail: document.querySelector('[data-ui="detail"]')?.textContent ?? null,
        error: document.querySelector('[data-ui="error"]:not([hidden])')?.textContent ?? null,
      }
    })()`,
    returnByValue: true,
  })
  pageState = evaluation.result?.value
  if (microphoneMode) {
    if (pageState?.state === 'idle' && !prepareClicked) {
      prepareClicked = true
      await clickAction('prepare')
    } else if (pageState?.state === 'ready' && !startClicked) {
      startClicked = true
      await clickAction('start')
    } else if (
      pageState?.state === 'recording' &&
      recordingStartedAt === undefined
    ) {
      recordingStartedAt = Date.now()
      requestsAtRecordingStart = requests.length
    } else if (
      pageState?.state === 'recording' &&
      recordingStartedAt !== undefined &&
      Date.now() - recordingStartedAt >= 5_500 &&
      !stopClicked
    ) {
      stopClicked = true
      await clickAction('stop')
    }
  }
  if (pageState?.state === 'result' || pageState?.state === 'error') break
  await new Promise((resolve) => setTimeout(resolve, 500))
}

if (screenshotPath) {
  await send('Emulation.setDeviceMetricsOverride', {
    width: viewportWidth,
    height: viewportHeight,
    deviceScaleFactor: 1,
    mobile: viewportWidth < 600,
  })
  const screenshot = await send('Page.captureScreenshot', {
    captureBeyondViewport: true,
    format: 'png',
  })
  await writeFile(screenshotPath, Buffer.from(screenshot.data, 'base64'))
}

socket.close()

const heightCm = Number(pageState?.heightCm)
const heightDifferenceCm = Math.abs(heightCm - expectedHeight)
const heightPassed =
  microphoneMode ||
  (Number.isFinite(heightCm) && heightDifferenceCm < toleranceCm)
const nonGetRequests = requests.filter((request) => request.method !== 'GET')
const foreignRequests = requests.filter(
  (request) => new URL(request.url).origin !== targetOrigin,
)
const requestsAfterRecordingStarted =
  requestsAtRecordingStart === undefined
    ? []
    : requests.slice(requestsAtRecordingStart)
const recordingDurationSeconds = Number(pageState?.recordingDurationSeconds)
const recordingDurationPassed =
  !microphoneMode ||
  (Number.isFinite(recordingDurationSeconds) && recordingDurationSeconds >= 3)
const report = {
  pageState,
  expectedHeightCm: expectedHeight,
  heightDifferenceCm,
  toleranceCm,
  requests,
  nonGetRequests,
  foreignRequests,
  requestsAfterRecordingStarted,
  loadingFailures,
  exceptions,
  screenshotPath: screenshotPath ?? null,
  passed:
    pageState?.state === 'result' &&
    Number.isFinite(heightCm) &&
    heightPassed &&
    recordingDurationPassed &&
    nonGetRequests.length === 0 &&
    foreignRequests.length === 0 &&
    requestsAfterRecordingStarted.length === 0 &&
    loadingFailures.length === 0 &&
    exceptions.length === 0,
}

console.log(JSON.stringify(report, null, 2))
if (!report.passed) process.exitCode = 1

async function clickAction(action) {
  const evaluation = await send('Runtime.evaluate', {
    expression: `document.querySelector('[data-action="${action}"]')?.click()`,
    returnByValue: true,
    userGesture: true,
  })
  if (evaluation.exceptionDetails) {
    throw new Error(`Failed to click ${action}`)
  }
}

async function waitForTarget(baseUrl) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < 10_000) {
    try {
      const response = await fetch(`${baseUrl}/json/list`)
      const targets = await response.json()
      const page = targets.find((candidate) => candidate.type === 'page')
      if (page?.webSocketDebuggerUrl) return page
    } catch {
      // Chrome may still be starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  throw new Error(`Chrome DevTools endpoint was not ready at ${baseUrl}`)
}
