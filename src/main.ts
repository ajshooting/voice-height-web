import './style.css'

import { PcmRecorder } from './audio/recorder'
import {
  InferenceClient,
  InferenceClientError,
} from './inference/client'
import type { InferenceWorkerResponse } from './inference/protocol'

export type AppAction = 'prepare' | 'start' | 'stop' | 'reset'

type StateDetail = {
  detail?: string
}

export type AppState =
  | ({ phase: 'idle' } & StateDetail)
  | ({ phase: 'preparing'; progress: number } & StateDetail)
  | ({ phase: 'ready' } & StateDetail)
  | ({ phase: 'starting' } & StateDetail)
  | ({ phase: 'recording'; elapsedSeconds: number; level: number } & StateDetail)
  | ({ phase: 'analyzing' } & StateDetail)
  | ({ phase: 'result'; heightCm: number } & StateDetail)
  | ({ phase: 'error'; message: string } & StateDetail)

export const APP_ACTION_EVENT = 'voice-height:action'

const actionLabels: Record<AppAction, string> = {
  prepare: 'モデルを準備',
  start: '録音を開始',
  stop: '録音を終了',
  reset: '最初に戻る',
}

const initialState: AppState = { phase: 'idle' }
let currentState: AppState = initialState
let appRoot: HTMLElement | null = null
const wiredRoots = new WeakSet<HTMLElement>()

const appMarkup = `
  <a class="skip-link" href="#main-content">本文へ移動</a>
  <main class="app-shell" id="main-content">
    <header class="intro">
      <h1>声から身長を推定</h1>
      <p class="intro__lead">普段の声を5秒ほど録音すると、身長の目安を表示します。</p>
    </header>

    <section class="estimator" aria-labelledby="estimator-title">
      <div class="status-copy" role="status" aria-live="polite" aria-atomic="true">
        <h2 id="estimator-title" data-ui="status">モデルを準備してください</h2>
        <p data-ui="detail">初回のみ、モデルと推論ランタイムを合計約48 MB取得します。</p>
      </div>

      <div class="progress-block" data-ui="progress-group" hidden>
        <div class="progress-block__labels">
          <span>モデルを読み込み中（約23.5 MB）</span>
          <span data-ui="progress-value">0%</span>
        </div>
        <progress data-ui="progress" max="1" value="0" aria-label="モデルの読み込み進捗"></progress>
      </div>

      <div class="recording-readout" data-ui="recording" hidden>
        <time data-ui="timer" datetime="PT0S">00:00</time>
        <div
          class="level-meter"
          data-ui="level"
          role="meter"
          aria-label="マイクの入力レベル"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow="0"
          aria-valuetext="入力なし"
        >
          <span class="level-meter__fill" data-ui="level-fill"></span>
        </div>
        <span class="sr-only">声の大きさ</span>
      </div>

      <p class="result" data-ui="result" tabindex="-1" hidden>
        <span class="result__prefix">推定身長</span>
        <span class="result__value"><span class="result__about">約</span><strong data-ui="result-value">---</strong><span>cm</span></span>
      </p>

      <p class="error-message" data-ui="error" role="alert" tabindex="-1" hidden></p>

      <div class="actions" data-ui="actions">
        <button class="button button--primary" type="button" data-action="prepare">
          モデルを準備
        </button>
        <button class="button button--primary" type="button" data-action="start" aria-describedby="recording-guide" hidden>
          録音を開始
        </button>
        <button class="button button--stop" type="button" data-action="stop" hidden>
          録音を終了
        </button>
        <button class="button button--secondary" type="button" data-action="reset" hidden>
          最初に戻る
        </button>
      </div>

      <p class="recording-guide" id="recording-guide">
        録音開始時に、ブラウザからマイクの使用許可を求められます。
      </p>
    </section>

    <section class="facts" aria-label="この推定について">
      <article class="fact">
        <h2>プライバシー</h2>
        <p>録音から推定までブラウザ内で処理し、音声をサーバーへ送信・保存しません。</p>
      </article>
      <article class="fact">
        <h2>精度</h2>
        <p>同方式をTIMITで評価した論文上のMAEは約5 cmです。ブラウザ版の実音声での再評価値ではなく、日本語音声での精度も未検証です。</p>
      </article>
      <article class="fact">
        <h2>用途</h2>
        <p>娯楽目的の推定です。健康・本人確認・採用など、重要な判断には使用しないでください。</p>
      </article>
    </section>

    <footer class="legal">
      <a href="./THIRD_PARTY_NOTICES.md">第三者ライセンス・モデル帰属</a>
      <a href="./LICENSE.txt">アプリのMIT License</a>
    </footer>
  </main>
`

function getRequiredElement<T extends Element>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector)

  if (!element) {
    throw new Error(`Required UI element was not found: ${selector}`)
  }

  return element
}

function clamp(value: number, minimum = 0, maximum = 1): number {
  if (!Number.isFinite(value)) {
    return minimum
  }

  return Math.min(maximum, Math.max(minimum, value))
}

function formatTime(elapsedSeconds: number): { label: string; datetime: string } {
  const totalSeconds = Math.max(0, Math.floor(Number.isFinite(elapsedSeconds) ? elapsedSeconds : 0))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  return {
    label: `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`,
    datetime: `PT${totalSeconds}S`,
  }
}

function showButton(
  root: HTMLElement,
  action: AppAction,
  options: { label?: string; disabled?: boolean } = {},
): HTMLButtonElement {
  const button = getRequiredElement<HTMLButtonElement>(root, `[data-action="${action}"]`)
  button.hidden = false
  button.disabled = options.disabled ?? false

  if (options.label) {
    button.textContent = options.label
  }

  return button
}

function resetDynamicUi(root: HTMLElement): void {
  root.querySelectorAll<HTMLButtonElement>('[data-action]').forEach((button) => {
    const action = button.dataset.action as AppAction
    button.hidden = true
    button.disabled = false
    button.textContent = actionLabels[action]
  })

  getRequiredElement<HTMLElement>(root, '[data-ui="progress-group"]').hidden = true
  getRequiredElement<HTMLElement>(root, '[data-ui="recording"]').hidden = true
  getRequiredElement<HTMLElement>(root, '[data-ui="result"]').hidden = true
  getRequiredElement<HTMLElement>(root, '[data-ui="error"]').hidden = true
}

function renderState(root: HTMLElement, state: AppState, shouldFocus = false): void {
  const status = getRequiredElement<HTMLElement>(root, '[data-ui="status"]')
  const detail = getRequiredElement<HTMLElement>(root, '[data-ui="detail"]')
  const progressGroup = getRequiredElement<HTMLElement>(root, '[data-ui="progress-group"]')
  const progress = getRequiredElement<HTMLProgressElement>(root, '[data-ui="progress"]')
  const progressValue = getRequiredElement<HTMLElement>(root, '[data-ui="progress-value"]')
  const recording = getRequiredElement<HTMLElement>(root, '[data-ui="recording"]')
  const timer = getRequiredElement<HTMLTimeElement>(root, '[data-ui="timer"]')
  const level = getRequiredElement<HTMLElement>(root, '[data-ui="level"]')
  const levelFill = getRequiredElement<HTMLElement>(root, '[data-ui="level-fill"]')
  const result = getRequiredElement<HTMLElement>(root, '[data-ui="result"]')
  const resultValue = getRequiredElement<HTMLElement>(root, '[data-ui="result-value"]')
  const error = getRequiredElement<HTMLElement>(root, '[data-ui="error"]')

  if (root.dataset.state === state.phase) {
    if (state.phase === 'preparing') {
      const normalizedProgress = clamp(state.progress)
      progress.value = normalizedProgress
      progressValue.textContent = `${Math.round(normalizedProgress * 100)}%`
      if (
        normalizedProgress === 1 &&
        state.detail &&
        detail.textContent !== state.detail
      ) {
        detail.textContent = state.detail
      }
    } else if (state.phase === 'recording') {
      const formattedTime = formatTime(state.elapsedSeconds)
      const levelPercentage = Math.round(clamp(state.level) * 100)
      timer.textContent = formattedTime.label
      timer.dateTime = formattedTime.datetime
      level.setAttribute('aria-valuenow', String(levelPercentage))
      level.setAttribute(
        'aria-valuetext',
        levelPercentage < 8 ? '入力が小さいです' : levelPercentage > 85 ? '入力が大きいです' : '入力されています',
      )
      levelFill.style.width = `${levelPercentage}%`
    } else if (
      state.phase === 'analyzing' &&
      state.detail &&
      detail.textContent !== state.detail
    ) {
      detail.textContent = state.detail
    }
    return
  }

  root.dataset.state = state.phase
  resetDynamicUi(root)

  switch (state.phase) {
    case 'idle':
      status.textContent = 'モデルを準備してください'
      detail.textContent = state.detail ?? '初回のみ、モデル約23.5 MBと推論ランタイム約24.3 MBを取得します。'
      if (shouldFocus) {
        showButton(root, 'prepare').focus({ preventScroll: true })
      } else {
        showButton(root, 'prepare')
      }
      break

    case 'preparing': {
      const normalizedProgress = clamp(state.progress)
      const percentage = Math.round(normalizedProgress * 100)
      status.textContent = '推定モデルを準備しています'
      detail.textContent = state.detail ?? 'この画面を開いたまま、しばらくお待ちください。'
      progressGroup.hidden = false
      progress.value = normalizedProgress
      progressValue.textContent = `${percentage}%`
      showButton(root, 'prepare', { label: '準備しています…', disabled: true })
      break
    }

    case 'ready':
      status.textContent = '準備できました'
      detail.textContent = state.detail ?? '普段の声で、文章を5秒ほど話してください。内容は自由です。'
      if (shouldFocus) {
        showButton(root, 'start').focus({ preventScroll: true })
      } else {
        showButton(root, 'start')
      }
      break

    case 'starting':
      status.textContent = 'マイクを準備しています'
      detail.textContent = state.detail ?? 'ブラウザのマイク使用確認に応答してください。'
      showButton(root, 'start', { label: 'マイクを準備しています…', disabled: true })
      break

    case 'recording': {
      const formattedTime = formatTime(state.elapsedSeconds)
      const normalizedLevel = clamp(state.level)
      const levelPercentage = Math.round(normalizedLevel * 100)
      status.textContent = '録音しています'
      detail.textContent = state.detail ?? '普段どおりの声で話してください。'
      recording.hidden = false
      timer.textContent = formattedTime.label
      timer.dateTime = formattedTime.datetime
      level.setAttribute('aria-valuenow', String(levelPercentage))
      level.setAttribute(
        'aria-valuetext',
        levelPercentage < 8 ? '入力が小さいです' : levelPercentage > 85 ? '入力が大きいです' : '入力されています',
      )
      levelFill.style.width = `${levelPercentage}%`
      if (shouldFocus) {
        showButton(root, 'stop').focus({ preventScroll: true })
      } else {
        showButton(root, 'stop')
      }
      break
    }

    case 'analyzing':
      status.textContent = '声の特徴を解析しています'
      detail.textContent = state.detail ?? '推定が終わるまで、少しだけお待ちください。'
      break

    case 'result': {
      const roundedHeight = Math.round(state.heightCm)
      status.textContent = '推定が完了しました'
      detail.textContent = state.detail ?? '声だけから推定した参考値です。実際の身長とは異なる場合があります。'
      resultValue.textContent = Number.isFinite(roundedHeight) ? String(roundedHeight) : '---'
      result.hidden = false
      showButton(root, 'start', { label: 'もう一度録音' })
      showButton(root, 'reset')

      if (shouldFocus) {
        result.focus({ preventScroll: true })
      }
      break
    }

    case 'error':
      status.textContent = '処理を続けられませんでした'
      detail.textContent = state.detail ?? '設定を確認して、もう一度お試しください。'
      error.textContent = state.message
      error.hidden = false
      showButton(root, 'reset')

      if (shouldFocus) {
        error.focus({ preventScroll: true })
      }
      break
  }
}

function wireActions(root: HTMLElement): void {
  if (wiredRoots.has(root)) {
    return
  }

  root.addEventListener('click', (event) => {
    const target = event.target

    if (!(target instanceof Element)) {
      return
    }

    const button = target.closest<HTMLButtonElement>('button[data-action]')

    if (!button || button.disabled || !root.contains(button)) {
      return
    }

    const action = button.dataset.action as AppAction
    root.dispatchEvent(
      new CustomEvent<{ action: AppAction }>(APP_ACTION_EVENT, {
        bubbles: true,
        detail: { action },
      }),
    )
  })

  wiredRoots.add(root)
}

/**
 * UIシェルを描画します。操作は `voice-height:action` イベントとして通知されます。
 */
export function renderApp(root?: HTMLElement): HTMLElement {
  if (typeof document === 'undefined') {
    throw new Error('renderApp() requires a browser document.')
  }

  const target = root ?? document.querySelector<HTMLElement>('#app')

  if (!target) {
    throw new Error('Application root was not found.')
  }

  target.innerHTML = appMarkup
  appRoot = target
  wireActions(target)
  renderState(target, currentState)

  return target
}

/**
 * 録音・推論処理から渡された状態だけをUIへ反映します。
 * `progress` と `level` は0〜1、`elapsedSeconds` は秒単位です。
 */
export function setAppState(nextState: AppState): void {
  const previousPhase = currentState.phase
  currentState = nextState

  if (typeof document === 'undefined') {
    return
  }

  const target = appRoot ?? renderApp()
  renderState(target, nextState, previousPhase !== nextState.phase)
}

const MAX_RECORDING_SECONDS = 10

export function initializeApp(root: HTMLElement): () => void {
  let inference = new InferenceClient()
  const recorder = new PcmRecorder()
  let modelState: 'idle' | 'preparing' | 'ready' = 'idle'
  let starting = false
  let stopping = false
  let operationVersion = 0
  const referenceParameters = new URLSearchParams(window.location.search)
  const referenceMode =
    import.meta.env.DEV && referenceParameters.has('reference')
  let referenceStarted = false

  const unsubscribeRecorder = recorder.subscribe(({ elapsedSeconds, level }) => {
    if (!recorder.recording || stopping) {
      return
    }

    setAppState({ phase: 'recording', elapsedSeconds, level })
    if (elapsedSeconds >= MAX_RECORDING_SECONDS) {
      void stopAndAnalyze()
    }
  })

  let unsubscribeInference = inference.subscribe((event) => {
    handleInferenceEvent(event)
  })

  const handleAction = (event: Event) => {
    const detail = (event as CustomEvent<{ action: AppAction }>).detail
    if (!detail) {
      return
    }

    switch (detail.action) {
      case 'prepare':
        prepareModel()
        break
      case 'start':
        void startRecording()
        break
      case 'stop':
        void stopAndAnalyze()
        break
      case 'reset':
        void reset()
        break
    }
  }
  root.addEventListener(APP_ACTION_EVENT, handleAction)

  function prepareModel(): void {
    if (modelState !== 'idle') {
      return
    }

    modelState = 'preparing'
    setAppState({ phase: 'preparing', progress: 0 })
    inference.prepare({
      model: new URL('models/ecapa-voxceleb.onnx', document.baseURI).href,
      regressors: new URL('models/height-regressors.json', document.baseURI).href,
      preferredProvider:
        referenceMode && referenceParameters.get('provider') === 'wasm'
          ? 'wasm'
          : undefined,
    })
  }

  function replaceInferenceWorker(): void {
    unsubscribeInference()
    inference.dispose()
    inference = new InferenceClient()
    unsubscribeInference = inference.subscribe((event) => {
      handleInferenceEvent(event)
    })
  }

  function handleInferenceEvent(event: InferenceWorkerResponse): void {
    switch (event.type) {
      case 'download-progress': {
        if (modelState !== 'preparing') {
          return
        }
        const hasKnownTotal =
          event.totalBytes !== undefined &&
          Number.isFinite(event.totalBytes) &&
          event.totalBytes > 0
        const progress = hasKnownTotal
          ? event.loadedBytes / event.totalBytes!
          : 0
        const loadedMb = event.loadedBytes / 1_000_000
        const detail = hasKnownTotal
          ? `${loadedMb.toFixed(1)} / ${(event.totalBytes! / 1_000_000).toFixed(1)} MB`
          : `${loadedMb.toFixed(1)} MB を受信しました。`
        setAppState({ phase: 'preparing', progress, detail })
        break
      }
      case 'ready':
        modelState = 'ready'
        setAppState({
          phase: 'ready',
          detail: `準備完了。${providerLabel(event.provider)}で端末内推論します。普段の声で5秒ほど話してください。`,
        })
        if (referenceMode && !referenceStarted) {
          referenceStarted = true
          void runDevelopmentReference()
        }
        break
      case 'initializing':
        if (modelState === 'preparing') {
          setAppState({
            phase: 'preparing',
            progress: 1,
            detail: '推論ランタイムを読み込み、モデルを初期化しています。',
          })
        }
        break
      case 'stage':
        setAppState({
          phase: 'analyzing',
          detail:
            event.stage === 'preprocessing'
              ? '録音を推論用の音声特徴へ変換しています。'
              : '声の特徴から身長の目安を計算しています。',
        })
        break
      case 'error':
        if (event.id === undefined) {
          modelState = 'idle'
          replaceInferenceWorker()
          setAppState({ phase: 'error', message: event.message })
        }
        break
      case 'result':
        break
    }
  }

  async function startRecording(): Promise<void> {
    if (modelState !== 'ready' || starting || recorder.recording || stopping) {
      return
    }

    operationVersion += 1
    const version = operationVersion
    starting = true
    setAppState({ phase: 'starting' })
    delete root.dataset.recordingDurationSeconds
    if (import.meta.env.DEV) {
      delete root.dataset.resultHeightCm
      delete root.dataset.resultProvider
    }
    try {
      await recorder.start()
      if (version !== operationVersion) {
        return
      }
      setAppState({ phase: 'recording', elapsedSeconds: 0, level: 0 })
    } catch (error) {
      if (version === operationVersion) {
        setAppState({
          phase: 'error',
          message: microphoneErrorMessage(error),
        })
      }
    } finally {
      starting = false
    }
  }

  async function stopAndAnalyze(): Promise<void> {
    if (!recorder.recording || stopping) {
      return
    }

    stopping = true
    const version = operationVersion
    setAppState({ phase: 'analyzing', detail: '録音を終了しています。' })
    try {
      const audio = await recorder.stop()
      root.dataset.recordingDurationSeconds = String(audio.durationSeconds)
      const result = await inference.infer(audio.samples, audio.sampleRate)
      if (version !== operationVersion) {
        return
      }

      if (import.meta.env.DEV) {
        root.dataset.resultHeightCm = String(result.heightCm)
        root.dataset.resultProvider = result.metrics.provider
      }

      setAppState({
        phase: 'result',
        heightCm: result.heightCm,
        detail: `録音 ${audio.durationSeconds.toFixed(1)}秒を、端末内で${(result.metrics.totalMs / 1000).toFixed(1)}秒で解析しました（${providerLabel(result.metrics.provider)}）。`,
      })
    } catch (error) {
      if (version !== operationVersion) {
        return
      }
      setAppState({
        phase: 'error',
        message:
          error instanceof InferenceClientError
            ? error.message
            : '録音または推定処理に失敗しました。もう一度お試しください。',
      })
    } finally {
      stopping = false
    }
  }

  async function reset(): Promise<void> {
    operationVersion += 1
    stopping = true
    try {
      await recorder.cancel()
    } finally {
      stopping = false
    }
    setAppState(
      modelState === 'ready'
        ? { phase: 'ready' }
        : { phase: 'idle' },
    )
  }

  async function runDevelopmentReference(): Promise<void> {
    try {
      setAppState({
        phase: 'analyzing',
        detail: '開発用の固定WAVでブラウザ推論を検証しています。',
      })
      const response = await fetch(
        new URL('tests/fixtures/speechbrain-example1.wav', document.baseURI),
        { cache: 'no-store' },
      )
      if (!response.ok) {
        throw new Error(`Reference WAV request failed (${response.status})`)
      }
      const { decodePcm16MonoWav } = await import('./audio/wav')
      const audio = decodePcm16MonoWav(await response.arrayBuffer())
      const result = await inference.infer(audio.samples, audio.sampleRate)
      root.dataset.referenceHeightCm = String(result.heightCm)
      root.dataset.referenceProvider = result.metrics.provider
      root.dataset.resultHeightCm = String(result.heightCm)
      root.dataset.resultProvider = result.metrics.provider
      setAppState({
        phase: 'result',
        heightCm: result.heightCm,
        detail: `固定WAVの検証結果：${result.heightCm.toFixed(6)} cm（${providerLabel(result.metrics.provider)}）`,
      })
    } catch (error) {
      setAppState({
        phase: 'error',
        message:
          error instanceof Error
            ? error.message
            : '固定WAVのブラウザ検証に失敗しました。',
      })
    }
  }

  if (referenceMode) {
    queueMicrotask(prepareModel)
  }

  return () => {
    operationVersion += 1
    root.removeEventListener(APP_ACTION_EVENT, handleAction)
    unsubscribeRecorder()
    unsubscribeInference()
    void recorder.cancel()
    inference.dispose()
  }
}

function providerLabel(provider: 'webgpu' | 'wasm'): string {
  return provider === 'webgpu' ? 'WebGPU' : 'WASM'
}

function microphoneErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      return 'マイクの使用が許可されませんでした。ブラウザの権限設定を確認してください。'
    }
    if (error.name === 'NotFoundError') {
      return '利用できるマイクが見つかりませんでした。接続を確認してください。'
    }
  }
  return error instanceof Error
    ? error.message
    : 'マイクを開始できませんでした。'
}

if (typeof document !== 'undefined') {
  const root = renderApp()
  const dispose = initializeApp(root)
  window.addEventListener('pagehide', dispose, { once: true })
}
