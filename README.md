# 声から身長を推定

マイクで録音した声から、身長の目安を推定する完全静的なWebアプリです。録音、音声特徴抽出、ニューラルネット推論、回帰はすべてブラウザ内で行います。推論APIや音声アップロードはありません。

> [!IMPORTANT]
> これは娯楽・技術検証用の推定です。健康、本人確認、採用などの判断には使用しないでください。

## 仕組み

```text
マイク PCM
  → mono / 16 kHz
  → SpeechBrain互換 80-bin FBank
  → ECAPA-TDNN（192次元embedding）
  → 内部の二値ルーティング
  → HeightCeleb男女別PLS回帰
  → 身長 [cm]
```

- speaker encoder: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- height model: [`stachu86/HeightCeleb-estimator-demo`](https://huggingface.co/spaces/stachu86/HeightCeleb-estimator-demo)
- browser runtime: ONNX Runtime Web（WebGPU優先、WASM fallback）
- UI: Vite + Vanilla TypeScript

モデル内部では学習データ由来の二値分類で回帰器を切り替えます。この値は本人の性別を決めるものではなく、UIにも表示しません。この二値構造しか扱えないこと自体がモデルの制約です。

## 精度の読み方

[HeightCeleb論文 Table 3](https://arxiv.org/abs/2410.12668)のTIMIT評価では、この階層モデルの平均絶対誤差（MAE）は男性データ `4.81 cm`、女性データ `4.73 cm` です。これは「個々の推定が約5 cm以内に入る確率」や信頼区間ではありません。

日本語話者、子ども、一般的なスマートフォン録音などでの精度は確認されていません。声色、体調、マイク、反響、雑音でも結果は変わります。また、音声から個人の身長を正確に決定できるものではありません。

## プライバシーと通信

- 録音PCM、音声特徴、embedding、推定結果を外部へ送信・保存しません。
- Analyticsや外部フォントは使用しません。
- 通信はページ、推論ランタイム、ONNX、回帰係数を取得する同一originの `GET` だけです。
- 初回準備は、量子化モデル約 `23.5 MB` とWebGPU/WASMランタイム約 `24.3 MB`、合計約 `48 MB` です。以後は通常のブラウザHTTPキャッシュを利用します。
- `23.5 MB` は配布・ダウンロード時の大きさです。量子化重みは実行時にFP32へ戻すため、常駐メモリや演算量まで4分の1になるわけではありません。

マイクは安全なコンテキスト（HTTPSまたはlocalhost）でのみ使用できます。将来GitHub Pagesへ配置する場合も、モデルはサイトと同じ静的ファイルとして配信されます。

## 開発

Node.js環境でpnpmを使います。

```sh
pnpm install
pnpm dev
```

検証と静的ビルド:

```sh
pnpm test
pnpm check
pnpm build
```

`vite.config.ts` の `base: "./"` により、生成した `dist/` はサブパスを含む静的ホスティングで利用できます。

`.github/workflows/deploy-pages.yml` は、`main`へのpushまたは手動実行時に、固定lockfileで依存を導入し、テストとproduction buildを通した`dist/`だけをGitHub Pagesへ送ります。GitHub側では、リポジトリのSettings → Pages → Build and deployment → Sourceを「GitHub Actions」に設定してください。

### 固定WAVのブラウザ検証

開発サーバーで `/?reference=1` を開くと、同梱のSpeechBrain example WAVを使って、FBankから最終身長まで自動検証できます。この経路はproduction buildから除外されます。

## 再現性と数値検証

変換元はrevisionとSHA-256を固定しています。

- HeightCeleb Space revision: `cd5feff37a9b8f77a69380dcc90fc55756f4330f`
- SpeechBrain model revision: `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`
- SpeechBrain学習時コード: `aa018540`
- 元ECAPA checkpoint SHA-256: `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2`

現在の移植検証結果:

| 経路 | 基準との差 |
|---|---:|
| PyTorch → FP32 ONNX（1 / 5 / 10秒） | cosine `1.0`、最大絶対差 `≤ 0.000134` |
| SpeechBrain Python → TypeScript FBank | 最大絶対差 `0.000488`、RMSE `0.0000203` |
| FP32 → mixed weight-INT8 ONNX（1秒 / 公式WAV / 5秒 / 10秒） | 最小cosine `0.999692`、最大身長差 `0.097364 cm`、全ルート一致 |

配布ONNXは、38個のConvのうち入力Conv 1個と最初のSE block内2個（計3個）をFP32に保ち、残る35 Convの重みをoutput-channelごとの対称INT8へ変換しています。演算自体はFP32です。83,475,226 bytesのFP32 ONNXから23,470,546 bytesへ縮小しました。詳細なハッシュ、各ケースの値、変換設定はモデル横のmanifestに記録しています。

これらは移植による数値差の検証であり、現実の音声に対する身長推定精度の再評価ではありません。

Pythonは公開サイトでは使いません。公式artifactからの参照値作成・ONNX変換だけを、PEP 723で依存を固定した `uv run --isolated --no-project` の一時環境で行います。checkpoint、pickle、FP32 ONNX、Python環境はリポジトリへ含めません。

固定revisionから取得済みのsource artifactを`<source-dir>/height/`と`<source-dir>/ecapa/`へ置いた場合の再生成コマンドは次のとおりです。export前にpickle、checkpoint、HyperPyYAML設定、fixtureを既知SHA-256で検証し、compress前にも公式FP32 ONNXのSHA-256を検証します。

```sh
uv run --isolated --no-project tools/export_reference.py \
  --repo-root . \
  --source-dir <source-dir> \
  --artifacts-dir <temporary-artifacts-dir>

uv run --isolated --no-project tools/compress_model.py \
  --input <temporary-artifacts-dir>/ecapa-embedding-fp32.onnx \
  --output public/models/ecapa-voxceleb.onnx \
  --manifest public/models/ecapa-voxceleb.manifest.json \
  --fixture-features tests/fixtures/speechbrain-example1.features.f32 \
  --fixture-reference tests/fixtures/speechbrain-example1.reference.json \
  --regressors public/models/height-regressors.json \
  --exclude-weight 'embedding_model.blocks.0*' \
  --exclude-weight 'embedding_model.blocks.1.se_block*' \
  --height-gate-all-cases \
  --force
```

本番ビルドには第三者通知と各ライセンス本文も含まれ、画面下部から参照できます。

## ライセンス

アプリ固有のコードはMIT Licenseです。配布モデルと依存ライブラリには別のライセンスが適用されます。帰属、固定revision、変換内容、学習データ由来の注意事項は [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) を参照してください。

なお、モデルカード上のライセンスは再配布の根拠になりますが、VoxCelebの元音声に関する著作権・肖像・プライバシーまで保証するものではありません。本リポジトリは学習音声や人物データを配布しません。
