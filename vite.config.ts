import { readFile } from 'node:fs/promises'

import { defineConfig } from 'vitest/config'

const legalAssets = [
  ['LICENSE', 'LICENSE.txt'],
  ['THIRD_PARTY_NOTICES.md', 'THIRD_PARTY_NOTICES.md'],
  ['licenses/Apache-2.0.txt', 'licenses/Apache-2.0.txt'],
  ['licenses/CC-BY-4.0.txt', 'licenses/CC-BY-4.0.txt'],
  ['licenses/ONNX-Runtime-MIT.txt', 'licenses/ONNX-Runtime-MIT.txt'],
  ['licenses/protobufjs-BSD-3-Clause.txt', 'licenses/protobufjs-BSD-3-Clause.txt'],
  ['licenses/platform-MIT.txt', 'licenses/platform-MIT.txt'],
  ['licenses/guid-typescript-ISC.txt', 'licenses/guid-typescript-ISC.txt'],
] as const

export default defineConfig({
  base: './',
  plugins: [
    {
      name: 'bundle-legal-notices',
      apply: 'build',
      async buildStart() {
        for (const [sourcePath, outputPath] of legalAssets) {
          this.emitFile({
            type: 'asset',
            fileName: outputPath,
            source: await readFile(new URL(sourcePath, import.meta.url)),
          })
        }
      },
    },
  ],
  build: {
    target: 'es2022',
  },
  test: {
    environment: 'node',
  },
})
