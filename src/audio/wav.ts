export interface DecodedPcmWav {
  readonly samples: Float32Array;
  readonly sampleRate: number;
}

/** Decodes the uncompressed 16-bit mono WAV used by the golden browser test. */
export function decodePcm16MonoWav(buffer: ArrayBuffer): DecodedPcmWav {
  const view = new DataView(buffer);
  if (
    view.byteLength < 12 ||
    readAscii(view, 0, 4) !== "RIFF" ||
    readAscii(view, 8, 4) !== "WAVE"
  ) {
    throw new TypeError("Invalid WAV container");
  }

  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataLength = 0;

  for (let offset = 12; offset + 8 <= view.byteLength; ) {
    const id = readAscii(view, offset, 4);
    const chunkLength = view.getUint32(offset + 4, true);
    const payloadOffset = offset + 8;
    if (payloadOffset + chunkLength > view.byteLength) {
      throw new TypeError(`Invalid WAV chunk length for ${id}`);
    }

    if (id === "fmt ") {
      if (chunkLength < 16) {
        throw new TypeError("Invalid WAV format chunk");
      }
      format = view.getUint16(payloadOffset, true);
      channels = view.getUint16(payloadOffset + 2, true);
      sampleRate = view.getUint32(payloadOffset + 4, true);
      bitsPerSample = view.getUint16(payloadOffset + 14, true);
    } else if (id === "data") {
      dataOffset = payloadOffset;
      dataLength = chunkLength;
    }

    offset = payloadOffset + chunkLength + (chunkLength % 2);
  }

  if (
    format !== 1 ||
    channels !== 1 ||
    bitsPerSample !== 16 ||
    sampleRate <= 0 ||
    dataOffset < 0 ||
    dataLength % 2 !== 0
  ) {
    throw new TypeError("Only 16-bit mono PCM WAV is supported");
  }

  const samples = new Float32Array(dataLength / 2);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(dataOffset + index * 2, true) / 32_768;
  }
  return { samples, sampleRate };
}

function readAscii(view: DataView, offset: number, length: number): string {
  let result = "";
  for (let index = 0; index < length; index += 1) {
    result += String.fromCharCode(view.getUint8(offset + index));
  }
  return result;
}
