// Collects microphone samples (channel 0) and posts them to the page in
// ~4096-frame batches. Outputs silence; recording only.
class GarvisRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(4096);
    this.filled = 0;
    this.port.onmessage = (event) => {
      if (event.data === 'flush') {
        this.flush();
      }
    };
  }

  flush() {
    if (this.filled > 0) {
      this.port.postMessage(this.buffer.slice(0, this.filled));
      this.filled = 0;
    }
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      let offset = 0;
      while (offset < channel.length) {
        const count = Math.min(channel.length - offset, this.buffer.length - this.filled);
        this.buffer.set(channel.subarray(offset, offset + count), this.filled);
        this.filled += count;
        offset += count;
        if (this.filled === this.buffer.length) {
          this.flush();
        }
      }
    }
    return true;
  }
}

registerProcessor('garvis-recorder', GarvisRecorder);
