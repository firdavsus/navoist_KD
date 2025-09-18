const socket = io();
const recordBtn = document.getElementById("record");
const output = document.getElementById("output");

let recording = false;
let audioContext;
let processor;
let source;

recordBtn.onclick = async () => {
  if (!recording) {
    audioContext = new AudioContext({ sampleRate: 16000 });
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (e) => {
      const floatSamples = e.inputBuffer.getChannelData(0);
      socket.emit("audio", floatSamples.buffer); // Send as ArrayBuffer (float32)
    };

    recordBtn.classList.add("recording");
    recording = true;
  } else {
    processor.disconnect();
    source.disconnect();
    audioContext.close();

    recordBtn.classList.remove("recording");
    recording = false;
  }
};

socket.on("text", (data) => {
  output.textContent = data.text;
});
