function init_state(){
    const f8 = new Float32Array([1.6396663188934326, -1.799014925956726, 0.2766458988189697, 2.800827741622925, 2.989149808883667, -1.7274534702301025, -1.4265387058258057, -0.7298376560211182]);
    const init_tensor = new ort.Tensor("float32", f8, [1, 1, 8])
    return init_tensor;
}

function init_zeros(){
    const f3_1_128 = new Float32Array(3*1*128);
    const zero_tensor = new ort.Tensor("float32", f3_1_128, [3, 1, 128]);
    return zero_tensor;
}

let state = init_state();
let hn = init_zeros();
let cn = init_zeros();

function reset(){
    state = init_state();
    hn = init_zeros();
    cn = init_zeros();
}

let rawKeys = [0, 0];
let keys = [0];

function processKeyDown(event){
    if(event.key == 'r'){
        reset();
    }
    if(event.keyCode === 0x25){
        rawKeys[0] = 1;
    }
    if(event.keyCode === 0x27){
        rawKeys[1] = 1;
    }
}

function processKeyUp(event){
    if(event.keyCode === 0x25){
        rawKeys[0] = 0;
    }
    if(event.keyCode === 0x27){
        rawKeys[1] = 0;
    }
}

document.body.addEventListener("keydown", processKeyDown);
document.body.addEventListener("keyup", processKeyUp);

function append_key_to_state(state){
    let f9 = new Float32Array(9);
    for(let i = 0; i < 8; i++){
        f9[i] = state.cpuData[i];
    }
    f9[8] = keys[0];
    return new ort.Tensor("float32", f9, [1, 1, 9]);
}

function remove_key_from_state(state){
    let f8 = new Float32Array(8);
    for(let i = 0; i < 8; i++){
        f8[i] = state.cpuData[i];
    }
    return new ort.Tensor("float32", f8, [1, 8]);
}

async function run_decoder_inference(session){
    const inputs = {
        'input': remove_key_from_state(state)
    };
    const results = await session.run(inputs);
    return results['output'];
}

function addTensors(t1, t2){
    let f8 = new Float32Array(8);
    for(let i = 0; i < 8; i++){
        f8[i] = t1.cpuData[i] + t2.cpuData[i];
    }
    return new ort.Tensor("float32", f8, [1, 1, 8]);
}

function unnormalizeTensor(t){
    const mu_diff = [ 2.4264e-03, -2.7636e-03,  4.4383e-05,  2.2945e-03,  3.8747e-03, -9.0996e-04, -1.4763e-03,  2.0510e-03];
    const sigma_diff = [0.6051, 0.5535, 0.5855, 0.6326, 0.9133, 0.6024, 0.4599, 0.7968];
    let f8 = new Float32Array(8);
    for(let i = 0; i < 16; i++){
        f8[i] = mu_diff[i] + t.cpuData[i] * sigma_diff[i];
    }
    return new ort.Tensor("float32", f8, [1, 1, 8]);
}

async function run_lstm_inference(session){
    const inputs = {
        'input': append_key_to_state(state),
        'h0': hn,
        'c0': cn
    };
    const results = await session.run(inputs);
    hn = results['hn'];
    cn = results['cn'];
    const unnormalized_output = unnormalizeTensor(results['output']);
    state = addTensors(state, unnormalized_output);
    return unnormalized_output;
}

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function drawTensor(data){
    const imageData = data.toImageData();
    ctx.putImageData(imageData, 0, 0);
}

let playing = true;

const slider_div = document.getElementById("sliders");
let sliders = [];
let labels = [];
for(let i = 0; i < 8; i++){
    const group_div = document.createElement("div");

    const slider = document.createElement("input");
    slider.type = "range";
    slider.style = "width: 50%";
    slider.min = -10;
    slider.max = 10;
    slider.step = "any";
    slider.value = 0;
    group_div.appendChild(slider);

    const label = document.createElement("span");
    label.innerHTML = "0.00";
    group_div.appendChild(label);

    slider.addEventListener("input", () => {
        label.innerHTML = Number(slider.value).toFixed(2);
        state.cpuData[i] = slider.value;
    });

    slider_div.appendChild(group_div);
    sliders.push(slider);
    labels.push(label);
}

async function main(){
    const lstm_session = await ort.InferenceSession.create(
        'onnx_lstm.onnx', 
        { executionProviders: ['cpu'], graphOptimizationLevel: 'all' }
    );

    const decoder_session = await ort.InferenceSession.create(
        'onnx_decoder.onnx', 
        { executionProviders: ['cpu'], graphOptimizationLevel: 'all' }
    );

    setInterval(async () => {
        const t1 = performance.now();
        if(playing){
            if(rawKeys[0] === rawKeys[1]){
                keys[0] = 1-keys[0];
            }else if(rawKeys[0]){
                keys[0] = 0;
            }else{
                keys[0] = 1;
            }
            await run_lstm_inference(lstm_session);
            for(let i = 0; i < 8; i++){
                sliders[i].value = state.cpuData[i];
                labels[i].innerHTML = state.cpuData[i].toFixed(2);
            }
        }
        const decoded_img = await run_decoder_inference(decoder_session);
        drawTensor(decoded_img);
        const elapsed = performance.now() - t1;
        document.getElementById("stats").innerHTML = `CPU Time / Frame: ${elapsed.toFixed(0)} ms, Max FPS: ${(1000/elapsed).toFixed(0)}`;
    }, 39);
}

main();

for(const c of ["left", "right", 'r']){
    const btn = document.getElementById(c);
    const keyCode = (c === "left") ? 0x25 : 0x27;
    btn.addEventListener("mousedown", () => {
        processKeyDown({"keyCode": keyCode, "key": c});
    });
    btn.addEventListener("mouseup", () => {
        processKeyUp({"keyCode": keyCode, "key": c});
    });
    btn.addEventListener("touchstart", () => {
        processKeyDown({"keyCode": keyCode, "key": c});
    });
    btn.addEventListener("touchend", () => {
        processKeyUp({"keyCode": keyCode, "key": c});
    });
}

const play_pause = document.getElementById("play-pause");

play_pause.addEventListener("click", () => {
    if(playing){
        play_pause.innerHTML = "play";
    }else{
        play_pause.innerHTML = "pause";
    }
    playing = !playing;
});