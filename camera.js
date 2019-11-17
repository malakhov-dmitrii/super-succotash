/* eslint-disable quotes */
/* eslint-disable arrow-parens */
/* eslint-disable camelcase */
/* eslint-disable comma-dangle */
/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from "@tensorflow-models/posenet";
import dat from "dat.gui";
import Stats from "stats.js";
import {
  drawBoundingBox,
  drawKeypoints,
  drawSkeleton,
  isMobile,
  toggleLoadingUI,
  tryResNetButtonName,
  tryResNetButtonText,
  updateTryResNetButtonDatGuiCss
} from "./demo_util";
const similarity = require("compute-cosine-similarity");
// const distance = require("euclidean-distance");

const TARGET_POS_1 = [{
  "score": 0.9908686876296997,
  "part": "nose",
  "position": {
    "x": 415.93650309244794,
    "y": 272.3636067708333
  }
}, {
  "score": 0.9932006001472473,
  "part": "leftEye",
  "position": {
    "x": 418.77252197265625,
    "y": 262.5507609049479
  }
}, {
  "score": 0.8547982573509216,
  "part": "rightEye",
  "position": {
    "x": 424.187744140625,
    "y": 274.3563028971354
  }
}, {
  "score": 0.9743635058403015,
  "part": "leftEar",
  "position": {
    "x": 414.1456298828125,
    "y": 245.94010416666663
  }
}, {
  "score": 0.23263892531394958,
  "part": "rightEar",
  "position": {
    "x": 427.6547037760417,
    "y": 275.5929972330729
  }
}, {
  "score": 0.9953951835632324,
  "part": "leftShoulder",
  "position": {
    "x": 339.6566162109375,
    "y": 229.3892822265625
  }
}, {
  "score": 0.9445495009422302,
  "part": "rightShoulder",
  "position": {
    "x": 383.0687662760417,
    "y": 332.59855143229163
  }
}, {
  "score": 0.9884846806526184,
  "part": "leftElbow",
  "position": {
    "x": 286.05466715494794,
    "y": 171.34035237630206
  }
}, {
  "score": 0.9777011275291443,
  "part": "rightElbow",
  "position": {
    "x": 369.07334391276044,
    "y": 395.269775390625
  }
}, {
  "score": 0.9650675058364868,
  "part": "leftWrist",
  "position": {
    "x": 233.67561848958337,
    "y": 103.88766479492188
  }
}, {
  "score": 0.9833654761314392,
  "part": "rightWrist",
  "position": {
    "x": 364.90559895833337,
    "y": 475.4837646484375
  }
}, {
  "score": 0.9955465793609619,
  "part": "leftHip",
  "position": {
    "x": 263.6195068359375,
    "y": 336.44140625
  }
}, {
  "score": 0.9902048110961914,
  "part": "rightHip",
  "position": {
    "x": 297.70245361328125,
    "y": 363.806640625
  }
}, {
  "score": 0.971397876739502,
  "part": "leftKnee",
  "position": {
    "x": 204.50716145833337,
    "y": 439.9601643880208
  }
}, {
  "score": 0.9034759402275085,
  "part": "rightKnee",
  "position": {
    "x": 306.1921793619792,
    "y": 455.48771158854163
  }
}, {
  "score": 0.9358130693435669,
  "part": "leftAnkle",
  "position": {
    "x": 171.828125,
    "y": 522.3367919921875
  }
}, {
  "score": 0.8826147317886353,
  "part": "rightAnkle",
  "position": {
    "x": 331.78057861328125,
    "y": 513.4849446614583
  }
}].map(item => [item.position.x, item.position.y]).flat();

const TARGET_POS_2 = [{
  "score": 0.9991829991340637,
  "part": "nose",
  "position": {
    "x": 247.74943033854169,
    "y": 156.07637532552081
  }
}, {
  "score": 0.995419979095459,
  "part": "leftEye",
  "position": {
    "x": 234.01680501302087,
    "y": 150.19091796875
  }
}, {
  "score": 0.9979812502861023,
  "part": "rightEye",
  "position": {
    "x": 257.09663899739587,
    "y": 148.95231119791666
  }
}, {
  "score": 0.9309232234954834,
  "part": "leftEar",
  "position": {
    "x": 223.02632649739587,
    "y": 156.54903157552081
  }
}, {
  "score": 0.953841507434845,
  "part": "rightEar",
  "position": {
    "x": 272.06247965494794,
    "y": 151.66215006510416
  }
}, {
  "score": 0.9993361830711365,
  "part": "leftShoulder",
  "position": {
    "x": 213.07816569010419,
    "y": 208.04302978515625
  }
}, {
  "score": 0.9985609650611877,
  "part": "rightShoulder",
  "position": {
    "x": 297.1982828776042,
    "y": 190.80411783854166
  }
}, {
  "score": 0.9982522130012512,
  "part": "leftElbow",
  "position": {
    "x": 136.6318359375,
    "y": 217.35866292317706
  }
}, {
  "score": 0.9969882369041443,
  "part": "rightElbow",
  "position": {
    "x": 329.58477783203125,
    "y": 220.26560465494788
  }
}, {
  "score": 0.982961118221283,
  "part": "leftWrist",
  "position": {
    "x": 54.04475911458337,
    "y": 220.89723714192706
  }
}, {
  "score": 0.9922548532485962,
  "part": "rightWrist",
  "position": {
    "x": 316.40753173828125,
    "y": 237.23555501302081
  }
}, {
  "score": 0.9977959394454956,
  "part": "leftHip",
  "position": {
    "x": 273.5451863606771,
    "y": 320.2342529296875
  }
}, {
  "score": 0.9996669292449951,
  "part": "rightHip",
  "position": {
    "x": 335.82442220052087,
    "y": 299.6274007161458
  }
}, {
  "score": 0.9937479496002197,
  "part": "leftKnee",
  "position": {
    "x": 283.44258626302087,
    "y": 416.9686279296875
  }
}, {
  "score": 0.9971883893013,
  "part": "rightKnee",
  "position": {
    "x": 405.29905192057294,
    "y": 368.97538248697913
  }
}, {
  "score": 0.9565935730934143,
  "part": "leftAnkle",
  "position": {
    "x": 275.13584391276044,
    "y": 509.43538411458326
  }
}, {
  "score": 0.9541161060333252,
  "part": "rightAnkle",
  "position": {
    "x": 483.2630818684896,
    "y": 436.8101806640625
  }
}].map(item => [item.position.x, item.position.y]).flat();

const TARGET_POS_3 = [{
  "score": 0.07121406495571136,
  "part": "nose",
  "position": {
    "x": 289.2438151041667,
    "y": 94.34494018554688
  }
}, {
  "score": 0.06686632335186005,
  "part": "leftEye",
  "position": {
    "x": 635.5045954386393,
    "y": 117.53727213541666
  }
}, {
  "score": 0.08135952055454254,
  "part": "rightEye",
  "position": {
    "x": 291.4835611979167,
    "y": 90.0947265625
  }
}, {
  "score": 0.9765129089355469,
  "part": "leftEar",
  "position": {
    "x": 356.6738688151042,
    "y": 97.97532145182291
  }
}, {
  "score": 0.9591643810272217,
  "part": "rightEar",
  "position": {
    "x": 296.8337809244792,
    "y": 93.65211995442706
  }
}, {
  "score": 0.9938938617706299,
  "part": "leftShoulder",
  "position": {
    "x": 392.890625,
    "y": 158.47733561197916
  }
}, {
  "score": 0.9985998272895813,
  "part": "rightShoulder",
  "position": {
    "x": 260.4107666015625,
    "y": 150.42964680989581
  }
}, {
  "score": 0.9513225555419922,
  "part": "leftElbow",
  "position": {
    "x": 467.5526021321615,
    "y": 168.97133382161456
  }
}, {
  "score": 0.9959094524383545,
  "part": "rightElbow",
  "position": {
    "x": 179.5926513671875,
    "y": 131.32814534505206
  }
}, {
  "score": 0.9644811749458313,
  "part": "leftWrist",
  "position": {
    "x": 549.9369812011719,
    "y": 143.46809895833331
  }
}, {
  "score": 0.9770116209983826,
  "part": "rightWrist",
  "position": {
    "x": 92.71769205729169,
    "y": 90.124755859375
  }
}, {
  "score": 0.9919902682304382,
  "part": "leftHip",
  "position": {
    "x": 344.3585408528646,
    "y": 347.7084554036458
  }
}, {
  "score": 0.9737558960914612,
  "part": "rightHip",
  "position": {
    "x": 262.13116455078125,
    "y": 339.0426839192708
  }
}, {
  "score": 0.9826898574829102,
  "part": "leftKnee",
  "position": {
    "x": 315.68357340494794,
    "y": 469.7528076171875
  }
}, {
  "score": 0.9939213991165161,
  "part": "rightKnee",
  "position": {
    "x": 260.64963785807294,
    "y": 460.00077311197913
  }
}, {
  "score": 0.5993714928627014,
  "part": "leftAnkle",
  "position": {
    "x": 318.701904296875,
    "y": 544.7017822265625
  }
}, {
  "score": 0.5228094458580017,
  "part": "rightAnkle",
  "position": {
    "x": 255.26517740885419,
    "y": 560.9686686197916
  }
}].map(item => [item.position.x, item.position.y]).flat();

const TARGET_POS_4 = [{
  "score": 0.9736052751541138,
  "part": "nose",
  "position": {
    "x": 227.93656412760419,
    "y": 231.23138427734375
  }
}, {
  "score": 0.9057209491729736,
  "part": "leftEye",
  "position": {
    "x": 217.8642578125,
    "y": 235.34895833333331
  }
}, {
  "score": 0.9607812762260437,
  "part": "rightEye",
  "position": {
    "x": 229.98763020833337,
    "y": 224.60563151041663
  }
}, {
  "score": 0.36542895436286926,
  "part": "leftEar",
  "position": {
    "x": 221.40983072916669,
    "y": 241.76798502604163
  }
}, {
  "score": 0.8499851226806641,
  "part": "rightEar",
  "position": {
    "x": 251.45902506510419,
    "y": 209.08685302734375
  }
}, {
  "score": 0.9991108775138855,
  "part": "leftShoulder",
  "position": {
    "x": 247.07421875,
    "y": 280.72589111328125
  }
}, {
  "score": 0.9957950115203857,
  "part": "rightShoulder",
  "position": {
    "x": 292.9853515625,
    "y": 215.8055419921875
  }
}, {
  "score": 0.9941942095756531,
  "part": "leftElbow",
  "position": {
    "x": 260.4375,
    "y": 347.35477701822913
  }
}, {
  "score": 0.9931871891021729,
  "part": "rightElbow",
  "position": {
    "x": 293.35137939453125,
    "y": 131.34718831380206
  }
}, {
  "score": 0.9799391031265259,
  "part": "leftWrist",
  "position": {
    "x": 258.7060750325521,
    "y": 425.9352620442708
  }
}, {
  "score": 0.8207806348800659,
  "part": "rightWrist",
  "position": {
    "x": 286.2252197265625,
    "y": 59.284027099609375
  }
}, {
  "score": 0.9987395405769348,
  "part": "leftHip",
  "position": {
    "x": 331.9275309244792,
    "y": 350.7056884765625
  }
}, {
  "score": 0.996920108795166,
  "part": "rightHip",
  "position": {
    "x": 362.64082845052087,
    "y": 333.686767578125
  }
}, {
  "score": 0.8712299466133118,
  "part": "leftKnee",
  "position": {
    "x": 279.6968180338542,
    "y": 439.2561848958333
  }
}, {
  "score": 0.9545165300369263,
  "part": "rightKnee",
  "position": {
    "x": 394.1129557291667,
    "y": 436.021240234375
  }
}, {
  "score": 0.574531078338623,
  "part": "leftAnkle",
  "position": {
    "x": 410.92087809244794,
    "y": 518.4794921875
  }
}, {
  "score": 0.780026376247406,
  "part": "rightAnkle",
  "position": {
    "x": 255.26094563802087,
    "y": 506.908203125
  }
}].map(item => [item.position.x, item.position.y]).flat();

const SUPER_COMBO = ['bu', 'ld', 'lu', 'rd', 'ld'];

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

let tickCounter = 60;
let tickNormalizerContainerY = [];

let isPoseDetected = false;
let isShot = false;
let isNeedToShoot = false;

let currentPose = -2;
let prevPose = -1;

let isSleeping = false;
let prevPoseTimestamp = -1;
let currentPoseTimestamp = -1;

let timeStart;
let timeEnd;

let isPose1 = true;
let isNone = false;


let isLeftUp = false;
let isLeftDown = false;
let isBothUp = false;
let isBothDown = false;
let isRightUp = false;
let isRightDown = false;

const shoot = document.querySelector("#shoot");

shoot.addEventListener("click", () => {
  console.log("Shot!");
  isNeedToShoot = true;
  isShot = false;
});
const stop = document.querySelector("#stop");
const pos1 = document.querySelector("#pos1");
const pos2 = document.querySelector("#pos2");
const pos3 = document.querySelector("#pos3");
const pos4 = document.querySelector("#pos4");

pos1.addEventListener("click", () => {
  updatePos({
    last_pose: {
      time_start: new Date().getTime() - 4000,
      time_end: new Date().getTime() - 500,
      pose_id: 1,
    }
  });
});

pos2.addEventListener("click", () => {
  updatePos({
    last_pose: {
      time_start: new Date().getTime() - 3000,
      time_end: new Date().getTime() - 200,
      pose_id: 2,
    }
  });
});

pos3.addEventListener("click", () => {
  updatePos({
    last_pose: {
      time_start: new Date().getTime() - 5000,
      time_end: new Date().getTime() - 10,
      pose_id: 3,
    }
  });
});

pos4.addEventListener("click", () => {
  updatePos({
    last_pose: {
      time_start: new Date().getTime() - 4000,
      time_end: new Date().getTime() - 50,
      pose_id: 4,
    }
  });
});

stop.addEventListener("click", () => {
  updatePos({
    last_pose: {
      time_start: timeStart,
      time_end: timeEnd,
      pose_id: 1,
    }
  });
});

// let counter = 10;
// setInterval(() => {
//   console.log(--counter);
//   if (counter < 0) {
//     isNeedToShoot = true;
//   }
// }, 1000);

const updatePos = data => {
  // fetch("https://arngry.herokuapp.com/pose", {
  fetch("http://localhost:3000/pose", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(res => console.log(res));
};

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Browser API navigator.mediaDevices.getUserMedia not available");
  }

  const video = document.getElementById("video");
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  console.log("Getting user media");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight
    }
  });
  console.log("Stream is ready");
  video.srcObject = stream;

  return new Promise(resolve => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.5 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 500;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 250;

const guiState = {
  algorithm: "single-pose",
  input: {
    architecture: "ResNet50",
    outputStride: defaultResNetStride,
    inputResolution: defaultResNetInputResolution,
    multiplier: defaultResNetMultiplier,
    quantBytes: defaultQuantBytes
  },
  singlePoseDetection: {
    minPoseConfidence: 0.3,
    minPartConfidence: 0.5
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.3,
    minPartConfidence: 0.3,
    nmsRadius: 30.0
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false
  },
  net: null
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[1].deviceId;
  }

  const gui = new dat.GUI({
    width: 300
  });

  let architectureController = null;
  guiState[tryResNetButtonName] = function() {
    architectureController.setValue("ResNet50");
  };
  gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
  updateTryResNetButtonDatGuiCss();

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController = gui.add(guiState, "algorithm", ["single-pose", "multi-pose"]);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder("Input");
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  architectureController = input.add(guiState.input, "architecture", ["MobileNetV1", "ResNet50"]);
  guiState.architecture = guiState.input.architecture;
  // Input resolution:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The higher the value of the input
  // resolution the better the accuracy but slower the speed.
  let inputResolutionController = null;

  function updateGuiInputResolution(inputResolution, inputResolutionArray) {
    if (inputResolutionController) {
      inputResolutionController.remove();
    }
    guiState.inputResolution = inputResolution;
    guiState.input.inputResolution = inputResolution;
    inputResolutionController = input.add(
      guiState.input,
      "inputResolution",
      inputResolutionArray
    );
    inputResolutionController.onChange(function(inputResolution) {
      guiState.changeToInputResolution = inputResolution;
    });
  }

  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  let outputStrideController = null;

  function updateGuiOutputStride(outputStride, outputStrideArray) {
    if (outputStrideController) {
      outputStrideController.remove();
    }
    guiState.outputStride = outputStride;
    guiState.input.outputStride = outputStride;
    outputStrideController = input.add(guiState.input, "outputStride", outputStrideArray);
    outputStrideController.onChange(function(outputStride) {
      guiState.changeToOutputStride = outputStride;
    });
  }

  // Multiplier: this parameter affects the number of feature map channels in
  // the MobileNet. The higher the value, the higher the accuracy but slower the
  // speed, the lower the value the faster the speed but lower the accuracy.
  let multiplierController = null;

  function updateGuiMultiplier(multiplier, multiplierArray) {
    if (multiplierController) {
      multiplierController.remove();
    }
    guiState.multiplier = multiplier;
    guiState.input.multiplier = multiplier;
    multiplierController = input.add(guiState.input, "multiplier", multiplierArray);
    multiplierController.onChange(function(multiplier) {
      guiState.changeToMultiplier = multiplier;
    });
  }

  // QuantBytes: this parameter affects weight quantization in the ResNet50
  // model. The available options are 1 byte, 2 bytes, and 4 bytes. The higher
  // the value, the larger the model size and thus the longer the loading time,
  // the lower the value, the shorter the loading time but lower the accuracy.
  let quantBytesController = null;

  function updateGuiQuantBytes(quantBytes, quantBytesArray) {
    if (quantBytesController) {
      quantBytesController.remove();
    }
    guiState.quantBytes = +quantBytes;
    guiState.input.quantBytes = +quantBytes;
    quantBytesController = input.add(guiState.input, "quantBytes", quantBytesArray);
    quantBytesController.onChange(function(quantBytes) {
      guiState.changeToQuantBytes = +quantBytes;
    });
  }

  function updateGui() {
    if (guiState.input.architecture === "MobileNetV1") {
      updateGuiInputResolution(defaultMobileNetInputResolution, [
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800
      ]);
      updateGuiOutputStride(defaultMobileNetStride, [8, 16]);
      updateGuiMultiplier(defaultMobileNetMultiplier, [0.5, 0.75, 1.0]);
    } else {
      // guiState.input.architecture === "ResNet50"
      updateGuiInputResolution(defaultResNetInputResolution, [
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800
      ]);
      updateGuiOutputStride(defaultResNetStride, [32, 16]);
      updateGuiMultiplier(defaultResNetMultiplier, [1.0]);
    }
    updateGuiQuantBytes(defaultQuantBytes, [1, 2, 4]);
  }

  updateGui();
  input.open();
  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder("Single Pose Detection");
  single.add(guiState.singlePoseDetection, "minPoseConfidence", 0.0, 1.0);
  single.add(guiState.singlePoseDetection, "minPartConfidence", 0.0, 1.0);

  let multi = gui.addFolder("Multi Pose Detection");
  multi
    .add(guiState.multiPoseDetection, "maxPoseDetections")
    .min(1)
    .max(20)
    .step(1);
  multi.add(guiState.multiPoseDetection, "minPoseConfidence", 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, "minPartConfidence", 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi
    .add(guiState.multiPoseDetection, "nmsRadius")
    .min(0.0)
    .max(40.0);
  multi.open();

  let output = gui.addFolder("Output");
  output.add(guiState.output, "showVideo");
  output.add(guiState.output, "showSkeleton");
  output.add(guiState.output, "showPoints");
  output.add(guiState.output, "showBoundingBox");
  output.open();

  architectureController.onChange(function(architecture) {
    // if architecture is ResNet50, then show ResNet50 options
    updateGui();
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case "single-pose":
        multi.close();
        single.open();
        break;
      case "multi-pose":
        single.close();
        multi.open();
        break;
    }
  });
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById("main").appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById("output");
  const ctx = canvas.getContext("2d");

  // since images are being fed from a webcam, we want to feed in the
  // original image and then just flip the keypoints' x coordinates. If instead
  // we flip the image, then correcting left-right keypoint pairs requires a
  // permutation on all the keypoints.
  const flipPoseHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.changeToArchitecture,
        outputStride: guiState.outputStride,
        inputResolution: guiState.inputResolution,
        multiplier: guiState.multiplier
      });
      toggleLoadingUI(false);
      guiState.architecture = guiState.changeToArchitecture;
      guiState.changeToArchitecture = null;
    }

    if (guiState.changeToMultiplier) {
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: guiState.outputStride,
        inputResolution: guiState.inputResolution,
        multiplier: +guiState.changeToMultiplier,
        quantBytes: guiState.quantBytes
      });
      toggleLoadingUI(false);
      guiState.multiplier = +guiState.changeToMultiplier;
      guiState.changeToMultiplier = null;
    }

    if (guiState.changeToOutputStride) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: +guiState.changeToOutputStride,
        inputResolution: guiState.inputResolution,
        multiplier: guiState.multiplier,
        quantBytes: guiState.quantBytes
      });
      toggleLoadingUI(false);
      guiState.outputStride = +guiState.changeToOutputStride;
      guiState.changeToOutputStride = null;
    }

    if (guiState.changeToInputResolution) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: guiState.outputStride,
        inputResolution: +guiState.changeToInputResolution,
        multiplier: guiState.multiplier,
        quantBytes: guiState.quantBytes
      });
      toggleLoadingUI(false);
      guiState.inputResolution = +guiState.changeToInputResolution;
      guiState.changeToInputResolution = null;
    }

    if (guiState.changeToQuantBytes) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();
      toggleLoadingUI(true);
      guiState.net = await posenet.load({
        architecture: guiState.architecture,
        outputStride: guiState.outputStride,
        inputResolution: guiState.inputResolution,
        multiplier: guiState.multiplier,
        quantBytes: guiState.changeToQuantBytes
      });
      toggleLoadingUI(false);
      guiState.quantBytes = guiState.changeToQuantBytes;
      guiState.changeToQuantBytes = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case "single-pose":
        const pose = await guiState.net.estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: "single-person"
        });
        poses = poses.concat(pose);
        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case "multi-pose":
        let all_poses = await guiState.net.estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: "multi-person",
          maxDetections: guiState.multiPoseDetection.maxPoseDetections,
          scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
          nmsRadius: guiState.multiPoseDetection.nmsRadius
        });

        poses = poses.concat(all_poses);
        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores

    if (isNeedToShoot && !isShot) {
      isShot = true;
      console.log(JSON.stringify(poses[0].keypoints));
    }

    let max = 1;
    let posId = 0;

    const posesArray = [TARGET_POS_1, TARGET_POS_2, TARGET_POS_3, TARGET_POS_4];

    posesArray.forEach((vector, idx) => {
      const poseVector1 = vector;
      const poseVector2 = poses[0].keypoints.map(item => [(item.position.x), Math.floor(item.position.y)]).flat();

      let cosineSimilarity = similarity(poseVector1, poseVector2);
      let distance = 2 * (1 - cosineSimilarity);
      const eDistance = Math.sqrt(distance);

      if (eDistance < max) {
        max = eDistance;
        posId = idx;
      }
      // currentPoseTimeDiff = new Date().getTime();
    });

    const poseVector1 = posesArray[posId];
    const poseVector2 = poses[0].keypoints.map(item => [(item.position.x), Math.floor(item.position.y)]).flat();

    let cosineSimilarity = similarity(poseVector1, poseVector2);
    let distance = 2 * (1 - cosineSimilarity);
    const eDistance = Math.sqrt(distance);

    // console.log(eDistance);
    let prevPosIdx = 0;

    if (eDistance < 0.2) {
      if (prevPosIdx !== posId || !isPose1) {
        prevPosIdx = posId;
        isPose1 = true;
        isNone = false;
        console.log("Pose detected ", posId);
        timeStart = new Date().getTime() - 3000;
        timeEnd = new Date().getTime();
        updatePos({
          last_pose: {
            time_start: timeStart,
            time_end: timeEnd,
            pose_id: posId,
          }
        });
      }
    } else {
      if (!isNone) {
        isPose1 = false;
        isNone = true;
        // console.log("None detected");
        // prevPoseTimestamp = new Date().getTime();
        // }
      }
    }

    const {
      keypoints
    } = poses[0];
    const leftHipNode = keypoints.find(point => point.part === 'leftHip');
    const rightHipNode = keypoints.find(point => point.part === 'rightHip');
    const leftShoulderNode = keypoints.find(point => point.part === 'leftShoulder');
    const rightShoulderNode = keypoints.find(point => point.part === 'rightShoulder');
    const leftWristNode = keypoints.find(point => point.part === 'leftWrist');
    const rightWristNode = keypoints.find(point => point.part === 'rightWrist');
    const leftKneeNode = keypoints.find(point => point.part === 'leftKnee');
    const rightKneeNode = keypoints.find(point => point.part === 'rightKnee');
    const leftAnkleNode = keypoints.find(point => point.part === 'leftAnkle');
    const rightAnkleNode = keypoints.find(point => point.part === 'rightAnkle');


    if (poses[0].score > 0.3) {
      // const avgShoulderX = (leftShoulderNode.position.x + rightShoulderNode.position.x) / 2;
      const avgShoulderY = (leftShoulderNode.position.y + rightShoulderNode.position.y) / 2;
      const avgHipY = (leftHipNode.position.y + leftHipNode.position.y) / 2;

      const avgBodyPoint = (avgHipY + avgShoulderY) / 2;
      if (tickCounter >= 0) {
        console.log(tickCounter);

        tickNormalizerContainerY.push(avgBodyPoint);
        tickCounter--;
      } else {
        tickCounter = 60;
        if (Math.max(...tickNormalizerContainerY) - Math.min(...tickNormalizerContainerY) > 100) {
          updatePos({
            last_pose: {
              time_start: new Date().getTime() - 500,
              time_end: new Date().getTime(),
              pose_id: 99,
            }
          });
        }
        tickNormalizerContainerY = [];
      }

      if (leftWristNode.position.y > leftShoulderNode.position.y &&
        rightWristNode.position.y > rightShoulderNode.position.y && !isBothDown) {
        isBothDown = true;
        isBothUp = false;
        isLeftUp = false;
        isLeftDown = false;
        isRightDown = false;
        isRightUp = false;
        console.log('Both down!');
      }

      if (leftWristNode.position.y < leftShoulderNode.position.y &&
        rightWristNode.position.y < rightShoulderNode.position.y && !isBothUp) {
        isBothDown = false;
        isBothUp = true;
        isLeftUp = false;
        isLeftDown = false;
        isRightDown = false;
        isRightUp = false;
        console.log('Both up!');
        updatePos({
          last_pose: {
            time_start: new Date().getTime() - 500,
            time_end: new Date().getTime(),
            pose_id: 999,
          }
        });
      }

      if (leftWristNode.position.y < leftShoulderNode.position.y && !isLeftUp && !isBothUp) {
        isBothDown = false;
        isBothUp = false;
        isLeftUp = true;
        isLeftDown = false;
        isRightDown = false;
        isRightUp = false;
        console.log('Left up!');
      }

      if (leftWristNode.position.y > leftShoulderNode.position.y && !isLeftDown && !isBothDown) {
        isBothDown = false;
        isBothUp = false;
        isLeftUp = false;
        isLeftDown = true;
        isRightDown = false;
        isRightUp = false;
        console.log('Left down!');
      }

      if (rightWristNode.position.y < rightShoulderNode.position.y && !isRightUp && !isBothUp) {
        isBothDown = false;
        isBothUp = false;
        isLeftUp = false;
        isLeftDown = false;
        isRightDown = false;
        isRightUp = true;
        console.log('Right up!');
      }

      if (rightWristNode.position.y > rightShoulderNode.position.y && !isRightDown && !isBothDown) {
        isBothDown = false;
        isBothUp = false;
        isLeftUp = false;
        isLeftDown = false;
        isRightDown = true;
        isRightUp = false;
        console.log('Right down!');
      }
      // const SUPER_COMBO = ['bu', 'ld', 'lu', 'rd', 'ld'];
    }

    // if (poses[0].score > 0.3 &&
    //   leftAnkleNode.score > 0.3 &&
    //   rightAnkleNode.score > 0.3 &&
    //   leftKneeNode.score > 0.3 &&
    //   rightKneeNode.score > 0.3 &&
    //   leftHipNode.score > 0.3 &&
    //   rightHipNode.score > 0.3) {
    //   const leftHip = leftHipNode.position;
    //   const rightHip = rightHipNode.position;
    //   const leftKnee = leftKneeNode.position;
    //   const rightKnee = rightKneeNode.position;
    //   const leftAnkle = leftAnkleNode.position;
    //   const rightAnkle = rightAnkleNode.position;

    //   // Long jump
    //   if (leftKnee.y < leftHip.y && rightKnee.y < rightHip.y) {
    //     console.log("LONG JUMP");
    //   }

    //   // Short jump
    //   if (leftAnkle.y < leftKnee.y && rightAnkle.y < rightKnee.y) {
    //     console.log("SHORT JUMP");
    //   }
    // }


    poses.forEach(({
      score,
      keypoints
    }) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }
    });

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  toggleLoadingUI(true);
  console.log("model loading");
  const net = await posenet.load({
    architecture: guiState.input.architecture,
    outputStride: guiState.input.outputStride,
    inputResolution: guiState.input.inputResolution,
    multiplier: guiState.input.multiplier,
    quantBytes: guiState.input.quantBytes
  });
  toggleLoadingUI(false);
  console.log("model ready");

  let video;

  try {
    console.log("loading the video");
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById("info");
    info.textContent =
      "this browser does not support video capture," +
      "or this device does not have a camera";
    info.style.display = "block";
    throw e;
  }

  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia =
  navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
