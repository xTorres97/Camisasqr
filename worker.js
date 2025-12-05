// worker.js
// Worker que carga OpenCV.js y realiza detección ORB + matching + optical flow
// Recibe mensajes: {type:'init', targetUrl, procW, procH} y {type:'frame', bitmap}, {type:'resize'}
// Devuelve: postMessage({type:'ready'}), postMessage({type:'result', matches, inliers, corners:[...]}), postMessage({type:'log', msg})

self.importScripts('https://docs.opencv.org/4.x/opencv.js');

let cvReady = false;
let orb = null, bf = null;
let templMat = null, templGray = null, templKeypoints = null, templDescriptors = null;
let prevGray = null, prevPts = null, templPts = null;
let procW = 160, procH = 120;
let MODE = 'detection'; // 'detection' | 'tracking'
let REDETECT_AFTER_MS = 2500;
let lastTrackTime = 0;
let MATCH_RATIO = 0.9;
let MAX_GOOD_MATCHES = 150;
let minMatchCount = 6;

self.cv = self.cv || {}; // guard
self.postMessage({type:'log', msg:'worker loaded, waiting cv...'});

self.cv = self.cv || {};

self.Module = self.Module || {};
self.Module.onRuntimeInitialized = () => {
  cvReady = true;
  postMessage({type:'ready'});
};

// Helper: load image from URL into cv.Mat (in worker)
async function loadImageToMat(url, maxSize=800){
  const resp = await fetch(url);
  const blob = await resp.blob();
  const imgBitmap = await createImageBitmap(blob);
  // draw into offscreen canvas
  const oc = new OffscreenCanvas(imgBitmap.width, imgBitmap.height);
  const cx = oc.getContext('2d');
  cx.drawImage(imgBitmap, 0, 0);
  let tw = imgBitmap.width, th = imgBitmap.height;
  if (Math.max(tw, th) > maxSize){
    const s = maxSize / Math.max(tw, th);
    tw = Math.round(tw * s); th = Math.round(th * s);
    const oc2 = new OffscreenCanvas(tw, th);
    const c2 = oc2.getContext('2d');
    c2.drawImage(imgBitmap, 0, 0, tw, th);
    const mat = cv.imread(oc2);
    imgBitmap.close();
    return mat;
  } else {
    const mat = cv.imread(oc);
    imgBitmap.close();
    return mat;
  }
}

function safeDelete(m){
  try{ if (m && typeof m.delete === 'function') m.delete(); }catch(e){}
}

async function initTemplate(url){
  safeDelete(templMat); safeDelete(templGray); safeDelete(templKeypoints); safeDelete(templDescriptors);
  templMat = await loadImageToMat(url, 800);
  templGray = new cv.Mat();
  cv.cvtColor(templMat, templGray, cv.COLOR_RGBA2GRAY);

  // try CLAHE (if available)
  try {
    let clahe = new cv.CLAHE(2.0, new cv.Size(8,8));
    let tmp = new cv.Mat();
    clahe.apply(templGray, tmp);
    templGray.delete(); templGray = tmp;
    clahe.delete();
    postMessage({type:'log', msg:'CLAHE applied to template'});
  } catch(e){ /* ignore */ }

  // init ORB and BF
  try {
    orb = new cv.ORB(600, 1.2, 8, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, 20);
  } catch(e){
    orb = new cv.ORB();
  }
  bf = new cv.BFMatcher(cv.NORM_HAMMING, false);

  templKeypoints = new cv.KeyPointVector();
  templDescriptors = new cv.Mat();
  try {
    orb.detect(templGray, templKeypoints);
    orb.compute(templGray, templKeypoints, templDescriptors);
  } catch(e){
    try { orb.detectAndCompute(templGray, new cv.Mat(), templKeypoints, templDescriptors); } catch(err){ postMessage({type:'error', msg:'ORB detect failed on template: '+err}); return; }
  }
  postMessage({type:'log', msg:`Template loaded KP:${templKeypoints.size()} desc:${templDescriptors.rows}`});
}

// Convert ImageBitmap -> cv.Mat (grayscale) using OffscreenCanvas
function bitmapToGrayMat(bitmap){
  const oc = new OffscreenCanvas(procW, procH);
  const c = oc.getContext('2d');
  c.drawImage(bitmap, 0, 0, procW, procH);
  // transfer bitmap closed by main
  const mat = cv.imread(oc);
  const g = new cv.Mat();
  cv.cvtColor(mat, g, cv.COLOR_RGBA2GRAY);
  mat.delete();
  return g; // caller must delete
}

// Do ORB detect+compute on a gray mat (returns keypoints vector and descriptors mat)
function detectAndCompute(grayMat){
  let kps = new cv.KeyPointVector();
  let desc = new cv.Mat();
  try {
    orb.detect(grayMat, kps);
    orb.compute(grayMat, kps, desc);
  } catch(e){
    try { orb.detectAndCompute(grayMat, new cv.Mat(), kps, desc); } catch(err){ postMessage({type:'log', msg:'detectAndCompute failed '+err}); }
  }
  return {kps, desc};
}

// knn matching + ratio test -> returns array of {queryIdx, trainIdx, distance}
function knnGoodMatches(des1, des2, ratio=0.9, maxMatches=150){
  const good = [];
  let matches = new cv.DMatchVectorVector();
  try { bf.knnMatch(des1, des2, matches, 2); }
  catch(e){ postMessage({type:'log', msg:'knnMatch error: '+e}); try{ matches.delete(); }catch(e){} return good; }
  for (let i=0;i<matches.size();i++){
    const mv = matches.get(i);
    if (mv.size() >= 2){
      const m = mv.get(0), n = mv.get(1);
      if (typeof m.distance !== 'undefined' && typeof n.distance !== 'undefined'){
        if (m.distance <= ratio * n.distance){
          good.push({queryIdx: m.queryIdx, trainIdx: m.trainIdx, distance: m.distance});
        }
      }
      // no m.delete() safe call (build dependent)
    } else if (mv.size() === 1){
      const m = mv.get(0);
      if (typeof m.distance !== 'undefined' && m.distance < 40) good.push({queryIdx: m.queryIdx, trainIdx: m.trainIdx, distance: m.distance});
    }
    try{ mv.delete(); }catch(e){}
    if (good.length >= maxMatches) break;
  }
  try{ matches.delete(); }catch(e){}
  return good;
}

// Build homography from good matches given kpsFrame and templKeypoints
function computeHomographyFromMatches(goodMatches, frameKps, templKps){
  const src = []; const dst = [];
  for (let i=0;i<goodMatches.length;i++){
    const gm = goodMatches[i];
    const q = gm.queryIdx, t = gm.trainIdx;
    const kpf = frameKps.get(q);
    const kpt = templKps.get(t);
    src.push(kpf.pt.x, kpf.pt.y);
    dst.push(kpt.pt.x, kpt.pt.y);
  }
  const srcMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, src);
  const dstMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, dst);
  const mask = new cv.Mat();
  let H = null;
  try {
    H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 5.0, mask);
  } catch(e){ postMessage({type:'log', msg:'findHomography failed '+e}); }
  // count inliers
  let inliers = 0;
  if (mask && !mask.isDeleted()){
    for (let i=0;i<mask.rows;i++) if (mask.ucharPtr(i,0)[0]) inliers++;
  }
  // cleanup
  try{ srcMat.delete(); dstMat.delete(); }catch(e){}
  return {H, mask, inliers};
}

// Given H (frame->templ) produce corners in frame-coords (proc-space)
// We'll invert H to map template -> frame, consistent with main earlier
function homographyToCorners(H){
  if (!H) return null;
  try {
    const H_inv = new cv.Mat();
    cv.invert(H, H_inv, cv.DECOMP_LU);
    const tw = templGray.cols, th = templGray.rows;
    const tplCorners = cv.matFromArray(4,1,cv.CV_32FC2, [0,0, tw,0, tw,th, 0,th]);
    const dstCorners = new cv.Mat();
    cv.perspectiveTransform(tplCorners, dstCorners, H_inv);
    const out = [];
    for (let i=0;i<4;i++){
      out.push(dstCorners.floatAt(i,0), dstCorners.floatAt(i,1));
    }
    tplCorners.delete(); dstCorners.delete(); H_inv.delete();
    return out;
  } catch(e){
    postMessage({type:'log', msg:'homographyToCorners err '+e});
    return null;
  }
}

// === main frame processing function ===
// Input: ImageBitmap sent from main, already sized to procW x procH
async function processFrameBitmap(bitmap){
  if (!cvReady || !templGray || !orb) {
    safeDelete(bitmap);
    postMessage({type:'result', matches:0, inliers:0, corners:null});
    return;
  }

  // convert bitmap -> gray mat
  const grayMat = bitmapToGrayMat(bitmap);
  try{ bitmap.close(); }catch(e){}

  if (MODE === 'tracking' && prevGray && prevPts && templPts){
    // try optical flow
    try {
      const nextPts = new cv.Mat();
      const status = new cv.Mat();
      const err = new cv.Mat();
      cv.calcOpticalFlowPyrLK(prevGray, grayMat, prevPts, nextPts, status, err, new cv.Size(21,21), 3);

      // collect good points
      const goodNext = []; const goodTempl = [];
      for (let i=0;i<status.rows;i++){
        if (status.ucharPtr(i,0)[0] === 1){
          const nx = nextPts.floatAt(i,0); const ny = nextPts.floatAt(i,1);
          const tx = templPts.floatAt(i,0); const ty = templPts.floatAt(i,1);
          if (!isFinite(nx) || !isFinite(ny)) continue;
          goodNext.push(nx, ny); goodTempl.push(tx, ty);
        }
      }

      nextPts.delete(); status.delete(); err.delete();

      if (goodNext.length/2 < 6){
        // tracking lost, fallback to detection
        MODE = 'detection';
        safeDelete(prevGray); safeDelete(prevPts); safeDelete(templPts);
        prevGray = null; prevPts = null; templPts = null;
        postMessage({type:'log', msg:'tracking lost -> detection'});
        // continue to detection below (so we don't return early)
      } else {
        // compute homography from next->templ
        const nextMat = cv.matFromArray(goodNext.length/2, 1, cv.CV_32FC2, goodNext);
        const templSub = cv.matFromArray(goodTempl.length/2, 1, cv.CV_32FC2, goodTempl);
        const mask = new cv.Mat();
        let H = null;
        try { H = cv.findHomography(nextMat, templSub, cv.RANSAC, 5.0, mask); } catch(e){ postMessage({type:'log', msg:'findHomography flow err '+e}); }
        let inliers = 0; if (mask){ for (let i=0;i<mask.rows;i++) if (mask.ucharPtr(i,0)[0]) inliers++; }
        let corners = null;
        if (H && !H.empty() && inliers >= Math.max(4, Math.floor((goodNext.length/2)*0.25))){
          corners = homographyToCorners(H);
          postMessage({type:'result', matches: goodNext.length/2, inliers, corners});
          // update prevPts with inlier filtered points (build new prevPts & templPts)
          const goodNextFiltered = []; const goodTemplFiltered = [];
          for (let i=0;i<mask.rows;i++){
            if (mask.ucharPtr(i,0)[0]){
              goodNextFiltered.push(goodNext[2*i], goodNext[2*i+1]);
              goodTemplFiltered.push(goodTempl[2*i], goodTempl[2*i+1]);
            }
          }
          safeDelete(prevPts); safeDelete(prevGray);
          prevPts = cv.matFromArray(goodNextFiltered.length/2, 1, cv.CV_32FC2, goodNextFiltered);
          templPts = cv.matFromArray(goodTemplFiltered.length/2, 1, cv.CV_32FC2, goodTemplFiltered);
          prevGray = grayMat.clone();
          lastTrackTime = performance.now();
          // cleanup
          try{ nextMat.delete(); templSub.delete(); mask.delete(); if (H && !H.isDeleted) H.delete(); }catch(e){}
          return;
        } else {
          // fallback to detection path (below) - cleanup temporals
          try{ nextMat.delete(); templSub.delete(); mask.delete(); if (H && !H.isDeleted) H.delete(); }catch(e){}
          // continue to detection
        }
      }
    } catch(err){
      postMessage({type:'log', msg:'optical flow exception: '+err});
      MODE = 'detection';
      safeDelete(prevGray); safeDelete(prevPts); safeDelete(templPts);
      prevGray = null; prevPts = null; templPts = null;
    }
  }

  // DETECTION path (ORB + matching)
  try {
    const det = detectAndCompute(grayMat);
    const frameKps = det.kps; const frameDesc = det.desc;
    const tplK = templKeypoints ? templKeypoints.size() : 0;
    const frmK = frameKps ? frameKps.size() : 0;

    if (!tplK || !frmK){
      // cleanup
      safeDelete(frameKps); safeDelete(frameDesc); safeDelete(grayMat);
      postMessage({type:'result', matches:0, inliers:0, corners:null});
      return;
    }

    const goodMatches = knnGoodMatches(frameDesc, templDescriptors, MATCH_RATIO, MAX_GOOD_MATCHES);
    if (goodMatches.length < minMatchCount){
      safeDelete(frameKps); safeDelete(frameDesc); safeDelete(grayMat);
      postMessage({type:'result', matches:goodMatches.length, inliers:0, corners:null});
      return;
    }

    // compute homography
    const {H, mask, inliers} = computeHomographyFromMatches(goodMatches, frameKps, templKeypoints);
    let corners = null;
    if (H && !H.empty() && inliers >= Math.max(4, Math.floor(goodMatches.length * 0.2))){
      corners = homographyToCorners(H);

      // prepare tracking mats: prevPts (frame points) and templPts (template points) only using inliers
      const goodFramePts = []; const goodTemplPts = [];
      for (let i=0;i<mask.rows;i++){
        if (mask.ucharPtr(i,0)[0]){
          // goodMatches[i] corresponds to match; but mask corresponds to order in matFromArray earlier:
          // we built src/dst arrays in same order as goodMatches, so index aligns
          goodFramePts.push(frameKps.get(i).pt.x, frameKps.get(i).pt.y);
          // BUT frameKps.get(i) is wrong — mapping must use queryIdx/trainIdx indices:
          // Rebuild properly:
        }
      }
      // Above approach risks mismatch; instead rebuild arrays from goodMatches & mask:
      const framePtsArr = [];
      const templPtsArr = [];
      for (let idx=0; idx<goodMatches.length; idx++){
        if (mask.ucharPtr(idx,0)[0]){
          const gm = goodMatches[idx];
          const q = gm.queryIdx, t = gm.trainIdx;
          const kpf = frameKps.get(q);
          const kpt = templKeypoints.get(t);
          framePtsArr.push(kpf.pt.x, kpf.pt.y);
          templPtsArr.push(kpt.pt.x, kpt.pt.y);
        }
      }

      // set tracking mats
      safeDelete(prevPts); safeDelete(prevGray); safeDelete(templPts);
      if (framePtsArr.length/2 >= 6){
        prevPts = cv.matFromArray(framePtsArr.length/2, 1, cv.CV_32FC2, framePtsArr);
        templPts = cv.matFromArray(templPtsArr.length/2, 1, cv.CV_32FC2, templPtsArr);
        prevGray = grayMat.clone();
        MODE = 'tracking';
        lastTrackTime = performance.now();
      } else {
        // not enough points to start reliable tracking: remain in detection
        MODE = 'detection';
      }

      postMessage({type:'result', matches:goodMatches.length, inliers, corners});
    } else {
      postMessage({type:'result', matches:goodMatches.length, inliers:inliers||0, corners:null});
    }

    // cleanup
    safeDelete(frameKps); safeDelete(frameDesc);
    safeDelete(H); try{ if (mask) mask.delete(); }catch(e){}
    // keep grayMat if tracking started (we cloned into prevGray), else delete
    if (!(MODE==='tracking' && prevGray)) { safeDelete(grayMat); }
    return;
  } catch(e){
    postMessage({type:'log', msg:'detection error: '+e});
    safeDelete(grayMat);
    postMessage({type:'result', matches:0, inliers:0, corners:null});
    return;
  }
}

// message handler
self.onmessage = async (ev) => {
  const d = ev.data;
  if (d.type === 'init'){
    // wait cvReady
    if (!cvReady){
      postMessage({type:'log', msg:'waiting cv runtime...'});
      // busy-wait small loop (up to a few seconds)
      let waited = 0;
      while(!cvReady && waited < 8000){ await new Promise(r=>setTimeout(r,100)); waited+=100; }
      if (!cvReady){ postMessage({type:'error', msg:'OpenCV runtime did not initialize'}); return; }
    }
    procW = d.procW || procW; procH = d.procH || procH;
    await initTemplate(d.targetUrl);
    postMessage({type:'log', msg:'template initialized in worker'});
    return;
  } else if (d.type === 'resize'){
    procW = d.procW; procH = d.procH;
    postMessage({type:'log', msg:`worker resized to ${procW}x${procH}`});
    return;
  } else if (d.type === 'frame'){
    // receive ImageBitmap (transfered ownership)
    const bitmap = d.bitmap || ev.data.bitmap;
    if (!bitmap){
      postMessage({type:'log', msg:'no bitmap in message'});
      return;
    }
    await processFrameBitmap(bitmap);
  }
};
