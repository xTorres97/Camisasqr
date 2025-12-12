// worker.js (ajustes: INLIER_RATIO reducido para aceptar matches grandes con menos % de inliers)
self.importScripts('https://docs.opencv.org/4.x/opencv.js');

let cvReady = false;
let orb=null, bf=null;
let templGray=null, templKeypoints=null, templDescriptors=null;
let prevGray=null, prevPts=null, templPts=null;
let procW=160, procH=120;
let MODE='detection';
let lastTrackTime = 0;

// parámetros ajustados
let MATCH_RATIO = 0.85;
let MAX_GOOD_MATCHES = 400;
let minMatchCount = 12;
let MIN_INLIERS_ABS = 8;
let INLIER_RATIO = 0.30; // <<--- BAJADO para aceptar 46/146

self.Module = self.Module || {};
self.Module.onRuntimeInitialized = () => { cvReady = true; postMessage({type:'ready'}); };

function safeDelete(m){ try{ if (m && typeof m.delete === 'function') m.delete(); }catch(e){} }

function bitmapToGrayMat(bitmap, w=procW, h=procH){
  const oc = new OffscreenCanvas(w,h);
  const c = oc.getContext('2d');
  c.drawImage(bitmap, 0, 0, w, h);
  const id = c.getImageData(0,0,w,h);
  const matRGBA = cv.matFromImageData(id);
  const gray = new cv.Mat();
  cv.cvtColor(matRGBA, gray, cv.COLOR_RGBA2GRAY);
  matRGBA.delete();
  try{ bitmap.close(); }catch(e){}
  return gray;
}

async function loadImageToMatGrayscale(url, maxSize=1400){
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('HTTP '+resp.status);
  const blob = await resp.blob();
  const bmp = await createImageBitmap(blob);
  let tw = bmp.width, th = bmp.height;
  if (Math.max(tw,th) > maxSize){
    const s = maxSize / Math.max(tw,th); tw = Math.round(tw*s); th = Math.round(th*s);
  }
  const oc = new OffscreenCanvas(tw, th);
  const c = oc.getContext('2d');
  c.drawImage(bmp, 0, 0, tw, th);
  const id = c.getImageData(0,0,tw,th);
  const matRGBA = cv.matFromImageData(id);
  const gray = new cv.Mat();
  cv.cvtColor(matRGBA, gray, cv.COLOR_RGBA2GRAY);
  matRGBA.delete();
  try{ bmp.close(); }catch(e){}
  return {matGray: gray, w: tw, h: th};
}

function detectAndCompute(grayMat){
  const kps = new cv.KeyPointVector();
  const desc = new cv.Mat();
  try { orb.detect(grayMat, kps); orb.compute(grayMat, kps, desc); }
  catch(e){ try { orb.detectAndCompute(grayMat, new cv.Mat(), kps, desc); } catch(err){ postMessage({type:'log', msg:'detectAndCompute fail '+err}); } }
  return {kps, desc};
}

function knnGoodMatches(des1, des2, ratio=MATCH_RATIO, maxMatches=MAX_GOOD_MATCHES){
  const good = []; let matches = new cv.DMatchVectorVector();
  try { bf.knnMatch(des1, des2, matches, 2); } catch(e){ postMessage({type:'log', msg:'knnMatch error '+e}); try{ matches.delete(); }catch(_){} return good; }
  for (let i=0;i<matches.size();i++){
    const mv = matches.get(i);
    if (mv.size() >= 2){
      const m = mv.get(0), n = mv.get(1);
      if (typeof m.distance !== 'undefined' && typeof n.distance !== 'undefined'){
        if (m.distance <= ratio * n.distance) good.push({queryIdx:m.queryIdx, trainIdx:m.trainIdx, distance:m.distance});
      }
    } else if (mv.size() === 1){
      const m = mv.get(0);
      if (typeof m.distance !== 'undefined' && m.distance < 40) good.push({queryIdx:m.queryIdx, trainIdx:m.trainIdx, distance:m.distance});
    }
    try{ mv.delete(); }catch(e){}
    if (good.length >= maxMatches) break;
  }
  try{ matches.delete(); }catch(e){}
  return good;
}

function computeHomographyFromMatches(goodMatches, frameKps, templKps){
  const src = [], dst = [];
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
  try { H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 5.0, mask); } catch(e){ postMessage({type:'log', msg:'findHomography err '+e}); }
  let inliers = 0;
  if (mask && !mask.isDeleted()){
    for (let i=0;i<mask.rows;i++) if (mask.ucharPtr(i,0)[0]) inliers++;
  }
  try{ srcMat.delete(); dstMat.delete(); }catch(e){}
  return {H, mask, inliers};
}

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
    for (let i=0;i<4;i++) out.push(dstCorners.floatAt(i,0), dstCorners.floatAt(i,1));
    tplCorners.delete(); dstCorners.delete(); H_inv.delete();
    return out;
  } catch(e){ postMessage({type:'log', msg:'homographyToCorners err '+e}); return null; }
}

async function initTemplate(url){
  safeDelete(templGray); safeDelete(templKeypoints); safeDelete(templDescriptors);
  const {matGray, w, h} = await loadImageToMatGrayscale(url, 1400);
  templGray = matGray;
  try { orb = new cv.ORB(1200, 1.2, 12, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, 20); } catch(e){ orb = new cv.ORB(); }
  bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
  templKeypoints = new cv.KeyPointVector();
  templDescriptors = new cv.Mat();
  try { orb.detect(templGray, templKeypoints); orb.compute(templGray, templKeypoints, templDescriptors); }
  catch(e){ try { orb.detectAndCompute(templGray, new cv.Mat(), templKeypoints, templDescriptors); } catch(err){ postMessage({type:'error', msg:'ORB detect fail '+err}); } }
  postMessage({type:'log', msg:`Template loaded KP:${templKeypoints.size()} desc:${templDescriptors.rows}`});
  postMessage({type:'templateInfo', w, h});
}

async function processFrameBitmap(bitmap){
  if (!cvReady || !templGray || !orb){ try{ bitmap.close(); }catch(e){} postMessage({type:'result', matches:0, inliers:0, corners:null}); return; }
  const grayMat = bitmapToGrayMat(bitmap, procW, procH);

  // tracking
  if (MODE === 'tracking' && prevGray && prevPts && templPts){
    try {
      const nextPts = new cv.Mat(), status = new cv.Mat(), err = new cv.Mat();
      cv.calcOpticalFlowPyrLK(prevGray, grayMat, prevPts, nextPts, status, err, new cv.Size(21,21), 3);
      const goodNext=[], goodTempl=[];
      for (let i=0;i<status.rows;i++){
        if (status.ucharPtr(i,0)[0] === 1){
          const nx = nextPts.floatAt(i,0), ny = nextPts.floatAt(i,1);
          const tx = templPts.floatAt(i,0), ty = templPts.floatAt(i,1);
          if (!isFinite(nx) || !isFinite(ny)) continue;
          goodNext.push(nx,ny); goodTempl.push(tx,ty);
        }
      }
      nextPts.delete(); status.delete(); err.delete();
      if (goodNext.length/2 < 8){
        MODE = 'detection'; safeDelete(prevPts); safeDelete(prevGray); safeDelete(templPts); prevPts=null; prevGray=null; templPts=null;
        postMessage({type:'log', msg:'tracking lost -> detection'});
      } else {
        const n = goodNext.length/2;
        const nextMat = cv.matFromArray(n,1,cv.CV_32FC2, goodNext);
        const templSub = cv.matFromArray(n,1,cv.CV_32FC2, goodTempl);
        const mask = new cv.Mat(); let H=null;
        try { H = cv.findHomography(nextMat, templSub, cv.RANSAC, 5.0, mask); } catch(e){ postMessage({type:'log', msg:'findHomography(flow) '+e}); }
        let inliers = 0; for (let i=0;i<mask.rows;i++) if (mask.ucharPtr(i,0)[0]) inliers++;
        if (H && !H.empty() && inliers >= Math.max(MIN_INLIERS_ABS, Math.floor(n*INLIER_RATIO))){
          const corners = homographyToCorners(H);
          const goodNextFiltered = [], goodTemplFiltered = [];
          for (let i=0;i<mask.rows;i++){
            if (mask.ucharPtr(i,0)[0]){ goodNextFiltered.push(goodNext[2*i], goodNext[2*i+1]); goodTemplFiltered.push(goodTempl[2*i], goodTempl[2*i+1]); }
          }
          safeDelete(prevPts); safeDelete(prevGray);
          prevPts = cv.matFromArray(goodNextFiltered.length/2,1,cv.CV_32FC2, goodNextFiltered);
          templPts = cv.matFromArray(goodTemplFiltered.length/2,1,cv.CV_32FC2, goodTemplFiltered);
          prevGray = grayMat.clone();
          lastTrackTime = performance.now();
          postMessage({type:'result', matches:n, inliers, corners});
          try{ nextMat.delete(); templSub.delete(); mask.delete(); if (H && !H.isDeleted) H.delete(); }catch(e){}
          return;
        } else {
          try{ nextMat.delete(); templSub.delete(); mask.delete(); if (H && !H.isDeleted) H.delete(); }catch(e){}
        }
      }
    } catch(err){
      postMessage({type:'log', msg:'optical flow err '+err});
      MODE='detection'; safeDelete(prevPts); safeDelete(prevGray); safeDelete(templPts); prevPts=null; prevGray=null; templPts=null;
    }
  }

  // detection
  try {
    const det = detectAndCompute(grayMat);
    const frameKps = det.kps, frameDesc = det.desc;
    const tplK = templKeypoints ? templKeypoints.size() : 0;
    const frmK = frameKps ? frameKps.size() : 0;
    if (!tplK || !frmK){ safeDelete(frameKps); safeDelete(frameDesc); safeDelete(grayMat); postMessage({type:'result', matches:0, inliers:0, corners:null}); return; }

    const goodMatches = knnGoodMatches(frameDesc, templDescriptors, MATCH_RATIO, MAX_GOOD_MATCHES);
    if (goodMatches.length < minMatchCount){ safeDelete(frameKps); safeDelete(frameDesc); safeDelete(grayMat); postMessage({type:'result', matches:goodMatches.length, inliers:0, corners:null}); return; }

    const {H, mask, inliers} = computeHomographyFromMatches(goodMatches, frameKps, templKeypoints);

    // Accept if absolute inliers or inlier ratio threshold (reduced ratio)
    const requiredInliers = Math.max(MIN_INLIERS_ABS, Math.floor(goodMatches.length * INLIER_RATIO));
    if (H && !H.empty() && inliers >= requiredInliers){
      const corners = homographyToCorners(H);
      const framePtsArr = [], templPtsArr = [];
      for (let idx=0; idx<goodMatches.length; idx++){
        if (mask.ucharPtr(idx,0)[0]){
          const gm = goodMatches[idx]; const q = gm.queryIdx, t = gm.trainIdx;
          const kpf = frameKps.get(q); const kpt = templKeypoints.get(t);
          framePtsArr.push(kpf.pt.x, kpf.pt.y); templPtsArr.push(kpt.pt.x, kpt.pt.y);
        }
      }
      safeDelete(prevPts); safeDelete(prevGray); safeDelete(templPts);
      if (framePtsArr.length/2 >= 8){
        prevPts = cv.matFromArray(framePtsArr.length/2,1,cv.CV_32FC2, framePtsArr);
        templPts = cv.matFromArray(templPtsArr.length/2,1,cv.CV_32FC2, templPtsArr);
        prevGray = grayMat.clone(); MODE='tracking'; lastTrackTime = performance.now();
      } else {
        MODE='detection';
      }
      postMessage({type:'result', matches:goodMatches.length, inliers, corners});
    } else {
      postMessage({type:'result', matches:goodMatches.length, inliers:inliers||0, corners:null});
    }

    safeDelete(frameKps); safeDelete(frameDesc);
    if (!(MODE === 'tracking' && prevGray)) safeDelete(grayMat);
    return;
  } catch(e){
    postMessage({type:'log', msg:'detection error '+e});
    safeDelete(grayMat);
    postMessage({type:'result', matches:0, inliers:0, corners:null});
    return;
  }
}

self.onmessage = async (ev) => {
  const d = ev.data;
  if (d.type === 'init'){
    if (!cvReady){ let waited=0; while(!cvReady && waited<8000){ await new Promise(r=>setTimeout(r,100)); waited+=100; } if (!cvReady){ postMessage({type:'error', msg:'OpenCV runtime not initialized'}); return; } }
    procW = d.procW || procW; procH = d.procH || procH;
    try { await initTemplate(d.targetUrl); postMessage({type:'log', msg:'template initialized'}); } catch(e){ postMessage({type:'error', msg:'initTemplate failed '+e}); }
  } else if (d.type === 'resize'){ procW = d.procW; procH = d.procH; postMessage({type:'log', msg:`worker resized to ${procW}x${procH}`}); }
  else if (d.type === 'frame'){
    const bitmap = d.bitmap || ev.data.bitmap;
    if (!bitmap){ postMessage({type:'log', msg:'no bitmap'}); return; }
    await processFrameBitmap(bitmap);
  }
};
