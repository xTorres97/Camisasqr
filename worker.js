// worker.js - versión debug con logs y thresholds relajados
self.importScripts('https://docs.opencv.org/4.x/opencv.js');

let cvReady = false;
self.Module = self.Module || {};
self.Module.onRuntimeInitialized = () => { cvReady = true; postMessage({type:'ready'}); };

// variables
let procW = 160, procH = 120;
let templGray = null, templKeypoints = null, templDescriptors = null;
let orb = null, bf = null;

// parámetros (relajados para debug)
let MATCH_RATIO = 0.9;
let MAX_GOOD_MATCHES = 800;
let minMatchCount = 6;
let MIN_INLIERS_ABS = 4;
let INLIER_RATIO = 0.28; // reducido para aceptar inliers bajos en grandes sets

function safeDelete(m){ try{ if (m && typeof m.delete === 'function') m.delete(); }catch(e){} }

function postLog(msg){ postMessage({type:'log', msg}); }
function postError(msg){ postMessage({type:'error', msg}); }

async function loadImageToMatGrayscale(url, maxSize=1400){
  postLog('loading template ' + url);
  const resp = await fetch(url, {mode:'cors'});
  if (!resp.ok) throw new Error('HTTP ' + resp.status);
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

async function initTemplate(url){
  try {
    safeDelete(templGray); safeDelete(templKeypoints); safeDelete(templDescriptors);
    const {matGray, w, h} = await loadImageToMatGrayscale(url, 1400);
    templGray = matGray;
    postLog(`template loaded raw size ${w}x${h}`);
    // init ORB and BF with strong params
    try { orb = new cv.ORB(1200, 1.2, 12, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, 20); } catch(e){ orb = new cv.ORB(); postLog('ORB fallback created'); }
    bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
    templKeypoints = new cv.KeyPointVector();
    templDescriptors = new cv.Mat();
    orb.detect(templGray, templKeypoints);
    orb.compute(templGray, templKeypoints, templDescriptors);
    postLog(`templ kp:${templKeypoints.size()} desc rows:${templDescriptors.rows}`);
    postMessage({type:'templateInfo', w, h});
  } catch(e){
    postError('initTemplate fail: ' + e.message);
  }
}

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

function detectAndCompute(grayMat){
  const kps = new cv.KeyPointVector();
  const desc = new cv.Mat();
  try { orb.detect(grayMat, kps); orb.compute(grayMat, kps, desc); }
  catch(e){ try { orb.detectAndCompute(grayMat, new cv.Mat(), kps, desc); } catch(err){ postLog('detectAndCompute fail '+err); } }
  return {kps, desc};
}

function knnGoodMatches(des1, des2){
  const good = []; let matches = new cv.DMatchVectorVector();
  try { bf.knnMatch(des1, des2, matches, 2); }
  catch(e){ postLog('knnMatch error '+e); try{ matches.delete(); }catch(_){} return good; }
  for (let i=0;i<matches.size();i++){
    const mv = matches.get(i);
    if (mv.size() >= 2){
      const m = mv.get(0), n = mv.get(1);
      if (m.distance <= MATCH_RATIO * n.distance) good.push({queryIdx:m.queryIdx, trainIdx:m.trainIdx, distance:m.distance});
    } else if (mv.size() === 1){
      const m = mv.get(0);
      if (m.distance < 40) good.push({queryIdx:m.queryIdx, trainIdx:m.trainIdx, distance:m.distance});
    }
    try{ mv.delete(); }catch(e){}
    if (good.length >= MAX_GOOD_MATCHES) break;
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
  try { H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 5.0, mask); } catch(e){ postLog('findHomography err '+e); }
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
  } catch(e){ postLog('homographyToCorners err '+e); return null; }
}

/* core processing */
async function processBitmap(bitmap){
  if (!cvReady || !templGray || !orb){ try{ bitmap.close(); }catch(e){} postMessage({type:'result', matches:0, inliers:0, corners:null}); return; }
  const gray = bitmapToGrayMat(bitmap, procW, procH);
  const det = detectAndCompute(gray);
  const frameKps = det.kps, frameDesc = det.desc;
  const tplK = templKeypoints ? templKeypoints.size() : 0;
  const frmK = frameKps ? frameKps.size() : 0;
  postMessage({type:'log', msg:`tplK=${tplK} frmK=${frmK}`});

  if (!tplK || !frmK){ safeDelete(frameKps); safeDelete(frameDesc); safeDelete(gray); postMessage({type:'result', matches:0, inliers:0, corners:null}); return; }

  const goodMatches = knnGoodMatches(frameDesc, templDescriptors);
  postMessage({type:'log', msg:`goodMatchesLen=${goodMatches.length}`});
  if (goodMatches.length < minMatchCount){ safeDelete(frameKps); safeDelete(frameDesc); safeDelete(gray); postMessage({type:'result', matches:goodMatches.length, inliers:0, corners:null, debug:`too few goodMatches (${goodMatches.length})`}); return; }

  const {H, mask, inliers} = computeHomographyFromMatches(goodMatches, frameKps, templKeypoints);
  postMessage({type:'log', msg:`homography inliers=${inliers}`});
  const required = Math.max(MIN_INLIERS_ABS, Math.floor(goodMatches.length * INLIER_RATIO));
  postMessage({type:'log', msg:`requiredInliers=${required}`});
  if (H && !H.empty() && inliers >= required){
    const corners = homographyToCorners(H);
    postMessage({type:'result', matches:goodMatches.length, inliers, corners, debug:`ok`});
  } else {
    postMessage({type:'result', matches:goodMatches.length, inliers:inliers||0, corners:null, debug:`homography rejected`});
  }

  safeDelete(frameKps); safeDelete(frameDesc); safeDelete(gray);
}

/* messages */
self.onmessage = async (ev) => {
  const d = ev.data;
  if (d.type === 'init'){
    if (!cvReady){ let wait=0; while(!cvReady && wait<8000){ await new Promise(r=>setTimeout(r,100)); wait+=100; } if (!cvReady){ postError('opencv not ready'); return; } }
    procW = d.procW || procW; procH = d.procH || procH;
    await initTemplate(d.targetUrl);
    postLog('init complete');
  } else if (d.type === 'resize'){
    procW = d.procW; procH = d.procH;
    postLog(`worker resize to ${procW}x${procH}`);
  } else if (d.type === 'frame'){
    const bitmap = d.bitmap || ev.data.bitmap;
    if (!bitmap){ postLog('no bitmap in frame'); return; }
    try { await processBitmap(bitmap); } catch(e){ postError('processBitmap err '+e.message); }
  }
};
