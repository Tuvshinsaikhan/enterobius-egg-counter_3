#!/usr/bin/env python3
"""Research prototype: camera-stabilized SAHI detection and conservative egg counting.
Not a clinically validated count. Fails closed when registration is unreliable.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def gray_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)


class Registration:
    def __init__(self, min_inliers=12, min_ratio=.30):
        self.orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reference = None
        self.world = np.eye(3, dtype=np.float64)
        self.min_inliers, self.min_ratio = min_inliers, min_ratio

    def update(self, frame):
        gray = gray_image(frame)
        kp, desc = self.orb.detectAndCompute(gray, None)
        if self.reference is None:
            self.reference = (kp, desc)
            return self.world.copy(), True, 0, 1.0
        prev_kp, prev_desc = self.reference
        if desc is None or prev_desc is None or len(desc) < 12 or len(prev_desc) < 12:
            return self.world.copy(), False, 0, 0.0
        pairs = self.matcher.knnMatch(desc, prev_desc, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .76 * b.distance]
        if len(good) < self.min_inliers:
            return self.world.copy(), False, len(good), 0.0
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([prev_kp[m.trainIdx].pt for m in good])
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=4, maxIters=4000, confidence=.995)
        if affine is None or inliers is None:
            return self.world.copy(), False, 0, 0.0
        n = int(inliers.sum())
        ratio = float(n / len(good))
        scale = float(np.linalg.norm(affine[:2, 0]))
        ok = bool(n >= self.min_inliers and ratio >= self.min_ratio and .90 <= scale <= 1.10)
        if ok:
            m = np.eye(3, dtype=np.float64)
            m[:2] = affine
            self.world = self.world @ m
            # Do not replace reference on failure: prevents corrupting the world map.
            self.reference = (kp, desc)
        return self.world.copy(), ok, n, ratio


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1)*max(0., y2-y1)
    aa = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return inter/max(aa+bb-inter, 1e-8)


def suppress_duplicates(dets, overlap=.25, center_factor=.45):
    """Conservative suppression of highly overlapping, same-object SAHI boxes."""
    selected = []
    for d in sorted(dets, key=lambda x: x['score'], reverse=True):
        cx, cy = d['center']
        short = min(d['size'])
        duplicate = False
        for old in selected:
            dist = np.linalg.norm(np.asarray([cx, cy])-old['center'])
            size_ratio = max(d['size'][0]*d['size'][1], old['size'][0]*old['size'][1]) / max(1., min(d['size'][0]*d['size'][1], old['size'][0]*old['size'][1]))
            if iou(d['box'], old['box']) >= overlap or (dist <= center_factor * min(short, min(old['size'])) and size_ratio <= 2.5):
                duplicate = True
                break
        if not duplicate:
            selected.append(d)
    return selected


def descriptor(gray, box):
    h, w = gray.shape
    x1, y1, x2, y2 = map(int, box)
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    if x2-x1 < 6 or y2-y1 < 6:
        return None
    patch = cv2.resize(gray[y1:y2, x1:x2], (24, 24)).astype(np.float32)
    patch = (patch - patch.mean())/(patch.std()+1e-4)
    return patch.ravel()


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))


class Tracker:
    def __init__(self, radius=42, min_hits=3, max_gap=100, revisit_radius=28):
        self.radius, self.min_hits, self.max_gap = radius, min_hits, max_gap
        self.revisit_radius = revisit_radius
        self.tracks = {}
        self.next_id = 1
        self.counted = set()

    def update(self, dets, index, motion_ok):
        if not motion_ok:
            return [{**d, 'id': None} for d in dets]
        eligible = [t for t in self.tracks.values() if t['confirmed'] or index-t['last'] <= self.max_gap]
        cost = np.full((len(eligible), len(dets)), 1e6, dtype=np.float64)
        for i, t in enumerate(eligible):
            for j, d in enumerate(dets):
                distance = float(np.linalg.norm(t['world']-d['world']))
                size_ratio = max(t['area'], d['area']) / max(1., min(t['area'], d['area']))
                # Old confirmed tracks remain eligible for revisits; use a stricter radius.
                limit = self.revisit_radius if index-t['last'] > self.max_gap else self.radius
                if distance <= limit and size_ratio <= 2.2:
                    sim = cosine(t['appearance'], d['appearance'])
                    if sim >= -.1:
                        cost[i,j] = distance + 7*abs(np.log(size_ratio)) + 8*(1-sim)
        matches = {}
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r,c] < 1e5:
                    matches[int(c)] = eligible[int(r)]
        result = []
        for j, d in enumerate(dets):
            if j in matches:
                t = matches[j]
                t['world'] = .75*t['world'] + .25*d['world']
                t['area'] = .75*t['area'] + .25*d['area']
                t['appearance'] = d['appearance']
                t['hits'] += 1
                t['last'] = index
            else:
                t = dict(id=self.next_id, world=d['world'].copy(), area=d['area'],
                         appearance=d['appearance'], hits=1, last=index, confirmed=False)
                self.tracks[self.next_id] = t
                self.next_id += 1
            if t['hits'] >= self.min_hits:
                t['confirmed'] = True
                self.counted.add(t['id'])
            result.append({**d, 'id': t['id']})
        for tid, t in list(self.tracks.items()):
            if not t['confirmed'] and index-t['last'] > self.max_gap:
                del self.tracks[tid]
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='detections.json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--interval', type=float, default=.10)
    p.add_argument('--slice', type=int, default=640)
    p.add_argument('--overlap', type=float, default=.20)
    p.add_argument('--confidence', type=float, default=.40)
    p.add_argument('--max-distance', type=float, default=42)
    p.add_argument('--revisit-distance', type=float, default=28)
    p.add_argument('--max-gap', type=int, default=100)
    p.add_argument('--min-hits', type=int, default=3)
    p.add_argument('--dedup-iou', type=float, default=.25)
    a = p.parse_args()
    if a.interval <= 0 or a.slice < 64 or not 0 <= a.overlap < 1:
        p.error('Invalid interval, slice or overlap')
    for name in ('model','video'):
        if not Path(getattr(a,name)).is_file():
            p.error(f'{name} file not found: {getattr(a,name)}')
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError('Invalid video FPS')
    width, height = int(cap.get(3)), int(cap.get(4))
    model = AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path=a.model,
        confidence_threshold=a.confidence, device=a.device)
    registration = Registration()
    tracker = Tracker(a.max_distance, a.min_hits, a.max_gap, a.revisit_distance)
    samples, failures = [], 0
    idx, next_t = 0, 0.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx/fps
        idx += 1
        if t + 1e-8 < next_t:
            continue
        next_t += a.interval
        matrix, motion_ok, inliers, ratio = registration.update(frame)
        failures += int(not motion_ok)
        pred = get_sliced_prediction(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), model,
            slice_height=a.slice, slice_width=a.slice,
            overlap_height_ratio=a.overlap, overlap_width_ratio=a.overlap,
            postprocess_type='GREEDYNMM', postprocess_match_metric='IOS',
            postprocess_match_threshold=.5, verbose=0)
        gray = gray_image(frame)
        dets = []
        for item in pred.object_prediction_list:
            b = item.bbox
            box = [float(b.minx),float(b.miny),float(b.maxx),float(b.maxy)]
            center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
            size = (box[2]-box[0], box[3]-box[1])
            dets.append(dict(box=box, center=center, size=size,
                score=float(item.score.value)))
        dets = suppress_duplicates(dets, overlap=a.dedup_iou)
        for d in dets:
            d['world'] = (matrix @ np.r_[d['center'],1])[:2]
            d['area'] = max(1.,d['size'][0]*d['size'][1])
            d['appearance'] = descriptor(gray,d['box'])
        tracked = tracker.update(dets,len(samples),motion_ok)
        samples.append(dict(t=round(t,3), c=len(tracked), total=len(tracker.counted),
            motion_ok=bool(motion_ok), inliers=int(inliers), inlier_ratio=round(float(ratio),3),
            b=[[round(float(x),2) for x in d['box']] for d in tracked],
            scores=[round(float(d['score']),4) for d in tracked],
            ids=[d['id'] for d in tracked]))
        print(f'{t:7.2f}s visible={len(tracked):3d} unique={len(tracker.counted):4d} motion={"OK" if motion_ok else "FAIL"}',flush=True)
    cap.release()
    payload = dict(width=width,height=height,fps=fps,sample_interval=a.interval,
        method='YOLO26s + SAHI + dedup + ORB/RANSAC + conservative world-coordinate tracking',
        unique_count_estimate=len(tracker.counted),registration_failures=failures,samples=samples)
    out = Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(f'Saved {out}; frames={len(samples)}; estimated unique={len(tracker.counted)}; registration failures={failures}')

if __name__ == '__main__':
    main()
