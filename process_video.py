"""Conservative microscopy egg counting. Run on Colab with ultralytics + sahi."""
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def circle_roi(frame):
    h,w=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Bright circular field, excluding black bars and watermark.
    bright=(gray>38).astype('uint8')*255
    bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(bright)
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>h*w*.04]
    if not candidates:
        return np.zeros((h,w),np.uint8)
    i=max(candidates,key=lambda k:stats[k,cv2.CC_STAT_AREA])
    mask=np.uint8(labels==i)*255
    # erode boundary to reject half-visible eggs and edge artifacts
    mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
    return mask


def features(frame,mask,orb):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    kp,des=orb.detectAndCompute(gray,mask)
    return kp,des


def registration(kp,des,refs):
    if des is None or len(kp)<14:return None,0,0
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    best=None
    for ref in reversed(refs):
        if ref['des'] is None:continue
        pairs=matcher.knnMatch(des,ref['des'],k=2)
        good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
        if len(good)<14:continue
        src=np.float32([kp[m.queryIdx].pt for m in good]);dst=np.float32([ref['kp'][m.trainIdx].pt for m in good])
        A,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=4000,confidence=.995)
        if A is None:continue
        ni=int(inliers.sum());ratio=ni/len(good)
        scale=float(np.linalg.norm(A[:2,0]));rot=abs(float(math.atan2(A[1,0],A[0,0])))
        if ni<14 or ratio<.48 or not .94<scale<1.06 or rot>.16:continue
        H=np.vstack((A,[0,0,1]));W=ref['world']@H
        if best is None or ni>best[1]:best=(W,ni,ratio)
    return best if best else (None,0,0)


def deduplicate(boxes,ios=.45,center_ratio=.35):
    boxes=sorted(boxes,key=lambda b:b[4],reverse=True)
    kept=[]
    for b in boxes:
        x1,y1,x2,y2,score=b;area=max(1,(x2-x1)*(y2-y1))
        duplicate=False
        for q in kept:
            a=max(1,(q[2]-q[0])*(q[3]-q[1]))
            ix=max(0,min(x2,q[2])-max(x1,q[0]));iy=max(0,min(y2,q[3])-max(y1,q[1]))
            inter=ix*iy
            dist=math.hypot((x1+x2-q[0]-q[2])/2,(y1+y2-q[1]-q[3])/2)
            scale=min(math.sqrt(area),math.sqrt(a))
            if inter/min(area,a)>ios or (dist<scale*center_ratio and max(area/a,a/area)<2.8):
                duplicate=True;break
        if not duplicate:kept.append(b)
    return kept


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--video',required=True)
    p.add_argument('--output',default='detections.json');p.add_argument('--debug-video',default='tracking_debug.mp4')
    p.add_argument('--device',default='cuda:0');p.add_argument('--interval',type=float,default=.1)
    p.add_argument('--slice',type=int,default=640);p.add_argument('--overlap',type=float,default=.2)
    p.add_argument('--confidence',type=float,default=.45);p.add_argument('--min-hits',type=int,default=4)
    p.add_argument('--max-distance',type=float,default=30);p.add_argument('--revisit-distance',type=float,default=22)
    args=p.parse_args()
    det=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=args.model,confidence_threshold=args.confidence,device=args.device)
    cap=cv2.VideoCapture(args.video)
    fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or w<=0 or h<=0:raise RuntimeError('Video could not be opened')
    step=max(1,round(args.interval*fps));outfps=fps/step
    writer=cv2.VideoWriter(args.debug_video,cv2.VideoWriter_fourcc(*'mp4v'),outfps,(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot write debug video')
    orb=cv2.ORB_create(nfeatures=4000,fastThreshold=6)
    refs=[];tracks=[];samples=[];next_id=1;failures=0
    # Every frame is registered to keyframes in a shared coordinate system.
    # On registration failure, no new ID is assigned; this avoids false counting.
    for fi in range(0,total_frames,step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,frame=cap.read()
        if not ok:break
        t=fi/fps;roi=circle_roi(frame);kp,des=features(frame,roi,orb)
        if not refs:
            world=np.eye(3);inliers=0;ratio=1.
            refs.append(dict(kp=kp,des=des,world=world.copy()))
        else:
            world,inliers,ratio=registration(kp,des,refs)
            if world is None:failures+=1
            elif inliers>=22 and fi//step%8==0:
                refs.append(dict(kp=kp,des=des,world=world.copy()))
                if len(refs)>80:refs=[refs[0]]+refs[-79:]
        prediction=get_sliced_prediction(frame,det,slice_height=args.slice,slice_width=args.slice,
            overlap_height_ratio=args.overlap,overlap_width_ratio=args.overlap,
            postprocess_type='GREEDYNMM',postprocess_match_metric='IOS',postprocess_match_threshold=.45,verbose=0)
        raw=[]
        for obj in prediction.object_prediction_list:
            x1,y1,x2,y2=map(float,obj.bbox.to_xyxy())
            cx=int((x1+x2)/2);cy=int((y1+y2)/2)
            if not (0<=cx<w and 0<=cy<h and roi[cy,cx]>0):continue
            if x2-x1<6 or y2-y1<6:continue
            raw.append([x1,y1,x2,y2,float(obj.score.value)])
        boxes=deduplicate(raw)
        shown=[];matched=set()
        if world is not None:
            # One-to-one assignment, globally shortest valid pairs first.
            candidates=[];observations=[]
            for b in boxes:
                center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2,1.])
                pt=(world@center)[:2];sz=np.array([b[2]-b[0],b[3]-b[1]])
                observations.append((pt,sz))
            for i,(pt,sz) in enumerate(observations):
                for j,tr in enumerate(tracks):
                    dist=float(np.linalg.norm(pt-tr['pos']))
                    limit=args.max_distance if tr['last']>=fi-3*step else args.revisit_distance
                    shape=max(sz[0]/max(tr['size'][0],1),tr['size'][0]/max(sz[0],1),sz[1]/max(tr['size'][1],1),tr['size'][1]/max(sz[1],1))
                    if dist<limit and shape<1.65:candidates.append((dist+8*(shape-1),i,j))
            used_obs=set();used_tracks=set();assigned={}
            for _,i,j in sorted(candidates):
                if i not in used_obs and j not in used_tracks:
                    assigned[i]=j;used_obs.add(i);used_tracks.add(j)
            for i,b in enumerate(boxes):
                pt,sz=observations[i]
                if i in assigned:
                    tr=tracks[assigned[i]]
                    tr['pos']=.8*tr['pos']+.2*pt;tr['size']=.8*tr['size']+.2*sz
                    tr['hits']+=1;tr['last']=fi
                else:
                    tr=dict(id=next_id,pos=pt.copy(),size=sz.copy(),hits=1,last=fi)
                    tracks.append(tr);next_id+=1
                confirmed=tr['hits']>=args.min_hits
                shown.append(dict(b=[round(float(x),1) for x in b[:4]],id=int(tr['id']),score=round(float(b[4]),3),confirmed=bool(confirmed),new=bool(confirmed and tr['hits']==args.min_hits)))
        else:
            shown=[dict(b=[round(float(x),1) for x in b[:4]],id=None,score=round(float(b[4]),3),confirmed=False,new=False) for b in boxes]
        unique=sum(tr['hits']>=args.min_hits for tr in tracks)
        samples.append(dict(t=round(t,3),c=len(boxes),total=int(unique),motion_ok=bool(world is not None),inliers=int(inliers),b=[o['b'] for o in shown],objects=shown))
        debug=frame.copy();overlay=debug.copy();overlay[roi==0]=(overlay[roi==0]*.45).astype('uint8');debug=overlay
        for o in shown:
            x1,y1,x2,y2=map(int,o['b']);color=(0,220,0) if o['confirmed'] else (0,180,255)
            if o['id'] is None:color=(80,80,255)
            cv2.rectangle(debug,(x1,y1),(x2,y2),color,2)
            tag='?' if o['id'] is None else str(o['id'])
            if o['new']:tag+=' NEW'
            cv2.putText(debug,'#'+tag,(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        status='OK' if world is not None else 'FAIL / COUNT PAUSED'
        cv2.rectangle(debug,(0,0),(min(w,740),36),(0,0,0),-1)
        cv2.putText(debug,f'{t:.1f}s visible={len(boxes)} unique={unique} motion={status} inliers={inliers}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),2)
        writer.write(debug)
        print(f'{t:6.2f}s visible={len(boxes):3d} unique={unique:3d} motion={status} inliers={inliers}',flush=True)
    cap.release();writer.release()
    payload=dict(width=w,height=h,unique_count_estimate=int(unique),registration_failures=int(failures),samples=samples)
    Path(args.output).write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    print('Saved:',args.output,args.debug_video,'unique estimate:',unique)

if __name__=='__main__':main()
