#!/usr/bin/env python3
"""Native, fixed-camera medium fire evolution with shared dense timelines."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gc
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capture_scene_fire_gallery import (
    create_sim, observe, make_suite, gray_haze, sha256, write_json, TYPES,
)
from scripts.capture_fire_type_gallery import FIRE_TYPE_LABELS

OUTPUT = ROOT/'outputs/fire_temporal_evolution_medium_20260912'
WIDTH, HEIGHT, FPS = 800, 600, 5
TIMES = np.arange(0., 602., 2.)
HOLD = 2 * FPS
VERSION = 1
REFERENCE = ROOT/'outputs/fire_gallery_medium_36_stronger_20260912'
SCENES = ('4ok3usBNeis','6s7QHgap2fW','BAbdmeyTvMZ')


def prepare(output,reference):
    records=[r for r in json.loads((reference/'inputs.json').read_text())['scenes'] if r['scene_id'] in SCENES]
    if len(records)!=3:raise ValueError('Missing one of the three selected scenes')
    for record in records:
        source=json.loads((reference/record['scene_id']/'manifest.json').read_text())
        if any(v['semantic_fallback'] for v in source['views']):
            raise ValueError('All selected template views must have genuine semantic sources')
        record.update(views=source['views'],reference_timeline=str((reference/source['timeline']).resolve()),reference_gallery=str(reference))
        dest=output/'scenes'/record['scene_id'];dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(record['plan_path'],dest/'plan.json')
    write_json(output/'inputs.json',dict(scenes=records,expected_videos=12,types=list(TYPES),
        save_dt_s=2.,duration_s=600.,note='Each multi-origin video synchronizes four views of one shared merged medium plan.'))
    return records


def dense_bake(args):
    record, output = args
    from utils.fire_world.propagation import run_propagation
    dest = Path(output)/'timelines'/record['scene_id']/record['plan_id']
    contract = dict(plan_sha256=record['plan_sha256'], inventory_sha256=record['inventory_sha256'],
                    solver_sha256=sha256(ROOT/'utils/fire_world/propagation.py'),
                    voxel_m=.15, dt=.5, save_dt=2., duration_s=600.)
    marker = dest/'provenance.json'
    if marker.exists() and json.loads(marker.read_text()) == contract:
        return record['scene_id'], 'cached'
    plan = json.loads(Path(record['plan_path']).read_text())
    inventory = json.loads(Path(record['inventory_path']).read_text())
    run_propagation(inventory, plan, voxel_m=.15, dt=.5, save_dt=2., out_dir=dest)
    with np.load(dest/'timeline.npz',allow_pickle=False) as dense, \
         np.load(record['reference_timeline'],allow_pickle=False) as previous:
        np.testing.assert_array_equal(dense['times'],TIMES)
        np.testing.assert_array_equal(dense['times'][::5],previous['times'])
        for key in ('flame','smoke','temp'):
            np.testing.assert_array_equal(dense[key][::5],previous[key])
    write_json(marker,contract)
    print(f"[bake] {record['scene_id']}: 301 frames; all 61 reference states identical",flush=True)
    return record['scene_id'], 'complete'


def frame_path(output, sid, slot, index):
    return output/'scenes'/sid/f'view_{slot:02d}'/f'frame_{index:04d}.jpg'


def save_image(path, image):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,94]):
        raise IOError(f'Failed to write {path}')


def render(record, output, pilot=False):
    from utils.fire_world.runtime import FireWorld
    from utils.fire_world.scene import FireScene, FireClock
    import torch
    sid = record['scene_id']
    dest = output/'scenes'/sid
    marker = dest/('pilot.json' if pilot else 'render_manifest.json')
    if not pilot and marker.exists() and json.loads(marker.read_text()).get('complete'):
        return
    fw = FireWorld.load(sid,record['plan_id'],out_root=output/'timelines')
    np.testing.assert_array_equal(fw.times,TIMES)
    scene = FireScene(fw=fw,clock=FireClock(mode='step'))
    suite = make_suite(scene,WIDTH,HEIGHT,True,117)
    suite.cfg.voxel.n_steps = 64
    # Keep native scene texture while sampling smooth voxel effects at half
    # resolution, a standard video preset; this does not subsample time.
    suite.cfg.voxel.render_scale = .5
    sim = create_sim(record,WIDTH,HEIGHT)
    rows, cameras = [], []
    write_json(output/'sensor_config.json',asdict(suite.cfg))
    started = time.monotonic()
    try:
        for view in record['views']:
            obs,state = observe(sim,np.asarray(view['agent_position']),view['selected_source'])
            cameras.append((view,obs,state))
            save_image(dest/f"view_{view['slot']:02d}"/'clean.png',cv2.cvtColor(obs['rgb'],cv2.COLOR_RGB2BGR))
            pose = state.sensor_states['depth']
            np.testing.assert_allclose(pose.position,view['camera_position'],atol=1e-6)
            q=pose.rotation
            np.testing.assert_allclose([q.x,q.y,q.z,q.w],view['camera_rotation_xyzw'],atol=1e-6)
        indices = [0,1,5,15,30,150,300] if pilot else range(len(TIMES))
        for index in indices:
            t = float(TIMES[index])
            for view,obs,state in cameras:
                # Video requests only RGB/thermal; use the same native voxel
                # sensor without spending time generating unused radar plots.
                rendered = suite.voxel_sensor.process(obs['rgb'],obs['depth'],agent_state=state,t_sim_s=t)
                if rendered['render_backend'] != 'torch':
                    raise RuntimeError('Native CUDA renderer required for this video job')
                display,_ = gray_haze(rendered['image'],obs['depth'],rendered['transmittance'],rendered['flame_mask'])
                save_image(frame_path(output,sid,view['slot'],index),cv2.cvtColor(display,cv2.COLOR_RGB2BGR))
                rows.append(dict(index=int(index),t_sim_s=t,slot=view['slot'],
                    mean_smoke_attenuation=float(np.mean(1-rendered['transmittance'])),
                    visible_flame_fraction=float(np.mean(rendered['flame_mask']>.03)),
                    rgb_sha256=sha256(frame_path(output,sid,view['slot'],index))))
            if index%10==0 or pilot:
                elapsed=time.monotonic()-started
                write_json(output/'render_progress.json',dict(scene=sid,t_sim_s=t,completed_frames=4*(int(index)+1),
                    seconds_elapsed=elapsed,pilot=pilot))
                print(f'[render] {sid} t={t:5.0f}s | {elapsed:.1f}s elapsed',flush=True)
        write_json(marker,dict(complete=not pilot,version=VERSION,scene_id=sid,plan_id=record['plan_id'],
            plan_sha256=record['plan_sha256'],resolution=[WIDTH,HEIGHT],ray_samples=64,render_scale=.5,
            timeline=str((output/'timelines'/sid/record['plan_id']/'timeline.npz').relative_to(output)),
            views=record['views'],frames=rows,elapsed_s=time.monotonic()-started))
    finally:
        sim.close()
        cache=getattr(scene,'_fire_torch_volume_cache',None)
        if cache is not None:cache.clear()
        del suite,sim,scene,fw
        gc.collect();torch.cuda.empty_cache()


def read_frame(path):
    image=cv2.imread(str(path))
    if image is None:raise IOError(f'Missing/unreadable image: {path}')
    if image.shape != (HEIGHT,WIDTH,3):raise ValueError(f'Wrong frame dimensions: {path}')
    return image


def video_frame(output,record,kind,index=None):
    multi=kind=='multi_origin'
    views=record['views'] if multi else [next(v for v in record['views'] if v['requested_type']==kind)]
    tiles=[]
    for view in views:
        p=(output/'scenes'/record['scene_id']/f"view_{view['slot']:02d}"/'clean.png'
           if index is None else frame_path(output,record['scene_id'],view['slot'],index))
        rgb=read_frame(p)
        if multi:
            tile=np.full((HEIGHT+28,WIDTH,3),24,np.uint8);tile[28:]=rgb
            region=view['selected_source'].get('region_id')
            region_label=str(region) if region is not None else 'unassigned'
            label=f"Origin {view['slot']}: {view['selected_source']['category']} | region {region_label}"
            cv2.putText(tile,label,(12,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,230,230),1,cv2.LINE_AA)
            tiles.append(tile)
        else:tiles.append(rgb)
    body=np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]) if multi else tiles[0]
    h,w=body.shape[:2]
    canvas=np.full((h+100,w,3),20,np.uint8);canvas[76:76+h]=body
    cv2.putText(canvas,'Temporal Evolution of Fire Scenarios',(16,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),1,cv2.LINE_AA)
    subtitle=f"{FIRE_TYPE_LABELS[kind]} | {record['scene_id']} | medium"
    cv2.putText(canvas,subtitle,(16,53),cv2.FONT_HERSHEY_SIMPLEX,.52,(190,205,225),1,cv2.LINE_AA)
    label='Before ignition | clean reference' if index is None else f'Simulation time: {TIMES[index]:06.1f} / 600 s | {2*FPS:g}x speed | {FPS:g} FPS'
    cv2.putText(canvas,label,(16,h+93),cv2.FONT_HERSHEY_SIMPLEX,.49,(235,235,235),1,cv2.LINE_AA)
    if index is not None:
        cv2.line(canvas,(0,72),(int((w-1)*TIMES[index]/600),72),(70,150,255),3)
    return canvas


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(record,output,kinds=TYPES):
    sid=record['scene_id']
    render_manifest=json.loads((output/'scenes'/sid/'render_manifest.json').read_text())
    if not render_manifest['complete'] or len(render_manifest['frames'])!=4*len(TIMES):
        raise ValueError(f'Incomplete render: {sid}')
    for kind in kinds:
        dest=output/kind/sid;dest.mkdir(parents=True,exist_ok=True)
        clean=video_frame(output,record,kind)
        h,w=clean.shape[:2]
        video=dest/'temporal_evolution.mp4'
        partial=dest/'temporal_evolution.partial.mp4'
        command=[ffmpeg(),'-y','-loglevel','error','-f','rawvideo','-pixel_format','bgr24',
                 '-video_size',f'{w}x{h}','-framerate',str(FPS),'-i','pipe:0','-an',
                 '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for _ in range(HOLD):process.stdin.write(clean.tobytes())
            for index in range(len(TIMES)):
                frame=video_frame(output,record,kind,index)
                process.stdin.write(frame.tobytes())
            for _ in range(HOLD):process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError(f'ffmpeg encode failed: {video}')
        except BaseException:
            process.kill();process.wait();raise
        partial.replace(video)
        samples=[video_frame(output,record,kind,i) for i in (0,5,15,30,90,300)]
        thumbs=[cv2.resize(im,(400,round(im.shape[0]*400/im.shape[1]))) for im in samples]
        save_image(dest/'temporal_stages.jpg',np.vstack([np.hstack(thumbs[:3]),np.hstack(thumbs[3:])]))
        save_image(dest/'poster.jpg',samples[2])
        write_json(dest/'manifest.json',dict(scene_id=sid,template=kind,plan_id=record['plan_id'],
            plan_sha256=record['plan_sha256'],frame_count=len(TIMES)+2*HOLD,fps=FPS,
            duration_s=(len(TIMES)+2*HOLD)/FPS,resolution=[w,h],simulation_times_s=TIMES.tolist(),
            clean_reference_hold_s=HOLD/FPS,final_state_hold_s=HOLD/FPS,speedup=2*FPS,
            selected_slots=[1,2,3,4] if kind=='multi_origin' else [TYPES.index(kind)+1],
            shared_timeline=str((output/'timelines'/sid/record['plan_id']/'timeline.npz').relative_to(output)),
            video_sha256=sha256(video),encoder_command=command,
            disclosure='Views of one merged medium plan; multi-origin video synchronizes four source cameras. Pre-ignition clean reference is not a simulated negative-time frame. No interpolated fire states.'))
        print(f'[encode] {kind}/{sid}: {len(TIMES)+2*HOLD} frames, {(len(TIMES)+2*HOLD)/FPS:.1f}s',flush=True)


def validate_and_index(output,records):
    failures=[];cards=[];results=[]
    for record in records:
        sid=record['scene_id']
        render_meta=json.loads((output/'scenes'/sid/'render_manifest.json').read_text())
        for slot in (1,2,3,4):
            frames=[r for r in render_meta['frames'] if r['slot']==slot]
            if [f['index'] for f in frames]!=list(range(len(TIMES))) or [f['t_sim_s'] for f in frames]!=TIMES.tolist():
                failures.append(f'{sid}/{slot}: incomplete native time sampling')
            if len({f['rgb_sha256'] for f in frames})<100:
                failures.append(f'{sid}/{slot}: raw scene evolution is unexpectedly static')
            for f in frames:
                if sha256(frame_path(output,sid,slot,f['index']))!=f['rgb_sha256']:
                    failures.append(f'{sid}/{slot}: raw frame hash changed')
        if sha256(output/'scenes'/sid/'plan.json')!=record['plan_sha256']:
            failures.append(f'{sid}: copied plan changed')
    for kind in TYPES:
        section=[]
        for record in records:
            sid=record['scene_id'];dest=output/kind/sid;video=dest/'temporal_evolution.mp4'
            m=json.loads((dest/'manifest.json').read_text())
            cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);count=0
            previous=None;changes=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                if list(frame.shape[1::-1])!=m['resolution']:failures.append(f'{kind}/{sid}: wrong resolution')
                if previous is not None and not np.array_equal(frame,previous):changes+=1
                previous=frame;count+=1
            cap.release()
            if count!=m['frame_count'] or abs(fps-m['fps'])>.001:failures.append(f'{kind}/{sid}: count/FPS mismatch')
            if fps<=0 or abs(count/max(fps,1e-9)-m['duration_s'])>.001:
                failures.append(f'{kind}/{sid}: duration mismatch')
            if changes<100:failures.append(f'{kind}/{sid}: evolution is unexpectedly static')
            subprocess.run([ffmpeg(),'-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
            results.append(dict(template=kind,scene=sid,decoded_frames=count,fps=fps,duration_s=count/fps,changed_frames=changes))
            link=f'{kind}/{sid}'
            section.append(f'<article><h3>{sid}</h3><video controls preload="metadata" poster="{link}/poster.jpg" src="{link}/temporal_evolution.mp4"></video><p><a href="{link}/temporal_evolution.mp4">Download MP4</a> · <a href="{link}/temporal_stages.jpg">Evolution stages</a> · <a href="{link}/manifest.json">Plan, camera slots and timing</a></p></article>')
        cards.append(f'<section><h2>{html.escape(FIRE_TYPE_LABELS[kind])}</h2>'+''.join(section)+'</section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Temporal Evolution of Fire Scenarios</title><style>body{background:#181b20;color:#eee;font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}a{color:#9acbff}video{width:100%;max-height:850px}article{margin:40px 0}section{margin-top:60px}</style><h1>Temporal Evolution of Fire Scenarios</h1><p>Four templates, three examples each. Actual medium FireWorld evolution from ignition through 600 s. Playback speed is labelled in each video. Each scene shares one plan and dense timeline. Multi-origin videos show four synchronized views. Each clip includes a clearly labelled clean reference and a final-state hold.</p>'+''.join(cards))
    source_paths=['scripts/capture_fire_temporal_evolution.py','scripts/capture_scene_fire_gallery.py',
                  'scripts/capture_fire_type_gallery.py','utils/fire_world/propagation.py',
                  'utils/fire_world/runtime.py','utils/fire_sensors/sensors/voxel_smoke.py',
                  'utils/fire_sensors/voxel_render.py','utils/fire_sensors/voxel_render_torch.py']
    write_json(output/'source_hashes.json',{p:sha256(ROOT/p) for p in source_paths})
    report=dict(passed=not failures,expected_videos=12,videos=len(results),failures=failures,results=results)
    write_json(output/'validation.json',report)
    if failures:raise RuntimeError(str(failures))
    print(json.dumps(report,indent=2),flush=True)


def main():
    global FPS, HOLD
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=OUTPUT)
    p.add_argument('--stage',choices=['prepare','bake','pilot','render','encode','validate','all'],default='all')
    p.add_argument('--reference',type=Path,default=REFERENCE)
    p.add_argument('--scene',action='append')
    p.add_argument('--template',action='append',choices=TYPES,help='Restrict encoding to selected templates')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--fps',type=int,default=5,help='Output frame rate; all 301 simulation frames are retained')
    p.add_argument('--hold-frames',type=int,help='Clean and final hold frames each; defaults to two seconds each')
    args=p.parse_args();output=args.output_dir.resolve()
    if args.fps<=0 or (args.hold_frames is not None and args.hold_frames<0):
        p.error('FPS must be positive and hold frames must be nonnegative')
    FPS=args.fps
    HOLD=2*FPS if args.hold_frames is None else args.hold_frames
    records=(prepare(output,args.reference.resolve()) if args.stage in ('prepare','all')
             else json.loads((output/'inputs.json').read_text())['scenes'])
    selected=[r for r in records if not args.scene or r['scene_id'] in args.scene]
    if not selected:raise ValueError('No selected scenes')
    if args.stage in ('bake','all'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(dense_bake,[(r,str(output)) for r in selected]):print(result,flush=True)
    if args.stage in ('pilot','render','all'):
        for record in selected:render(record,output,pilot=args.stage=='pilot')
    if args.stage in ('encode','all'):
        for record in selected:encode(record,output,args.template or TYPES)
    if args.stage in ('validate','all'):validate_and_index(output,records)


if __name__=='__main__':main()
