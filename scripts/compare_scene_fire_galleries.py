#!/usr/bin/env python3
"""Audit and display matched native gallery revisions without changing images."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np


def brightened_flame_fraction(clean, rendered, flame_mask):
    """Orange pixels brightened above clean RGB, rejecting warm furniture.

    This is an image visibility diagnostic, not physical burning area.
    Use pre-presentation RGB so the gray haze cannot increase this count.
    """
    clean = clean.astype(np.float32)
    rgb = rendered.astype(np.float32)
    red, green, blue = rgb[...,0], rgb[...,1], rgb[...,2]
    return float(np.mean((flame_mask > .03) & (red > clean[...,0]+10)
                        & (red > 120) & (red > green+15) & (green > blue+5)))


def compare(reference, revised, completed_only=False):
    records = json.loads((revised / 'inputs.json').read_text())['scenes']
    expected_scenes = len(records)
    if completed_only:
        records = [r for r in records if (revised/r['scene_id']/'manifest.json').exists()]
    if not records:
        raise ValueError('No completed scenes to compare')
    rows, failures, weaker_fire, weaker_brightened, pages = [], [], [], [], []
    for record in records:
        sid = record['scene_id']
        old = json.loads((reference/sid/'manifest.json').read_text())
        new = json.loads((revised/sid/'manifest.json').read_text())
        if old['plan_sha256'] != new['plan_sha256']:
            failures.append(f'{sid}: plan changed')
        if not (reference/old['timeline']).samefile(revised/new['timeline']):
            failures.append(f'{sid}: timeline is not shared')
        old_views = {v['slot']:v for v in old['views']}
        cards = []
        for b in new['views']:
            a = old_views[b['slot']]
            key = f"{sid}/{b['slot']}"
            for field in ('selected_source', 'agent_position', 'camera_position',
                          'camera_rotation_xyzw', 'fire_time_s', 'resolution'):
                if a[field] != b[field]:
                    failures.append(f'{key}: {field} changed')
            with np.load(reference/sid/a['directory']/'fire_sensor_arrays.npz') as az, \
                 np.load(revised/sid/b['directory']/'fire_sensor_arrays.npz') as bz:
                for field in ('rgb', 'depth_clean'):
                    if not np.array_equal(az[field], bz[field]):
                        failures.append(f'{key}: clean sensor {field} changed')
                old_attenuation = float(np.mean(1-az['transmittance']))
                new_attenuation = float(np.mean(1-bz['transmittance']))
                old_rgb = cv2.cvtColor(cv2.imread(str(reference/sid/a['directory']/'rgb_fire_smoke_physical.png')),cv2.COLOR_BGR2RGB)
                new_rgb = cv2.cvtColor(cv2.imread(str(revised/sid/b['directory']/'rgb_fire_smoke_physical.png')),cv2.COLOR_BGR2RGB)
                old_brightened = brightened_flame_fraction(az['rgb'],old_rgb,az['thermal_flame_mask'])
                new_brightened = brightened_flame_fraction(bz['rgb'],new_rgb,bz['thermal_flame_mask'])
            old_fire = a['physical_visibility_metrics']['rgb_fire_fraction']
            new_fire = b['physical_visibility_metrics']['rgb_fire_fraction']
            row = dict(scene=sid, slot=b['slot'], old_fire_fraction=old_fire,
                       new_fire_fraction=new_fire, old_attenuation=old_attenuation,
                       new_attenuation=new_attenuation,
                       old_brightened_flame_fraction=old_brightened,
                       new_brightened_flame_fraction=new_brightened)
            rows.append(row)
            if new_attenuation <= old_attenuation:
                failures.append(f'{key}: smoke attenuation did not increase')
            if new_fire <= old_fire:
                weaker_fire.append(key)
            if new_brightened <= old_brightened:
                weaker_brightened.append(key)
            before = os.path.relpath(reference/sid/a['directory']/'template_scene.png', revised)
            after = f"{sid}/{b['directory']}/template_scene.png"
            cards.append(f'<h3>{b["slot"]}. {html.escape(b["label"])}</h3>'
                         f'<div class="pair"><figure><figcaption>Before</figcaption><img loading="lazy" src="{before}"></figure>'
                         f'<figure><figcaption>Stronger fire and smoke</figcaption><img loading="lazy" src="{after}"></figure></div>')
        pages.append(f'<section><h2><a href="{sid}/index.html">{sid}</a></h2>'+''.join(cards)+'</section>')
    report = dict(scenes=len(records), expected_scenes=expected_scenes,
                  complete=len(records)==expected_scenes, views=len(rows), failures=failures,
                  passed=not failures, weaker_fire_views=weaker_fire,
                  weaker_brightened_flame_views=weaker_brightened,
                  metric_note='Orange pixel fractions can include warm furniture; brightened_flame_fraction additionally requires red radiance to exceed clean RGB by 10/255. Both are visibility diagnostics, not physical burning area.',
                  mean_old_fire_fraction=float(np.mean([r['old_fire_fraction'] for r in rows])),
                  mean_new_fire_fraction=float(np.mean([r['new_fire_fraction'] for r in rows])),
                  mean_old_attenuation=float(np.mean([r['old_attenuation'] for r in rows])),
                  mean_new_attenuation=float(np.mean([r['new_attenuation'] for r in rows])),
                  mean_old_brightened_flame_fraction=float(np.mean([r['old_brightened_flame_fraction'] for r in rows])),
                  mean_new_brightened_flame_fraction=float(np.mean([r['new_brightened_flame_fraction'] for r in rows])),
                  matched_views=rows)
    stem = 'comparison_partial' if completed_only else 'comparison'
    (revised/f'{stem}.json').write_text(json.dumps(report,indent=2)+'\n')
    (revised/f'{stem}.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched fire and smoke comparison</title>'
        '<style>body{background:#181b20;color:#eee;font:16px system-ui;margin:30px}a{color:#9acbff}.pair{display:flex;gap:16px}figure{margin:0;width:50%}img{width:100%}section{margin:60px 0}</style>'
        '<h1>Matched fire and smoke comparison</h1><p>Same sources, camera poses, simulation times and shared medium timelines. Optical rendering revision only.</p>'
        '<a href="index.html">Scene gallery</a>'+''.join(pages))
    if not completed_only:
        index = revised/'index.html'
        page = index.read_text()
        marker = '<h1>Medium multi-origin fire gallery</h1>'
        banner = '<h1>Medium multi-origin fire gallery · stronger fire and smoke</h1><p><a href="comparison.html">Compare every view with the original</a> · <a href="comparison.json">Matched validation report</a></p>'
        index.write_text(page.replace(marker,banner))
    source_paths = json.loads((reference/'source_hashes.json').read_text())['sources']
    source_paths[__file__] = ''
    sources = {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}
    (revised/'source_hashes.json').write_text(json.dumps({'sources':sources},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='matched_views'},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--revised',type=Path,required=True)
    parser.add_argument('--completed-only',action='store_true',help='Write a clearly marked partial report while rendering')
    args = parser.parse_args()
    raise SystemExit(0 if compare(args.reference.resolve(),args.revised.resolve(),args.completed_only)['passed'] else 1)
