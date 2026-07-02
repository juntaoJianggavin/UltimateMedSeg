#!/usr/bin/env python3
"""Run smoke tests for all APRIL-MedSeg training paradigms.

For each YAML in configs/training_paradigms, we:
1. Load the YAML and patch data paths / hyperparameters for fast CPU testing
2. Generate architecture-matching checkpoints for source-free DA and distillation
3. Run 1-epoch smoke test with the correct entry script
4. Report PASS/FAIL/SKIP
"""
import os
import sys
import yaml
import tempfile
import subprocess
import time
import json
import traceback
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = '/root/nas/nas_9c2/jjt/conda_env/ultimedseg/bin/python'
os.chdir(BASE_DIR)

# Make project importable
sys.path.insert(0, BASE_DIR)
import torch
from medseg.model_builder import build_model

results = {'pass': [], 'fail': [], 'skip': []}


def run_cmd(cmd, timeout=600):
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=BASE_DIR, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
        )
        if proc.returncode < 0:
            return False, f"KILLED by signal {-proc.returncode} (likely OOM)\n" + (proc.stdout + proc.stderr)[-500:]
        return proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def make_patched_yaml(cfg, suffix='test'):
    fd, path = tempfile.mkstemp(suffix=f'_{suffix}.yaml', dir=os.path.join(BASE_DIR, 'output_test'))
    os.close(fd)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


def extract_error(output):
    if not output:
        return 'No output'
    lines = output.strip().split('\n')
    for l in reversed(lines):
        if any(k in l for k in ['Error', 'Exception', 'Traceback', 'RuntimeError']):
            return l.strip()[:200]
    return lines[-1].strip()[:200]


def create_matching_checkpoint(model_cfg, ckpt_path):
    """Build model from cfg and save its state dict as a matching checkpoint."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    try:
        model = build_model(model_cfg)
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
        return True
    except Exception as e:
        return False, str(e)


def patch_common_training(cfg):
    if 'training' not in cfg:
        cfg['training'] = {}
    cfg['training']['epochs'] = 1
    cfg['training']['batch_size'] = 2
    cfg['training']['num_workers'] = 0
    cfg['training']['val_interval'] = 1
    cfg['training']['save_interval'] = 999
    if 'model' in cfg and isinstance(cfg['model'], dict):
        if 'encoder' in cfg['model'] and isinstance(cfg['model']['encoder'], dict):
            cfg['model']['encoder']['pretrained'] = False
        if 'decoder' in cfg['model'] and isinstance(cfg['model']['decoder'], dict):
            dec = cfg['model']['decoder']
            if 'params' in dec and isinstance(dec['params'], dict):
                dec['params'].pop('pretrained', None)
    return cfg


# ═══════════════════════════════════════════════════════════
# Domain Adaptation
# ═══════════════════════════════════════════════════════════
def test_domain_adaptation():
    print("\n" + "="*70)
    print("  DOMAIN ADAPTATION")
    print("="*70)

    da_dir = os.path.join(BASE_DIR, 'configs', 'training_paradigms', 'domain_adaptation')
    source_free_methods = {
        'tent', 'dpl', 'class_balanced_mt', 'uncertainty_self_training',
        'dual_reference', 'shot_loss', 'adamss_loss', 'sf_tta_loss', 'fpl_plus_loss',
        'crst',
    }
    yamls = sorted([f for f in os.listdir(da_dir) if f.endswith('.yaml')])

    for yf in yamls:
        name = yf.replace('.yaml', '')
        yaml_path = os.path.join(da_dir, yf)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        cfg = patch_common_training(cfg)
        cfg['data']['type'] = 'image_mask'

        da_cfg = cfg.get('domain_adaptation', {})
        da_method = da_cfg.get('method', '')
        is_source_free = da_cfg.get('source_free', da_method in source_free_methods)

        if is_source_free:
            # Create architecture-specific source checkpoint
            ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'source_model_{name}.pth')
            ok = create_matching_checkpoint(cfg.get('model', {}), ckpt_path)
            if not ok:
                results['skip'].append({'cat': 'DA', 'name': name, 'error': 'Failed to build source checkpoint'})
                print(f"\n  [DA] {name} ... SKIP (checkpoint build failed)")
                continue
            cfg['data']['pretrained_model'] = ckpt_path
        else:
            if 'source' not in cfg.get('data', {}):
                cfg.setdefault('data', {})['source'] = {
                    'image_dir': './data/source/images',
                    'mask_dir': './data/source/masks',
                }

        if 'target' not in cfg.get('data', {}):
            cfg.setdefault('data', {})['target'] = {'image_dir': './data/target/images'}
        if 'val' not in cfg.get('data', {}):
            cfg.setdefault('data', {})['val'] = {
                'image_dir': './data/target_val/images',
                'mask_dir': './data/target_val/masks',
            }

        patched_path = make_patched_yaml(cfg, f'da_{name}')
        out_dir = os.path.join(BASE_DIR, 'output_test', f'da_{name}')
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  [DA] {name} ... ", end='', flush=True)
        t0 = time.time()
        success, output = run_cmd([
            PYTHON, 'train_domain_adaptation.py',
            '--config', patched_path,
            '--output_dir', out_dir,
            '--device', 'cpu',
        ], timeout=600)
        elapsed = time.time() - t0

        status = 'PASS' if success else 'FAIL'
        err = extract_error(output) if not success else ''
        results[status.lower()].append({'cat': 'DA', 'name': name, 'error': err})
        print(f"{status} ({elapsed:.1f}s)" + (f" — {err}" if err else ''))

        try:
            os.unlink(patched_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════
# Semi-Supervised
# ═══════════════════════════════════════════════════════════
def test_semi_supervised():
    print("\n" + "="*70)
    print("  SEMI-SUPERVISED")
    print("="*70)

    semi_dir = os.path.join(BASE_DIR, 'configs', 'training_paradigms', 'semi_supervision')
    yamls = sorted([f for f in os.listdir(semi_dir) if f.endswith('.yaml')])

    for yf in yamls:
        name = yf.replace('.yaml', '')
        yaml_path = os.path.join(semi_dir, yf)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        cfg = patch_common_training(cfg)
        cfg['data']['type'] = 'image_mask'
        cfg['training']['labeled_batch_size'] = 2
        cfg['training']['unlabeled_batch_size'] = 2

        # AllSpark's all-pairs cross-attention is O(B²·N²) — reduce img_size to avoid OOM on CPU
        if name == 'allspark':
            cfg['data']['img_size'] = 96
            if 'model' in cfg and isinstance(cfg['model'], dict):
                cfg['model']['img_size'] = 96

        semi_cfg = cfg.get('semi', {})
        if 'params' in semi_cfg and 'second_model' in semi_cfg.get('params', {}):
            sm = semi_cfg['params']['second_model']
            if isinstance(sm, dict) and 'encoder' in sm:
                sm['encoder']['pretrained'] = False

        data = cfg.setdefault('data', {})
        if 'labeled_dir' not in data:
            data['labeled_dir'] = './data/labeled'
        if 'unlabeled_dir' not in data:
            data['unlabeled_dir'] = './data/unlabeled'
        if 'val_dir' not in data:
            data['val_dir'] = './data/val'

        test_list_path = os.path.join(BASE_DIR, 'data', 'test', 'list.txt')
        if 'test_list' in data:
            if not os.path.exists(str(data['test_list'])):
                if os.path.exists(test_list_path):
                    data['test_list'] = test_list_path
                else:
                    del data['test_list']

        patched_path = make_patched_yaml(cfg, f'semi_{name}')
        out_dir = os.path.join(BASE_DIR, 'output_test', f'semi_{name}')
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  [Semi] {name} ... ", end='', flush=True)
        t0 = time.time()
        success, output = run_cmd([
            PYTHON, 'semi_train.py',
            '--config', patched_path,
            '--output_dir', out_dir,
            '--device', 'cpu',
        ], timeout=600)
        elapsed = time.time() - t0

        status = 'PASS' if success else 'FAIL'
        err = extract_error(output) if not success else ''
        results[status.lower()].append({'cat': 'Semi', 'name': name, 'error': err})
        print(f"{status} ({elapsed:.1f}s)" + (f" — {err}" if err else ''))

        try:
            os.unlink(patched_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════
# Distillation
# ═══════════════════════════════════════════════════════════
def test_distillation():
    print("\n" + "="*70)
    print("  DISTILLATION")
    print("="*70)

    kd_dir = os.path.join(BASE_DIR, 'configs', 'training_paradigms', 'distillation')
    yamls = sorted([f for f in os.listdir(kd_dir) if f.endswith('.yaml')])

    for yf in yamls:
        name = yf.replace('.yaml', '')

        # Skip template configs — not actual training paradigms
        if name in ('student_small', 'teacher_large'):
            results['skip'].append({'cat': 'KD', 'name': name, 'error': 'Template config, not a paradigm'})
            print(f"\n  [KD] {name} ... SKIP (template config)")
            continue

        yaml_path = os.path.join(kd_dir, yf)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        cfg = patch_common_training(cfg)
        cfg['data']['type'] = 'image_mask'

        # cirkd_minibatch has a B*B pixel-pair loop — reduce img_size to avoid CPU timeout
        if name == 'cirkd_minibatch':
            cfg['data']['img_size'] = 32
            if 'model' in cfg:
                cfg['model']['img_size'] = 32

        data = cfg.setdefault('data', {})
        # train_distillation.py passes image_dir as root_dir to GenericDataset,
        # which then appends images/ and masks/. Use parent dirs as root_dir.
        if 'source' not in data:
            data['source'] = {'root_dir': './data/source'}
        else:
            data['source'].pop('image_dir', None)
            data['source'].pop('mask_dir', None)
            data['source'].setdefault('root_dir', './data/source')
        if 'target' not in data:
            data['target'] = {'root_dir': './data/target'}
        else:
            data['target'].pop('image_dir', None)
            data['target'].setdefault('root_dir', './data/target')
        if 'val' not in data:
            data['val'] = {'root_dir': './data/target_val'}
        else:
            data['val'].pop('image_dir', None)
            data['val'].pop('mask_dir', None)
            data['val'].setdefault('root_dir', './data/target_val')

        # Create architecture-specific teacher checkpoint
        teacher_ckpt = os.path.join(BASE_DIR, 'checkpoints', f'teacher_model_{name}.pth')
        ok = create_matching_checkpoint(cfg.get('model', {}), teacher_ckpt)
        if not ok:
            results['skip'].append({'cat': 'KD', 'name': name, 'error': 'Failed to build teacher checkpoint'})
            print(f"\n  [KD] {name} ... SKIP (teacher checkpoint build failed)")
            continue

        patched_path = make_patched_yaml(cfg, f'kd_{name}')
        out_dir = os.path.join(BASE_DIR, 'output_test', f'kd_{name}')
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  [KD] {name} ... ", end='', flush=True)
        t0 = time.time()
        success, output = run_cmd([
            PYTHON, 'train_distillation.py',
            '--teacher_config', patched_path,
            '--student_config', patched_path,
            '--teacher_ckpt', teacher_ckpt,
            '--output_dir', out_dir,
            '--device', 'cpu',
        ], timeout=600)
        elapsed = time.time() - t0

        status = 'PASS' if success else 'FAIL'
        err = extract_error(output) if not success else ''
        results[status.lower()].append({'cat': 'KD', 'name': name, 'error': err})
        print(f"{status} ({elapsed:.1f}s)" + (f" — {err}" if err else ''))

        try:
            os.unlink(patched_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════
# Weakly Supervised
# ═══════════════════════════════════════════════════════════
def test_weakly_supervised():
    print("\n" + "="*70)
    print("  WEAKLY SUPERVISED")
    print("="*70)

    weak_dir = os.path.join(BASE_DIR, 'configs', 'training_paradigms', 'weak_supervision')
    yamls = sorted([f for f in os.listdir(weak_dir) if f.endswith('.yaml')])

    for yf in yamls:
        name = yf.replace('.yaml', '')
        yaml_path = os.path.join(weak_dir, yf)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        cfg = patch_common_training(cfg)
        cfg['data']['type'] = 'image_mask'

        data = cfg.setdefault('data', {})
        if 'train_dir' not in data and 'root_dir' not in data:
            data['root_dir'] = './data'
        if 'val_dir' not in data:
            data['val_dir'] = './data/val'

        # Map weak_supervision.method → --supervision_type arg.
        # box/cam/mil/image_label go to their explicit branches;
        # all registry-based losses use the method name directly so
        # train_weakly_supervised.py falls into the 'else' dispatch.
        method = cfg.get('weak_supervision', {}).get('method', 'box')
        _METHOD_TO_SUPTYPE = {
            'box_supervised': 'box', 'boxinst': 'box',
            'cam': 'cam', 'cam_loss': 'cam',
            'mil': 'mil', 'image_label': 'image_label',
        }
        sup_type = _METHOD_TO_SUPTYPE.get(method, method)

        patched_path = make_patched_yaml(cfg, f'weak_{name}')
        out_dir = os.path.join(BASE_DIR, 'output_test', f'weak_{name}')
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  [Weak] {name} ... ", end='', flush=True)
        t0 = time.time()
        success, output = run_cmd([
            PYTHON, 'train_weakly_supervised.py',
            '--config', patched_path,
            '--supervision_type', sup_type,
            '--output_dir', out_dir,
            '--device', 'cpu',
        ], timeout=600)
        elapsed = time.time() - t0

        status = 'PASS' if success else 'FAIL'
        err = extract_error(output) if not success else ''
        results[status.lower()].append({'cat': 'Weak', 'name': name, 'error': err})
        print(f"{status} ({elapsed:.1f}s)" + (f" — {err}" if err else ''))

        try:
            os.unlink(patched_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════
# Text-Guided
# ═══════════════════════════════════════════════════════════
def test_text_guided():
    print("\n" + "="*70)
    print("  TEXT-GUIDED")
    print("="*70)

    text_dir = os.path.join(BASE_DIR, 'configs', 'training_paradigms', 'text_guided')
    yamls = sorted([f for f in os.listdir(text_dir) if f.endswith('.yaml')])

    for yf in yamls:
        name = yf.replace('.yaml', '')
        yaml_path = os.path.join(text_dir, yf)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        data = cfg.get('data', {})
        data_type = data.get('type', '')

        # Skip configs requiring external checkpoints that don't exist
        mllm = cfg.get('mllm', {})
        needs_ckpt = False
        if isinstance(mllm, dict):
            mask_gen = mllm.get('mask_generator', {})
            if isinstance(mask_gen, dict):
                ckpt = mask_gen.get('checkpoint', '')
                if ckpt and not os.path.exists(ckpt):
                    needs_ckpt = True

        if needs_ckpt:
            results['skip'].append({'cat': 'Text', 'name': name, 'error': 'Needs external checkpoint'})
            print(f"\n  [Text] {name} ... SKIP (needs external checkpoint)")
            continue

        # Skip LanguiDe (requires torch>=2.6 + external MLLM weights)
        if 'languide' in name:
            results['skip'].append({'cat': 'Text', 'name': name, 'error': 'Requires torch>=2.6 and external MLLM weights'})
            print(f"\n  [Text] {name} ... SKIP (requires torch>=2.6 + external MLLM weights)")
            continue

        # Skip LViT — BERT+ViT on CPU times out in smoke test
        if 'lvit' in name:
            results['skip'].append({'cat': 'Text', 'name': name, 'error': 'BERT+ViT too slow on CPU for smoke test'})
            print(f"\n  [Text] {name} ... SKIP (BERT+ViT too slow on CPU)")
            continue

        # Skip MLLM inference pipeline configs (grounder+mask_generator, no model: section)
        if 'model' not in cfg:
            results['skip'].append({'cat': 'Text', 'name': name, 'error': 'MLLM inference pipeline config (no model section)'})
            print(f"\n  [Text] {name} ... SKIP (MLLM inference pipeline, not trainable)")
            continue

        # Skip CLIP-based configs that need HuggingFace model download
        tg = cfg.get('model', {}).get('text_guided', {})
        if isinstance(tg, dict) and tg.get('prompt_mode') == 'clip' and tg.get('use_external_encoder'):
            results['skip'].append({'cat': 'Text', 'name': name, 'error': 'Requires CLIP model download from HuggingFace (offline)'})
            print(f"\n  [Text] {name} ... SKIP (requires CLIP download)")
            continue

        cfg = patch_common_training(cfg)

        patched_path = make_patched_yaml(cfg, f'text_{name}')
        out_dir = os.path.join(BASE_DIR, 'output_test', f'text_{name}')
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  [Text] {name} ... ", end='', flush=True)
        t0 = time.time()
        success, output = run_cmd([
            PYTHON, 'train_text_guided.py',
            '--config', patched_path,
            '--output_dir', out_dir,
            '--device', 'cpu',
        ], timeout=600)
        elapsed = time.time() - t0

        status = 'PASS' if success else 'FAIL'
        err = extract_error(output) if not success else ''
        results[status.lower()].append({'cat': 'Text', 'name': name, 'error': err})
        print(f"{status} ({elapsed:.1f}s)" + (f" — {err}" if err else ''))

        try:
            os.unlink(patched_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'output_test'), exist_ok=True)

    test_domain_adaptation()
    test_semi_supervised()
    test_distillation()
    test_weakly_supervised()
    test_text_guided()

    print("\n\n" + "="*70)
    print("  FINAL REPORT")
    print("="*70)
    print(f"  PASS: {len(results['pass'])}")
    print(f"  FAIL: {len(results['fail'])}")
    print(f"  SKIP: {len(results['skip'])}")

    if results['fail']:
        print(f"\n  ── FAILED ──")
        for item in results['fail']:
            print(f"  [{item['cat']}] {item['name']}")
            if item['error']:
                print(f"         {item['error']}")

    if results['skip']:
        print(f"\n  ── SKIPPED ──")
        for item in results['skip']:
            print(f"  [{item['cat']}] {item['name']}: {item['error']}")

    with open(os.path.join(BASE_DIR, 'output_test', 'all_paradigm_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to output_test/all_paradigm_results.json")
