# hparam_search.py – unified Optuna-based hyperparameter search for AE variants

"""
python hparam_search.py --model dae --sweep sweeps/cae.yaml --n_trials 25
"""

from __future__ import annotations
import argparse, os, json
from datetime import datetime
import optuna
import torch
import inspect
import shutil, csv
# --- added imports for retrieval evaluation ---
import re
import copy

from utils.load_config import load_config, init_logger
from utils.training_utils import set_seed
from utils.data_utils import prepare_datasets
# NEW: import the real evaluation data loader and the main pipeline runner
from utils.data_utils import load_evaluation_data, prepare_inference_chunks
from main import PipelineRunner
from retrieval.embedder import EmbeddingCompressor

from training.train_cae import train_cae
from training.train_vae import train_vae
from training.train_dae import train_dae
from training.train_base import train_base

REGISTRY = {
    'cae': train_cae,
    'vae': train_vae,
    'dae': train_dae,
    'base': train_base,
}

# --- Retrieval evaluation (aligned with main) -------------------

# Removed obsolete cached-loader and direct FAISS eval; we now reuse PipelineRunner

def _load_sweep_yaml(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_space(trial: optuna.Trial, space_cfg: dict) -> dict:
    params = {}
    for name, spec in space_cfg.items():
        kind = spec['type']
        if kind == 'int':
            step = spec.get('step')
            low_i = int(spec['low']); high_i = int(spec['high'])
            if step is not None:
                params[name] = trial.suggest_int(name, low_i, high_i, step=int(step))
            else:
                params[name] = trial.suggest_int(name, low_i, high_i)
        elif kind == 'float':
            low = float(spec['low'])
            high = float(spec['high'])
            params[name] = trial.suggest_float(name, low, high, log=spec.get('log', False))
        elif kind == 'categorical':
            params[name] = trial.suggest_categorical(name, spec['choices'])
        else:
            raise ValueError(f'Unsupported space type: {kind}')
    return params

def allocate_device(trial_idx: int, available_gpus: list[int] | None) -> str | None:
    if not available_gpus:
        return None
    return f"cuda:{available_gpus[trial_idx % len(available_gpus)]}"


def extract_val_from_meta(ckpt_path: str) -> float:
    # history/ meta stem alignment
    if ckpt_path is None:
        return float('inf')
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    meta_path = os.path.join('models', 'history', stem + '.json')
    if not os.path.exists(meta_path):
        return float('inf')
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return float(meta.get('best_val_loss', float('inf')))
    except Exception:
        return float('inf')


def main():
    ap = argparse.ArgumentParser(description='Hyperparameter search for AE variants')
    ap.add_argument('--config', default='./config/config.yaml')
    ap.add_argument('--model', required=True, choices=list(REGISTRY.keys()))
    ap.add_argument('--sweep', required=True, help='YAML defining search space')
    ap.add_argument('--study', default=None)
    ap.add_argument('--storage', default='sqlite:///optuna_hpo.db')
    ap.add_argument('--n_trials', type=int, default=30)
    ap.add_argument('--timeout', type=int, default=None)
    ap.add_argument('--sampler', default='tpe', choices=['tpe','cmaes','random'])
    ap.add_argument('--pruner', default='asha', choices=['asha','median','none'])
    ap.add_argument('--direction', default='minimize', choices=['minimize','maximize'])
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = init_logger(cfg['logging'])
    train_cfg = cfg.get('training', {})
    base_seed = train_cfg.get('seed', 42)
    set_seed(base_seed, train_cfg.get('deterministic', False), logger=log.main)

    sweep_cfg = _load_sweep_yaml(args.sweep)
    space_cfg = sweep_cfg['space']
    constants = sweep_cfg.get('constants', {})

    ds_path = prepare_datasets(cfg, variant=args.model, dataset_override=None)

    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=base_seed)
    elif args.sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=base_seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=base_seed)

    if args.pruner == 'asha':
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
    elif args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    else:
        pruner = optuna.pruners.NopPruner()

    study_name = args.study or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction=args.direction, sampler=sampler, pruner=pruner,
                                storage=args.storage, study_name=study_name, load_if_exists=True)

    trainer_fn = REGISTRY[args.model]
    available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None

    def objective(trial: optuna.Trial):
        sampled = build_space(trial, space_cfg)
        params = {**constants, **sampled,
                  'dataset_path': ds_path,
                  'logger': log,
                  }
        params['device'] = allocate_device(trial.number, available_gpus)

        # default fallback hyperparams from config if not in space/constants
        model_defaults = cfg.get('models', {}).get(args.model, {})
        for key, default_val in model_defaults.items():
            params.setdefault(key, default_val)

        # Provide placeholder values if missing common args
        params.setdefault('epochs', cfg.get('training', {}).get('epochs', 20))
        params.setdefault('batch_size', cfg.get('training', {}).get('batch_size', 256))
        params.setdefault('lr', cfg.get('training', {}).get('learning_rate', 1e-3))

        # unify common optional args
        for opt_name, cfg_key in [('weight_decay','weight_decay'), ('adam_beta1','adam_beta1'), ('adam_beta2','adam_beta2')]:
            if opt_name not in params:
                params[opt_name] = cfg.get('training', {}).get(cfg_key, params.get(opt_name, 0.0 if 'weight_decay' in opt_name else 0.9))

        # Build descriptive base name (condensed hyperparams)
        tag_parts = []
        if 'latent_dim' in params: tag_parts.append(f"lat{int(params['latent_dim'])}")
        if 'hidden_dim' in params: tag_parts.append(f"hid{int(params['hidden_dim'])}")
        if 'lr' in params: tag_parts.append(f"lr{float(params['lr']):g}")
        if 'batch_size' in params: tag_parts.append(f"bs{int(params['batch_size'])}")
        if args.model == 'cae' and 'margin' in params: tag_parts.append(f"m{float(params['margin']):.2f}")
        if args.model == 'vae' and 'beta' in params: tag_parts.append(f"b{float(params['beta']):.2f}")
        param_tag = '_'.join(tag_parts)
        prefix = study_name if not study_name.startswith(f"{args.model}_") else study_name
        base_name = f"{prefix}_t{trial.number}_{param_tag}" if param_tag else f"{prefix}_t{trial.number}"

        # callback for pruning
        def report_cb(epoch, train_loss, val_loss, lr):
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        params['report_cb'] = report_cb
        # Provide model_save_path; do not add trial_suffix to avoid duplication
        params['model_save_path'] = os.path.join(cfg['paths']['checkpoints_dir'], base_name + '.pth')
        params['trial_suffix'] = None

        # ---- filter unsupported kwargs (e.g., dataset_file) ----
        trainer_sig = inspect.signature(trainer_fn)
        # If trainer supports **kwargs, pass everything through
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_sig.parameters.values()):
            filtered_params = params
        else:
            allowed = set(trainer_sig.parameters.keys())
            filtered_params = {k: v for k, v in params.items() if k in allowed}
            dropped = [k for k in params.keys() if k not in allowed]
            if dropped:
                log.main.debug("Dropped unsupported params for %s: %s", args.model, dropped)

        try:
            ckpt_path = trainer_fn(**filtered_params)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned("OOM")
        except optuna.TrialPruned:
            raise
        except Exception as e:
            log.main.error("Trial %d failed: %s", trial.number, e)
            return float('inf')

        val_metric = extract_val_from_meta(ckpt_path)
        trial.set_user_attr('ckpt', ckpt_path)
        return val_metric

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    # Save trials summary CSV
    history_dir = os.path.join('models', 'history')
    os.makedirs(history_dir, exist_ok=True)
    prefix = study_name if not study_name.startswith(f"{args.model}_") else study_name
    trials_csv = os.path.join(history_dir, f"hpo_{prefix}.csv")
    # Collect all param keys
    all_param_keys = sorted({k for t in study.trials for k in t.params.keys()})
    fieldnames = ['number', 'value', 'state'] + all_param_keys + ['ckpt', 'duration']
    with open(trials_csv, 'w', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for t in study.trials:
            row = {'number': t.number, 'value': t.value, 'state': str(t.state), 'ckpt': t.user_attrs.get('ckpt'), 'duration': getattr(t, 'duration', None)}
            for k in all_param_keys:
                row[k] = t.params.get(k)
            writer.writerow(row)

    # ---------------- Post-selection: retrieval re-rank --------------------
    EVAL_TOP_K = int(os.getenv("HPO_EVAL_TOPK", 5))
    SAMPLE_Q   = int(os.getenv("HPO_SAMPLE_QUERIES", 2000))
    PRIMARY    = os.getenv("HPO_PRIMARY", "ndcg@10")

    finals = [t for t in study.trials if (t.value is not None and t.state.name == "COMPLETE")]
    finals = sorted(finals, key=lambda t: t.value)[:EVAL_TOP_K]

    # Prepare evaluation artifacts ONCE (dataset, chunking, base embeddings)
    dataset_name = cfg.get('data', {}).get('dataset', 'squad')
    queries, corpus_docs, relevant = load_evaluation_data(dataset_name, max_samples=SAMPLE_Q)
    ch_cfg = cfg.get('chunking', {})
    use_chunking = bool(ch_cfg.get('enabled', False))
    corpus_texts = corpus_docs
    corpus_doc_ids = list(range(len(corpus_texts)))
    if use_chunking:
        chunks, chunk_index = prepare_inference_chunks(
            corpus_texts,
            mode=ch_cfg.get('mode', 'sliding'),
            max_tokens=ch_cfg.get('max_tokens', 128),
            stride=ch_cfg.get('stride', 64),
            min_tokens=ch_cfg.get('min_tokens', 48),
            tokenizer_name=ch_cfg.get('tokenizer_name', cfg['embedding_model']['name']),
            index_out=ch_cfg.get('index_out'),
            store_chunk_text=ch_cfg.get('store_chunk_text', True),
        )
        corpus_texts = chunks
        corpus_doc_ids = chunk_index['doc_id'].astype(int).tolist()

    # Base compressor (no AE) and base embeddings for reuse across trials
    base_compressor = EmbeddingCompressor(
        base_model_name=cfg['embedding_model']['name'],
        autoencoder=None,
        device=cfg.get('training', {}).get('device') or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    with torch.inference_mode():
        base_corpus_emb = base_compressor.encode_text(list(corpus_texts), compress=False)
        base_query_emb = base_compressor.encode_text(list(queries), compress=False)

    # Parse PRIMARY K (e.g., ndcg@10 → 10)
    mK = re.search(r"@(\d+)$", PRIMARY)
    primary_k = int(mK.group(1)) if mK else 10

    # Clone cfg and force metric list to align with PRIMARY K
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval.setdefault('evaluation', {})
    cfg_eval['evaluation']['retrieval_metrics'] = [f"Recall@{primary_k}", f"MRR@{primary_k}", f"nDCG@{primary_k}"]
    cfg_eval.setdefault('retrieval', {})
    cfg_eval['retrieval']['top_k'] = int(primary_k)
    if use_chunking:
        cfg_eval['retrieval']['candidate_k'] = int(max(cfg_eval['retrieval'].get('candidate_k', primary_k * 3), primary_k))

    ret_records = []
    best_ret_score = -1.0
    best_ret_trial = None
    for t in finals:
        ckpt = t.user_attrs.get("ckpt")
        if not ckpt or not os.path.exists(ckpt):
            log.main.warning("Skipped trial %d: missing ckpt", t.number)
            continue
        try:
            runner = PipelineRunner(
                cfg_eval, args.model, log,
                pre_corpus_texts=corpus_texts,
                pre_corpus_doc_ids=corpus_doc_ids,
                base_corpus_embeddings=base_corpus_emb,
                base_query_embeddings=base_query_emb,
                checkpoint_override=ckpt,
            )
            result = runner.process(queries, corpus_texts, relevant_docs=relevant, generate=False)
            retm = result.get('retrieval_metrics', {})
            # Lower-case metric keys and pick primary
            metrics = {}
            for k, v in retm.items():
                try:
                    metrics[k.lower()] = float(v.get('mean', 0.0))
                except Exception:
                    try:
                        metrics[k.lower()] = float(v)
                    except Exception:
                        continue
            score = float(metrics.get(PRIMARY, -1.0))
            ret_records.append({"trial": t.number, "ckpt": ckpt, **metrics})
            if score > best_ret_score:
                best_ret_score, best_ret_trial = score, t
        except Exception as e:
            log.main.warning("Retrieval eval failed (trial %d): %s", t.number, e)

    if ret_records:
        ret_csv = os.path.join(history_dir, f"hpo_{args.model}_{study_name}_retrieval.csv")
        keys = sorted({k for r in ret_records for k in r.keys()})
        with open(ret_csv, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=keys); writer.writeheader(); writer.writerows(ret_records)
        print("[HPO][RET] Metrics CSV:", ret_csv)

    if best_ret_trial is not None:
        print(f"[HPO][RET] Best by {PRIMARY}: trial={best_ret_trial.number} ckpt={best_ret_trial.user_attrs.get('ckpt')} score={best_ret_score:.4f}")
        try:
            ret_ckpt = best_ret_trial.user_attrs.get('ckpt')
            if ret_ckpt and os.path.exists(ret_ckpt):
                ret_copy = os.path.join(os.path.dirname(ret_ckpt), 'bmret_' + os.path.basename(ret_ckpt))
                shutil.copy2(ret_ckpt, ret_copy)
                print('[HPO][RET] Best retrieval model copy:', ret_copy)
        except Exception as e:
            log.main.warning('Could not copy best retrieval model: %s', e)

    # Best model prefix copy
    best_ckpt = study.best_trial.user_attrs.get('ckpt')
    bm_ckpt = None
    if best_ckpt and os.path.exists(best_ckpt):
        bm_dir = os.path.dirname(best_ckpt)
        bm_ckpt = os.path.join(bm_dir, 'bm_' + os.path.basename(best_ckpt))
        try:
            shutil.copy2(best_ckpt, bm_ckpt)
        except Exception as e:
            log.main.warning("Could not copy best model: %s", e)

    print('[HPO] Trials CSV:', trials_csv)
    print('[HPO] Best value:', study.best_trial.value)
    print('[HPO] Best params:', study.best_trial.params)
    print('[HPO] Best ckpt:', best_ckpt)
    if bm_ckpt:
        print('[HPO] Best model copy:', bm_ckpt)



if __name__ == "__main__":
    main()
