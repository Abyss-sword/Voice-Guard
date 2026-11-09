"""Compare original MOSNet inference in separate Torch and TensorFlow environments."""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['torch','tensorflow','compare'],required=True)
    parser.add_argument('--out-dir',type=Path,default=Path('outputs/paper_quality/mosnet-parity'))
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    inputs=args.out_dir/'inputs.npz'
    if not inputs.exists():
        rng=np.random.default_rng(20261109)
        np.savez(inputs,**{f'frames_{frames}':rng.uniform(0,12,(1,frames,257)).astype(np.float32)
                          for frames in (17,33,84)})
    with np.load(inputs,allow_pickle=False) as source:values={key:source[key] for key in source.files}
    if args.backend=='torch':
        import torch
        from vguard.metrics.mos_original import OriginalMOSNet
        torch.set_num_threads(1);model=OriginalMOSNet()
        with torch.inference_mode():
            outputs={key:model.frame_scores(torch.from_numpy(value)).numpy() for key,value in values.items()}
    elif args.backend=='tensorflow':
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        model=tf.keras.models.load_model('checkpoints/mosnet-original/cnn_blstm.h5',compile=False)
        outputs={key:np.asarray(model(value,training=False)[1]).squeeze(-1) for key,value in values.items()}
    else:
        with np.load(args.out_dir/'torch.npz') as a,np.load(args.out_dir/'tensorflow.npz') as b:
            checks=[{'input':key,'torch_mean':float(a[key].mean()),'tensorflow_mean':float(b[key].mean()),
                     'max_frame_abs_error':float(np.max(np.abs(a[key]-b[key])))} for key in values]
        receipt={'checks':checks,'passed':all(item['max_frame_abs_error']<1e-4 for item in checks),
            'tolerance':1e-4,'inputs_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),
            'port_sha256':hashlib.sha256((ROOT/'vguard/metrics/mos_original.py').read_bytes()).hexdigest(),
            'claim_boundary':'Fixed magnitude inputs and original Keras H5; numerical parity is not proof of equivalence to Voice Guard authors unpublished evaluator.'}
        (args.out_dir/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(receipt,indent=2))
        if not receipt['passed']:raise SystemExit(1)
        return
    np.savez(args.out_dir/(args.backend+'.npz'),**outputs)
    print(args.backend,{key:float(value.mean()) for key,value in outputs.items()})


if __name__=='__main__':main()
