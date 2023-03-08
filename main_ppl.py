from extension.perplexity import perplexity
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', default='wikitext_103', type=str)
parser.add_argument('--hf_batch_size', default=16, type=int)
parser.add_argument('--model_id', required=True, type=str)
parser.add_argument('--load_path', default="/proj/gpu_d_98001/huggingface/hub/models--bigscience--bloom-1B1/snapshots/1e718be072aa40714c9dc34e35ea6b64979a65ad", type=str)
parser.add_argument('--output_path', required=True, type=str)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.scenario == 'wikitext_103':
        #with open("restricted/wikitext_103/wikitext103_processed.test.txt",'r') as f:
        with open("restricted/wikitext_103/wiki103.ppl.txt",'r') as f:
            #lines = f.readlines()
            lines = f.read()
        from extension.perplexity import detokenizers
        lines = detokenizers.wikitext_detokenizer(lines)
    else:
        print(f"{args.scenario} not implemented")
        exit() 

    p = perplexity.DS_Perplexity()
    result = p.compute(
        lines,
        model_name = args.load_path, 
        batch_size = args.hf_batch_size, 
        add_start_token = False, 
        device='cuda', 
        max_length=None
    )
    
    model_result_path = os.path.join(args.output_path, 'runs', 'ppl', f"{args.scenario}:model={args.model_id}")
    if not os.path.exists(model_result_path):
        os.makedirs(model_result_path)
    with open(os.path.join(model_result_path, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    