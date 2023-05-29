# Multi-cards inference using huggingface accelerate
import logging
from absl import app, flags
import readline # required, support delete in terminal
import json
from tqdm import tqdm
import os
import torch
import requests
import traceback
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from cruise.utilities.hdfs_io import hopen, hput

from seed_finetune.models.generator import Generator
from seed_finetune.utils.common import parse_args
from seed_finetune.utils.text import preprocess, postprocess

from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint

from open_lark import OpenLark
from open_lark.dt_drive import DriveFileType, DriveFileUser, \
    DriveFilePermission, DriveFileUserPermission, DriveFilePublicLinkSharePermission
from open_lark.dt_enum import MessageType

TENANT_ACCESS_TOKEN_URL = \
  "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint", None, "A specificed saved checkpoint path, or the output directory."
)

flags.DEFINE_string(
    "step", None, "The checkpoint step to be used."
)

flags.DEFINE_float(
    "top_p", 0.7, "The topp"
)

flags.DEFINE_integer(
    "top_k", None, "The topk"
)

flags.DEFINE_float(
    "temperature", 1.0, "The temperature"
)

flags.DEFINE_integer(
    "max_length", 2048, "The maximum tokens (ignore prompt) to generate."
)

flags.DEFINE_integer(
    "max_new_tokens", 2048, "The maximum tokens (ignore prompt) to generate."
)

flags.DEFINE_string(
    "input_file", None, "The input file to be processed."
)

flags.DEFINE_string(
    "output_file", None, "The output file to be processed."
)

flags.DEFINE_string(
    "model", None, "The yaml model configuration path."
)

flags.DEFINE_string(
    "data", None, "The yaml data module configuration path."
)

flags.DEFINE_string(
    "tokenizer", None, "The yaml data module configuration path."
)

flags.DEFINE_string(
    "tokenizer_type", None, "The yaml data module configuration path."
)

flags.DEFINE_string(
    "generator", None, "The yaml generator configuration path."
)

flags.DEFINE_integer(
    "n_gpus", 8, "Number of GPUs used for inference."
)

flags.DEFINE_bool(
    "use_hf_accelerate", False, "Whether to use huggingface accelerate for infer"
)

flags.DEFINE_bool(
    "add_special_tokens", True, "Whether to add specical tokens (SEP/EOS) for dialog history"
)

flags.DEFINE_string(
    "model_name", None, "The evaluation model name.")

flags.DEFINE_string(
    "APP_ID", "cli_a021df404f78100e",
    "The Lark bot app id.")

flags.DEFINE_string(
    "APP_SECRET", "MK4uoXE8ENQBoSrTWo7vGfMMp8bqGe5Z",
    "The Lark bot app secret.")

flags.DEFINE_string(
    "email", None,
    "You email address.")

flags.DEFINE_string(
    "FOLDER_TOKEN", None,
    "The lark cloud driver folder token")

flags.DEFINE_string(
    "eval_precision", "bf16",
    "The precision when load model")

def infer_transformer_device_map(n_gpus=8):
    device_map = {
        'gpt.transformer.wte': 0,
        'gpt.transformer.wpe': 0,
        'gpt.transformer.pre_token_proj': 0,
        'gpt.transformer.post_token_proj': 0,
        'gpt.transformer.drop': 0,
        'gpt.transformer.ln_f': 0,
        'gpt.lm_head': 0,
        'gpt.loss_fct': 0
    }

    for i in range(64):
        device_map['gpt.transformer.h.' + str(i)] = i % n_gpus
    return device_map

def initialize(local_rank=0):
    """Prepare model, tokenizer, checkpoint"""
    
    model_config, generation_config, checkpoint, tokenizer = parse_args(FLAGS)
    rprint("Initialize model...")
    if FLAGS.use_hf_accelerate:
        with init_empty_weights():
            model = Generator(model_config=model_config, generation_config=generation_config)
            model.eval()
        
        rprint(f"Load from: {checkpoint}")
        device_map = infer_transformer_device_map(FLAGS.n_gpus)
        rprint(f"Device map: ", json.dumps(device_map, indent=2))
        model = load_checkpoint_and_dispatch(
            model, checkpoint, device_map=device_map, no_split_module_classes=["GPT2Block"]
        )
        rprint(f"Final device map: ", json.dumps(model.hf_device_map, indent=2))
    else:
        #print(model_config.gradient_checkpointing)
        model = Generator(model_config=model_config, generation_config=generation_config)
        rprint(f"Load checkpoint from {checkpoint}")
        model.load(checkpoint)
        if FLAGS.eval_precision == "bf16":
            model.to(torch.bfloat16)
        else:
            model.half()
        model.eval()
        model = model.cuda(local_rank)
    return model, tokenizer

def play():
    model, tokenizer = initialize()
    console = Console()
    help = "### Usage" + "\n" + \
           "- type `/exit` to exit bot" + "\n" + \
           "- type `/debug on/off` to enable/disable original model input/output" + "\n" + \
           "- type `/multi-turn on/off` to enable/disable multi turn mode" + "\n" + \
           "- type `/clear` to clear dialogue history"
    console.print(Markdown(help))
    is_debug = False
    is_multi_turn = False
    history = []
    temperature = FLAGS.temperature
    top_k = FLAGS.top_k
    top_p = FLAGS.top_p
    max_new_tokens = FLAGS.max_new_tokens
    while True:
        try:
            text = console.input(">> ")
            if text == "/exit":
                console.print("See you!")
                break
            if text.startswith("/debug"):
                switch = text.split()[-1]
                if switch == "on":
                    console.print("Debug mode: on")
                    is_debug = True
                else:
                    console.print("Debug mode: off")
                    is_debug = False
                continue
            if text.startswith("/multi-turn"):
                switch = text.split()[-1]
                if switch == "on":
                    console.print("Enable multi-turn")
                    is_multi_turn = True
                else:
                    console.print("Disable multi-turn")
                    is_multi_turn = False
                continue
            
            if text.startswith("/topp"):
                top_p = float(text.split()[-1])
                continue

            if text.startswith("/topk"):
                if text.split()[-1] == "none":
                    top_k = None
                else:
                    top_k = int(text.split()[-1])
                continue

            if text.startswith("/temperature"):
                temperature = float(text.split()[-1])
                continue
            
            if text.startswith("/clear"):
                history = []
                console.print("Clear dialogue history.")
                continue

            history.append(text)
            
            if is_multi_turn:
                prompts = history
            else:
                prompts = history[-1:]
            input_ids, prompt_len = preprocess(
                prompts, tokenizer, add_special_tokens=FLAGS.add_special_tokens, max_length=FLAGS.max_length,
                max_length_per_turn=FLAGS.max_length)
            if is_debug:
                console.print(
                    f'Prompt: {text}, Tokens: {tokenizer.decode(input_ids[0].tolist())}, Token Ids: {input_ids}, Length: {prompt_len}'
                )
                console.print(
                    f'top_p: {top_p}, top_k: {top_k}, temperature: {temperature}'
                )
            outputs = model.generate(
                input_ids, top_p=top_p, top_k=top_k, temperature=temperature,
                max_new_tokens=max_new_tokens, max_length=FLAGS.max_length)

            response = postprocess(outputs.sequences[0], prompt_len, tokenizer)

            history.append(response)
            md = Markdown(response)
            console.print(md)
            if is_debug:
                console.print("Raw: \n", [response])

        except Exception as e:
            console.print(e)
            console.print(traceback.format_exc())

def load_test_data(input_file):
    test_data = []
    with hopen(input_file, 'r') as f:
        for idx, line in enumerate(f):
            dt = json.loads(line.strip("\n"))
            # TODO: support multi-turn
            if isinstance(dt, list):
                dt = dt[0]
            if "id" not in dt:
                dt["id"] = str(idx)
            test_data.append(dt)
    return test_data

def infer(input_file, output_file):
    model, tokenizer = initialize()
    console = Console()
    try:
        test_data = load_test_data(input_file)
        responses=[]
        for data in tqdm(test_data):
            text = data['prompt']
            input_ids, prompt_len = preprocess(
                [text], tokenizer, add_special_tokens=FLAGS.add_special_tokens,
                max_length=FLAGS.max_length, max_length_per_turn=FLAGS.max_length)
            console.print(f'Prompt: {text}, Len: {prompt_len}')
            outputs = model.generate(
                input_ids, top_p=FLAGS.top_p, top_k=FLAGS.top_k, temperature=FLAGS.temperature,
                max_new_tokens=FLAGS.max_new_tokens
            )

            response = postprocess(outputs.sequences[0], prompt_len, tokenizer)
            console.print(f'Response: {response}')
            responses.append(
                json.dumps(
                    {
                        "id": data['id'],
                        "prompt": data['prompt'],
                        "prompt_type": data.get("prompt_type", ""),
                        "ref": data.get("ref", ""),
                        "response": response
                    }, ensure_ascii=False) + "\n"
                )
        with open(os.path.basename(output_file), 'w') as f:
            for line in responses:
                f.write(line)
        hput(os.path.basename(output_file), os.path.dirname(output_file))
        return True
    except Exception as e:
        console.print(e)
        console.print(traceback.format_exc())
        return False

def get_tenant_access_token():
  """Get tenant access token used for lark authorization."""
  return requests.post(
    url=TENANT_ACCESS_TOKEN_URL,
        data={
            "app_id": FLAGS.APP_ID,
            "app_secret": FLAGS.APP_SECRET
        }
    ).json()

def set_option_list(sheet_token, sheet_id):
  url = f'https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{sheet_token}/dataValidation'
  headers = {
    "Authorization": 'Bearer ' + get_tenant_access_token()["tenant_access_token"],
    "Content-Type": "application/json; charset=utf-8"
  }
  body = {
    'range': f'{sheet_id}!E2:E101',
    'dataValidationType': 'list',
    'dataValidation': {
      'conditionValues': ['2', '1', '0'],
      'options': {
        'multipleValues': False,
        'highlightValidData': True,
        'colors': ['#1bce00', '#ffd966', '#f44336']
      }
    }
  }
  rsp = requests.post(url, json=body, headers=headers)
  if rsp.status_code == 200:
    print("Set options list success.")
  return rsp

def send_lark_sheet(input_file):
    if not FLAGS.email or not FLAGS.FOLDER_TOKEN:
        logging.info("Email and folder token should be provided.")
        return None
    lark = OpenLark(FLAGS.APP_ID, FLAGS.APP_SECRET)
    tenant_access_token = get_tenant_access_token()["tenant_access_token"]
    sheet = lark.create_drive_file(
        tenant_access_token,
        folder_token=FLAGS.FOLDER_TOKEN,
        title=FLAGS.model_name,
        type=DriveFileType.sheet
    )
    sheet_meta = lark.get_drive_sheet_meta(
        tenant_access_token,
        sheet_token=sheet.token
    )
    # Set options
    lark.write_drive_sheet_cells(
        tenant_access_token, sheet_token=sheet.token, sheet_id=sheet_meta.sheets[0].id,
        range="A1:E1", values=[("prompt", "prompt_type", "ref", "response", "score")])

    values = []
    with hopen(input_file) as f:
        for line in f:
            instance = json.loads(line)
            values.append((
                str(instance["prompt"]),
                str(instance["prompt_type"]),
                str(instance.get("ref", "")),
                str(instance["response"]),
            ))
    tenant_access_token = get_tenant_access_token()["tenant_access_token"]
    lark.append_write_drive_sheet_cells(tenant_access_token,
                                        sheet_token=sheet.token,
                                        sheet_id=sheet_meta.sheets[0].id,
                                        range="A:D",
                                        values=values)

    set_option_list(sheet_token=sheet.token, sheet_id=sheet_meta.sheets[0].id)

    lark.update_drive_file_public_permission(
        user_access_token=get_tenant_access_token()["tenant_access_token"],
        file_token=sheet.token,
        file_type=DriveFileType.sheet,
        copy_print_export_status=False,
        comment=True,
        tenant_shareable=False,
        link_share_entity=DriveFilePublicLinkSharePermission.tenant_readable,
        external_access=False,
        invite_external=False
    )
    lark.add_drive_file_permission(
        user_access_token=get_tenant_access_token()["tenant_access_token"],
        file_token=sheet.token,
        file_type=DriveFileType.sheet,
        members=[DriveFileUserPermission(
        email=FLAGS.email,
        permission=DriveFilePermission.edit
    )])

    lark.send_raw_message(
        email=FLAGS.email, msg_type=MessageType.text,
        content={
        "text": FLAGS.model_name + sheet.url
    })

def main(_):
    if FLAGS.input_file:
        infer(FLAGS.input_file, FLAGS.output_file)
        send_lark_sheet(FLAGS.output_file)
    else:
        play()

if __name__ == '__main__':
    app.run(main)
