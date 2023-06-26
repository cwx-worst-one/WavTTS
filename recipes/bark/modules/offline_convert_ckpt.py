import traceback
import torch
import collections
import argparse
from recipes.bark.lit_modules.v1.valle_coarse import ValleCoarseModule
from samantha.models.ctiga_llama import create_ctiga_from_sparse_llama


def remap_optimizer_state_dict(orig_optimizer_states, n_layer):
    ORIG_WEIGHT_LAYER_OFFSET = 9
    ORIG_W_Q_OFFSET = 0
    ORIG_W_K_OFFSET = 1
    ORIG_W_V_OFFSET = 2
    ORIG_W_O_OFFSET = 3
    ORIG_W_FFN_W1_OFFSET = 4
    ORIG_W_FFN_W2_OFFSET = 5
    ORIG_W_FFN_W3_OFFSET = 6
    ORIG_W_ATTN_NORM_OFFSET = 7
    ORIG_W_FFN_NORM_OFFSET = 8

    ORIG_OPT_LAYER_OFFSET = 9
    ORIG_OPT_Q_OFFSET = 0
    ORIG_OPT_K_OFFSET = 1
    ORIG_OPT_V_OFFSET = 2
    ORIG_OPT_O_OFFSET = 3
    ORIG_OPT_FFN_W1_OFFSET = 4
    ORIG_OPT_FFN_W2_OFFSET = 5
    ORIG_OPT_FFN_W3_OFFSET = 6
    ORIG_OPT_ATTN_NORM_OFFSET = 7
    ORIG_OPT_FFN_NORM_OFFSET = 8

    CTIGA_WEIGHT_LAYER_OFFSET = 7
    CTIGA_W_QKV_OFFSET = 1
    CTIGA_W_OUT_OFFSET = 2
    CTIGA_W_NORM1_OFFSET = 3
    CTIGA_W_MLP_FC1_OFFSET = 4
    CTIGA_W_MLP_FC2_OFFSET = 5
    CTIGA_W_NORM2_OFFSET = 6

    CTIAG_OPT_LAYER_OFFSET = 6
    CTIGA_OPT_QKV_OFFSET = 0
    CTIGA_OPT_OUT_OFFSET = 1
    CTIGA_OPT_NORM1_OFFSET = 2
    CTIGA_OPT_MLP_FC1_OFFSET = 3
    CTIGA_OPT_MLP_FC2_OFFSET = 4
    CTIGA_OPT_NORM2_OFFSET = 5

    remap_optimizer_states = []
    for opt_idx, orig_optimizer_state in enumerate(orig_optimizer_states):
        assert len(orig_optimizer_states[opt_idx]["state"]) == ORIG_OPT_LAYER_OFFSET * n_layer + 3

        remap_optimizer_state = {"state": [], "param_groups": orig_optimizer_state["param_groups"]}
        for i in range(remap_optimizer_state["param_groups"]):
            assert (
                len(orig_optimizer_states[opt_idx]["param_groups"][i]["params"]) == ORIG_OPT_LAYER_OFFSET * n_layer * 3
            )
            remap_optimizer_state["param_groups"][i]["params"] = list(range(CTIAG_OPT_LAYER_OFFSET * n_layer + 3))

        # wte
        orig_wte_opt_state = orig_optimizer_state["state"][0]
        remap_wte_opt_state = dict(
            step=orig_wte_opt_state,
            exp_avg=orig_wte_opt_state["exp_avg"].detach().clone(),
            exp_avg_sq=orig_wte_opt_state["exp_avg_sq"].detach().clone(),
        )
        remap_optimizer_state["state"] += [remap_wte_opt_state]

        # layers
        for layer_idx in range(n_layer):
            orig_q_opt_state = orig_optimizer_state["state"][layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_Q_OFFSET]
            orig_k_opt_state = orig_optimizer_state["state"][layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_K_OFFSET]
            orig_v_opt_state = orig_optimizer_state["state"][layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_V_OFFSET]
            orig_o_opt_state = orig_optimizer_state["state"][layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_O_OFFSET]
            orig_ffn_w1_opt_state = orig_optimizer_state["state"][
                layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_FFN_W1_OFFSET
            ]
            orig_ffn_w2_opt_state = orig_optimizer_state["state"][
                layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_FFN_W2_OFFSET
            ]
            orig_ffn_w3_opt_state = orig_optimizer_state["state"][
                layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_FFN_W3_OFFSET
            ]
            orig_attn_norm_opt_state = orig_optimizer_state["state"][
                layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_ATTN_NORM_OFFSET
            ]
            orig_ffn_norm_opt_state = orig_optimizer_state["state"][
                layer_idx * ORIG_OPT_LAYER_OFFSET + ORIG_OPT_FFN_NORM_OFFSET
            ]

            ctiga_qkv_opt_state = dict(
                step=orig_q_opt_state["step"],
                exp_avg=torch.cat(
                    [x["exp_avg"] for x in [orig_q_opt_state, orig_k_opt_state, orig_v_opt_state]], dim=0
                ),
                exp_avg_sq=torch.cat(
                    [x["exp_avg_sq"] for x in [orig_q_opt_state, orig_k_opt_state, orig_v_opt_state]], dim=0
                ),
            )
            ctiga_o_opt_state = dict(
                step=orig_o_opt_state,
                exp_avg=orig_o_opt_state["exp_avg"].detach().clone(),
                exp_avg_sq=orig_o_opt_state["exp_avg_sq"].detach().clone(),
            )
            ctiga_norm1_opt_state = dict(
                step=orig_attn_norm_opt_state,
                exp_avg=orig_attn_norm_opt_state["exp_avg"].detach().clone(),
                exp_avg_sq=orig_attn_norm_opt_state["exp_avg_sq"].detach().clone(),
            )
            ctiga_mlp_w1_opt_state = dict(
                step=orig_q_opt_state["step"],
                exp_avg=torch.cat([x["exp_avg"] for x in [orig_ffn_w1_opt_state, orig_ffn_w3_opt_state]]),
                exp_avg_sq=torch.cat([x["exp_avg_sq"] for x in [orig_ffn_w1_opt_state, orig_ffn_w3_opt_state]]),
            )
            ctiga_mlp_w2_opt_state = dict(
                step=orig_ffn_w2_opt_state,
                exp_avg=orig_ffn_w2_opt_state["exp_avg"].detach().clone(),
                exp_avg_sq=orig_ffn_w2_opt_state["exp_avg_sq"].detach().clone(),
            )
            ctiga_norm2_opt_state = dict(
                step=orig_ffn_norm_opt_state,
                exp_avg=orig_ffn_norm_opt_state["exp_avg"].detach().clone(),
                exp_avg_sq=orig_ffn_norm_opt_state["exp_avg_sq"].detach().clone(),
            )
            remap_optimizer_state["state"] += [
                ctiga_qkv_opt_state,
                ctiga_o_opt_state,
                ctiga_norm1_opt_state,
                ctiga_mlp_w1_opt_state,
                ctiga_mlp_w2_opt_state,
                ctiga_norm2_opt_state,
            ]
        orig_norm_opt_state = orig_optimizer_state["state"][1 + ORIG_OPT_LAYER_OFFSET * n_layer]
        remap_norm_opt_state = dict(
            step=orig_norm_opt_state,
            exp_avg=orig_norm_opt_state["exp_avg"].detach().clone(),
            exp_avg_sq=orig_norm_opt_state["exp_avg_sq"].detach().clone(),
        )
        orig_lmhead_opt_state = orig_optimizer_state["state"][2 + ORIG_OPT_LAYER_OFFSET * n_layer]
        remap_lmhead_opt_state = dict(
            step=orig_lmhead_opt_state,
            exp_avg=orig_lmhead_opt_state["exp_avg"].detach().clone(),
            exp_avg_sq=orig_lmhead_opt_state["exp_avg_sq"].detach().clone(),
        )
        remap_optimizer_state["state"] += [remap_norm_opt_state, remap_lmhead_opt_state]
        assert remap_optimizer_state == CTIAG_OPT_LAYER_OFFSET + 3
        remap_optimizer_states.append(remap_optimizer_state)
    return remap_optimizer_states


def transfer_to_ctiga(orig_ckpt_path, ctiga_ckpt_path):
    orig_model_ckpt = torch.load(orig_ckpt_path, map_location="cpu")
    print(orig_model_ckpt.keys())
    # exit()
    # init orig llama model
    orig_llama_model = orig_model_ckpt["hyper_parameters"]["model_cls"]()
    orig_state_dict = orig_model_ckpt["state_dict"]
    orig_state_dict = collections.OrderedDict(
        {key.replace('model.', ''): value for key, value in orig_state_dict.items()}
    )

    orig_llama_model.load_state_dict(orig_state_dict)
    print(f"[INFO] Load orig llama model finished from {orig_ckpt_path}")

    # remap to ctiga llama model
    ctiga_llama_model = create_ctiga_from_sparse_llama(orig_llama_model)
    print(f"[INFO] Init&&Load ctiga llama model finished.")

    # update state_dict && opt state&& hyper_parameters in model's checkpoint
    orig_model_ckpt["state_dict"].clear()
    orig_model_ckpt["state_dict"].update({"model." + k: v for k, v in ctiga_llama_model.state_dict().items()})
    orig_model_ckpt["optimizer_states"].clear()
    orig_model_ckpt["optimizer_states"] += remap_optimizer_state_dict(
        orig_model_ckpt["optimizer_states"], orig_llama_model.params.n_layers
    )
    orig_model_ckpt["hyper_parameters"]["provider"] = "ctiga"

    torch.save(orig_model_ckpt, ctiga_ckpt_path)
    print(f"[INFO] Save transferred checkpoint to {ctiga_ckpt_path}")

    # check
    try:
        print(ValleCoarseModule.load_from_checkpoint(ctiga_ckpt_path))
    except Exception as e:
        raise RuntimeError(f"Transfer to ctiga failed due to {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transfer to ctiga model")
    parser.add_argument("-i", "--in_checkpoint", type=str, required=True)
    parser.add_argument("-o", "--out_checkpoint", type=str, required=True)
    args = parser.parse_args()
    transfer_to_ctiga(args.in_checkpoint, args.out_checkpoint)
