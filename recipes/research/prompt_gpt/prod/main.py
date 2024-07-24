from recipes.research.prompt_gpt.prod import load_promptgpt_model, sample_prompt_gpt

if __name__ == "__main__":
    commit_hash = "55f2f18"
    model = load_promptgpt_model(commit_hash, device="cuda", cache=True)

    prompt = "Disco dancing track with a retro feel of the 1970's"

    pred_text = sample_prompt_gpt(model, prompt, temperature=1.0)

    print(pred_text)
