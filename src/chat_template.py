"""Chat template formatting for different model families."""


def get_chat_template(tokenizer, model_name, user_message):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": user_message}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return user_message
