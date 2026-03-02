
def build_bedrock_inference_data_object(idx, pred, model_name, system_prompt=None, max_tokens=512,
                                        temperature=0.0):

    if "anthropic" in model_name:
        data_object = {
            "recordID": idx,
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                # "top_p": self.top_p,
                "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": pred
                        }
                        ]
                    }
                ]
            }
        }
        if system_prompt:
            data_object['modelInput']['system'] = system_prompt.value

    elif "amazon.nova" in model_name:
        data_object = {
            "recordID": idx,
            "modelInput": {
                # "max_tokens": max_tokens,
                # "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": pred,
                            }
                        ]
                    }
                ]
            }
        }
        if system_prompt:
            data_object['modelInput']['system'] = [{"text": system_prompt.value}]

    elif "qwen" in model_name:
        data_object = {
            "recordId": idx,  # Also change to lowercase 'd' for consistency
            "modelInput": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": pred
                    }
                ],
            }
        }
        if system_prompt:
            data_object['modelInput']['messages'].insert(0, {
                "role": "system",
                "content": system_prompt.value
            })

    else:
        raise ValueError(f"Unsupported model type for bedrock inference data object: {model_name}")

    return data_object


def gather_response_bedrock_inference(response, model_name):

    if "anthropic" in model_name:
        parsed_response = response['modelOutput']['content']
        stop_reason = response['modelOutput']['stop_reason']
    elif "amazon.nova" in model_name:
        parsed_response = response['modelOutput']['output']['message']['content']
        stop_reason = response['modelOutput']['output']['stop_reason']
    elif "qwen" in model_name:
        # print(response)
        parsed_response = response['modelOutput']['choices'][0]['message']['content']
        stop_reason = response['modelOutput']['choices'][0]['finish_reason']
    else:
        raise ValueError(f"Unsupported model type for bedrock inference data object: {model_name}")
    return parsed_response, stop_reason
