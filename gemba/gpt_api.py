import os
import sys
import time
import ipdb
import logging
from termcolor import colored
from datetime import datetime
import openai
import tqdm
import boto3
import json

from gemba.bedrock_utils import build_bedrock_inference_data_object, gather_response_bedrock_inference


# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, verbose=False):
        self.verbose = verbose

        if "AWS_ACCESS_KEY_ID" in os.environ or "AWS_PROFILE" in os.environ:
            # Bedrock API access
            region_name = os.environ.get("AWS_REGION_NAME", "us-east-1")
            profile_name = os.environ.get("AWS_PROFILE")
            if profile_name:
                session = boto3.Session(profile_name=profile_name)
                self.client = session.client(service_name='bedrock-runtime', region_name=region_name)
            else:
                self.client = boto3.client(service_name='bedrock-runtime', region_name=region_name)
            self.api_type = "bedrock"
            # S3 temporary data config
            self.input_data_config = {
                "s3InputDataConfig": {
                    "s3Uri": f"s3://ctrlpost-bedrock-inference-bucket/input_data/input_gemba_tmp.jsonl",
                }
            }
            self.output_data_config = {
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://ctrlpost-bedrock-inference-bucket/output_data/",
                }
            }
        elif "OPENAI_AZURE_ENDPOINT" in os.environ:
            assert "OPENAI_AZURE_KEY" in os.environ, "OPENAI_AZURE_KEY not found in environment"

            # Azure API access
            self.client = openai.AzureOpenAI(
                api_key=os.environ["OPENAI_AZURE_KEY"],
                azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                api_version="2023-07-01-preview"
            )
            self.api_type = "openai"
        elif "OPENAI_API_KEY" in os.environ:
            # OpenAI API access
            self.client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )
            self.api_type = "openai"
        else:
            raise Exception("OPENAI_API_KEY, OPENAI_AZURE_KEY, or AWS credentials not found in environment")

        logging.getLogger().setLevel(logging.CRITICAL)  # in order to suppress all these HTTP INFO log messages

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None):
        request_key = json.dumps({"model": model, "temperature": temperature, "prompt": prompt}, sort_keys=True)

        if cache is not None and request_key in cache and cache[request_key] is not None and len(cache[request_key]) > 0:
            answers = cache[request_key]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens)
            if cache is not None:
                cache[request_key] = answers

        # there is no valid answer
        if len(answers) == 0:
            return [{
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                    }]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]
            answer_id += 1
            answer = parse_response(full_answer)
            if self.verbose:
                print(f"Answer (t={temperature}): " + colored(answer, "yellow") + " (" + colored(full_answer, "blue") + ")", file=sys.stderr)
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                }
            )

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0:
            return self.request(prompt, model, parse_response, temperature=temperature + 1, answer_id=answer_id, cache=cache)

        return parsed_answers

    def request_batch(self, df, model, parse_response, max_tokens=None):

        # Prepare the data for batch inference in AWS Bedrock
        id2prompt = {}
        with (open(f'tmp/input_gemba_tmp.jsonl', 'w') as f):
            for idx, src in df.iterrows():
                # Write each prediction to the file
                data_object = build_bedrock_inference_data_object(idx, src['prompt'], model, max_tokens=max_tokens)
                f.write(json.dumps(data_object) + '\n')
                id2prompt[idx] = src['prompt']

        # Upload the input data to S3
        s3 = boto3.client('s3')
        s3.upload_file(
            f'tmp/input_gemba_tmp.jsonl', "ctrlpost-bedrock-inference-bucket",
            f'input_data/input_gemba_tmp.jsonl.jsonl')

        # Invoke batch job with the input data
        print(f"S3 region: {self.bedrock_client.meta.region_name}")
        response = self.bedrock_client.create_model_invocation_job(
            jobName=f"gemba-job-{int(time.time())}",
            roleArn="arn:aws:iam::209378968454:role/ctrlpost-bedrock-inference-role",
            modelId=model,
            inputDataConfig=self.input_data_config,
            outputDataConfig=self.output_data_config
        )

        # Wait for the job to complete
        job_arn = response.get('jobArn')
        # job_arn = "arn:aws:bedrock:ap-southeast-2:209378968454:model-invocation-job/1gxm5lod7lje"
        while True:
            response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status = response['status']
            if status == 'Completed':
                print("Job succeeded")
                break
            if status in ['Failed', 'Cancelled']:
                raise RuntimeError(f"Job failed with status {status}")
            time.sleep(60)  # Wait for a minute before checking again

        # Once the job has succeeded, download the output data
        suffix_job = job_arn.split('model-invocation-job/')[-1]
        s3.download_file("ctrlpost-bedrock-inference-bucket", f"output_data/{suffix_job}/input_gemba_tmp.jsonl.out",
                         f'tmp/output_gemba_tmp.jsonl')

        parsed_answers = []
        # Read the output data
        with open(f'tmp/output_gemba_tmp.jsonl', 'r') as f:
            for line in f:
                full_response = json.loads(line)
                answer_id = int(response['recordId']) - 1
                try:
                    full_answer, stop_reason = gather_response_bedrock_inference(full_response, model)
                except Exception as e:
                    # print(f"Error parsing response: {e}")
                    full_answer = ""
                    stop_reason = "unknown"
                answer = parse_response(full_answer)
                if self.verbose:
                    print(f"Answer (t=0): " + colored(answer, "yellow") + " (" + colored(full_answer,
                                                                                                     "blue") + ")",
                          file=sys.stderr)
                if answer is None:
                    continue
                parsed_answers.append(
                    {
                        "temperature": 0,
                        "answer_id": answer_id,
                        "answer": answer,
                        "prompt": id2prompt[answer_id],
                        "finish_reason": stop_reason,
                        "model": model,
                    }
                )

        return parsed_answers


    def request_api(self, prompt, model, temperature=0, max_tokens=None):
        if temperature > 10:
            return []

        while True:
            try:
                if self.api_type == "bedrock":
                    return self.call_bedrock_api(prompt, model, temperature, max_tokens)
                
                response = self.call_api(prompt, model, temperature, max_tokens)
                break
            except Exception as e:
                # response was filtered
                if hasattr(e, 'code'):
                    if e.code == 'content_filter':
                        return []
                    print(e.code, file=sys.stderr)
                if hasattr(e, 'error') and e.error['code'] == 'invalid_model_output':
                    return []

                # frequent error is reaching the API limit
                print(colored("Error, retrying...", "red"), file=sys.stderr)
                print(e, file=sys.stderr)
                time.sleep(1)

        answers = []
        for choice in response.choices:
            if choice.message.content is None:
                return []
            if hasattr(choice, "message"):
                answer = choice.message.content.strip()
            else:
                answer = choice.text.strip()
                
            # one of the responses didn't finish, we need to request more tokens
            if choice.finish_reason != "stop":
                if self.verbose:
                    print(colored(f"Increasing max tokens to fit answers.", "red") + colored(answer, "blue"), file=sys.stderr)
                print(f"Finish reason: {choice.finish_reason}", file=sys.stderr)
                if max_tokens is None:
                    return []
                return self.request_api(prompt, model, temperature=temperature, max_tokens=max_tokens + 200)

            answers.append({
                "answer": answer,
                "finish_reason": choice.finish_reason,
            })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_bedrock_api(self, prompt, model, temperature, max_tokens):
        # Bedrock models have different request/response formats. 
        # Here we support Claude (Anthropic) and general Converse API if possible, 
        # but for simplicity and broader compatibility within GEMBA's structure, 
        # we'll use the Converse API which is more unified.
        
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{
                "role": "user",
                "content": [{"text": prompt}],
            }]

        # Convert OpenAI-style messages to Bedrock Converse messages if needed
        bedrock_messages = []
        system_prompts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_prompts.append({"text": content})
            else:
                if isinstance(content, str):
                    bedrock_messages.append({
                        "role": role,
                        "content": [{"text": content}]
                    })
                else:
                    bedrock_messages.append({
                        "role": role,
                        "content": content
                    })

        inference_config = {
            "temperature": temperature / 10.0,
            # "topP": 1.0,
        }
        if max_tokens:
            inference_config["maxTokens"] = max_tokens

        try:
            params = {
                "modelId": model,
                "messages": bedrock_messages,
                "inferenceConfig": inference_config,
            }
            if system_prompts:
                params["system"] = system_prompts

            response = self.client.converse(**params)

            # Generalized response parsing
            if 'output' in response and 'message' in response['output'] and 'content' in response['output']['message']:
                content = response['output']['message']['content']
            elif 'content' in response:
                # Fallback if content is directly in response (some models/mock responses)
                content = response['content']
            else:
                if self.verbose:
                    print(f"Unexpected Bedrock response format: {response}", file=sys.stderr)
                return []

            if isinstance(content, list):
                texts = [block['text'] for block in content if isinstance(block, dict) and 'text' in block]
            elif isinstance(content, dict) and 'text' in content:
                texts = [content['text']]
            elif isinstance(content, str):
                texts = [content]
            else:
                texts = []

            if not texts:
                # If no text block is found, it might be a tool use or something else.
                # For GEMBA, we expect text. If missing, we should probably handle it or log it.
                if self.verbose:
                    print(f"No text in Bedrock response: {response}", file=sys.stderr)
                return []
            
            answer = "\n".join(texts).strip()
            finish_reason = response['stopReason']
            
            # Map Bedrock stop reasons to OpenAI finish reasons
            if finish_reason == "end_turn":
                finish_reason = "stop"
            elif finish_reason == "max_tokens":
                finish_reason = "length"

            if finish_reason != "stop":
                if self.verbose:
                    print(colored(f"Increasing max tokens to fit answers.", "red") + colored(answer, "blue"), file=sys.stderr)
                if max_tokens is None:
                    # Default max tokens if none provided and it didn't finish
                    return self.call_bedrock_api(prompt, model, temperature, 500)
                return self.call_bedrock_api(prompt, model, temperature, max_tokens + 200)

            return [{
                "answer": answer,
                "finish_reason": finish_reason,
            }]
        except Exception as e:
            print(colored("Error calling Bedrock, retrying...", "red"), file=sys.stderr)
            print(e, file=sys.stderr)
            time.sleep(1)
            return self.call_bedrock_api(prompt, model, temperature, max_tokens)

    def call_api(self, prompt, model, temperature, max_tokens):
        parameters = {
            "temperature": temperature/10,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "model": model
        }

        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens

        if isinstance(prompt, list):
            # check that prompt contain list of dictionaries with role and content
            assert all(isinstance(p, dict) for p in prompt), "Prompts must be a list of dictionaries."
            assert all("role" in p and "content" in p for p in prompt), "Prompts must be a list of dictionaries with role and content."

            parameters["messages"] = prompt
        else:
            parameters["messages"] = [{
                "role": "user",
                "content": prompt,
            }]

        return self.client.chat.completions.create(**parameters)
    
    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=None, inference_type='on_demand'):

        if inference_type == 'on_demand':
            answers = []
            for i, row in tqdm.tqdm(df.iterrows(), total=len(df), file=sys.stderr):
                prompt = row["prompt"]
                parsed_answers = self.request(prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens)
                answers += parsed_answers
        elif inference_type == 'batch':
            answers = self.request_batch(df, model, parse_mqm_answer, max_tokens=max_tokens)
        else:
            raise Exception(f"Inference type {inference_type} not supported.")
        return answers
