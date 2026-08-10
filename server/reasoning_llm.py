from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class ChatGPT:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        is_langsmith: bool = False,
    ):
        self.api_key = settings.llm_api_key

        if is_langsmith:
            self.client = wrap_openai(
                OpenAI(
                    api_key=self.api_key,
                    base_url=settings.llm_base_url,
                    max_retries=0, # openAI 自带重试机制
                )
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.llm_base_url,
                max_retries=0,
            )

        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=0.5,
            min=0.5,
            max=4,
        ),
        reraise=True,
    )
    def generate(
        self,
        sys_prompt: str,
        prompt: str,
        temperature: float = 0.0,
    ):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                # {
                #     "role": "system",
                #     "content": sys_prompt,
                # },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=False,
            temperature=temperature,
        )

        return response.choices[0].message.content


class DeepSeek:
    pass