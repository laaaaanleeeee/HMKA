import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()


class GeminiLLM:

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        max_attempts: int = 3,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_attempts = max_attempts

    def generate(self, prompt: str) -> str:
        for attempt in range(self.max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                answer = (response.text or "").strip()

                if not answer:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return answer

            except errors.ClientError as exc:
                if exc.code == 429:
                    raise RuntimeError(
                        "Gemini API quota exhausted (429). "
                        "Check your API quota or billing settings."
                    ) from exc
                raise

            except errors.ServerError as exc:
                if exc.code not in (500, 502, 503, 504):
                    raise

                if attempt == self.max_attempts - 1:
                    raise

                wait_seconds = 2 ** (attempt + 1)
                print(
                    f"[Gemini] Server error {exc.code}. "
                    f"Retrying in {wait_seconds}s "
                    f"(attempt {attempt + 1}/{self.max_attempts})..."
                )
                time.sleep(wait_seconds)