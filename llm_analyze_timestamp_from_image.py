from langchain_openai import ChatOpenAI
from typing import Union
from pathlib import Path
import base64
from time import sleep
from langchain.schema.messages import HumanMessage, SystemMessage

class ImageAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize the ImageAnalyzer with OpenAI API key.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
            api_key=self.api_key,
            max_tokens=300
        )

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _validate_image_path(self, image_path: Union[str, Path]) -> bool:
        """
        Validate if the image path exists and is a file.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        return True

    def analyze_image(self, image_path: Union[str, Path], custom_prompt: str = None) -> str:
        """
        Analyze an image using the GPT-4 Vision model.
        
        Args:
            image_path: Path to the image file
            custom_prompt: Optional custom system prompt. If None, uses default prompt.
            
        Returns:
            str: Analysis result from the model
        """
        # Validate and encode image
        self._validate_image_path(image_path)
        base64_image = self._encode_image(image_path)

        # Default system prompt if none provided
        default_prompt = """
        Task: Extract the date and time from the given image. The date and time location is on the top-left corner.
        Note: return only the date and time using the format YYYY-MM-DD HH:MM:SS without any introductory text.
        """
        
        system_prompt = custom_prompt if custom_prompt else default_prompt

        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]

        # Make API call with retry logic
        retry = 0
        max_retries = 3
        retry_delay = 2

        while True:
            try:
                response = self.llm.invoke(messages)
                break
            except Exception as e:
                if 'internal_server_error' in str(e):
                    retry += 1
                    if retry > max_retries:
                        raise Exception(f"Failed after {max_retries} retries: {str(e)}")
                    print(f"Retry attempt {retry} of {max_retries}")
                    sleep(retry_delay)
                else:
                    raise e

        return response.content

# Example usage:
def llm_analyze_timestamp_from_image(image_path: str) -> str:
    """
    Simple function to analyze timestamp from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Extracted timestamp
    """
    try:
        analyzer = ImageAnalyzer(api_key='sk-proj-KTZRGHDVRfnT6SuJnF13T3BlbkFJhipu548mOgZZQGH8mOIG')
        return analyzer.analyze_image(image_path)
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return None