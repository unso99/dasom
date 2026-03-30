import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def _encode_image(image_path: str) -> tuple[str, str]:
    """
    이미지 파일을 base64로 인코딩하고 미디어 타입을 반환합니다.

    Returns:
        (base64_data, media_type)
    """
    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded, media_type


def stream_multimodal_answer(
    image_path: str,
    user_prompt: str,
    system_prompt: str = "당신은 이미지를 분석하는 AI 어시스턴트입니다.",
    model_name: str = "gpt-4.1-mini",
):
    """
    이미지와 텍스트 프롬프트를 받아 스트리밍 방식으로 답변을 생성합니다.

    Args:
        image_path   : 로컬 이미지 파일 경로
        user_prompt  : 사용자 질문
        system_prompt: 시스템 프롬프트
        model_name   : 사용할 OpenAI 모델명

    Yields:
        str: 스트리밍 토큰
    """
    llm = ChatOpenAI(model=model_name, temperature=0)

    base64_data, media_type = _encode_image(image_path)

    messages = [
        {"role": "system", "content": system_prompt},
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_data}",
                },
            },
            {
                "type": "text",
                "text": user_prompt,
            },
        ]),
    ]

    for chunk in llm.stream(messages):
        yield chunk.content