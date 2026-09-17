# coding=utf-8
import requests


DEFAULT_MODEL = "gemma4:26b"
DEFAULT_BASE_URL = "http://localhost:11434"

CLEAN_PROMPT = """\
请清理下面这段语音识别得到的文字稿。

要求：
1. 保留原始内容、事实、观点和含义，不要总结、扩写或改写。
2. 删除没有实际意义的口语填充词，例如“嗯”“呃”“那个”“就是”等。
3. 删除明显的口语重复。
4. 如果说话者在一句话中进行了自我修正，保留最终想表达的内容。
5. 添加合理的标点和分段，使文字更容易阅读。
6. 只有在语音识别错误非常明确时才进行修正。
7. 对不确定的人名、书名、机构名、专业术语等，不要猜测。
8. 尽量保留说话者原本的语言风格。
9. 如果原文中的句子本身不完整、含义不明确或说话者没有说完，
   不要根据上下文补全说话者可能想表达的内容。
10. 清理的目标不是把文字改写成更好的文章，而是得到
    “如果说话者去掉口头冗余后，实际说出来的内容”。
11. 只输出清理后的文字稿，不要添加解释。

原始文字稿：

{text}
"""


class TranscriptCleaner:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def clean(self, text: str, think: bool=False) -> str:
        if not text.strip():
            return ""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": CLEAN_PROMPT.format(text=text),
                "stream": False,
                "think": think,
            },
        )
        response.raise_for_status()

        return response.json()["response"].strip()


if __name__ == '__main__':
    cleaner = TranscriptCleaner()
    raw = """嗯，我来试一下，如果不做任何准备，直接拿起手机来直接录音，可能会出现一个什么样的结果吧？嗯，我刚才在做的就是把那个语音的识别，还有就是嗯，语音的识别啊，对啊，还有一个就是clean cleaner，就是将识别出来的一个包含了可能包含大量重复冗余、停顿、口头禅的一些各种各样的情况，把它给清理一下，清理成比较简简洁一些的那样的一个文字吧。"""
    print(f'raw: {raw}')
    print(f'cleaned: {cleaner.clean(raw)}')
