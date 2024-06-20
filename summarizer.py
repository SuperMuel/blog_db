from langchain import hub
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.language_models import BaseChatModel
from langsmith import traceable

from rss_feed_parser import RSSFeedParser


class ChainOfDensity:
    """
    A class that wraps the Chain of Density (CoD) prompting technique to iteratively forces the model to pack more detail into a fixed length summary.
    """

    def __init__(
        self,
        model: BaseChatModel,
        content_category: str = "Article",
        entity_range: str = "2-3",
        max_words: int = 100,
        iterations: int = 6,
    ):
        self.model = model
        self.content_category = content_category
        self.entity_range = entity_range
        self.max_words = max_words
        self.iterations = iterations
        self.chain = self._get_chain()

    def _get_chain(self):
        prompt = hub.pull("supermuel/chain-of-density-prompt-multilingual")
        json_parser = SimpleJsonOutputParser()
        return prompt | self.model | json_parser | self._extract_last_summary

    def _get_input(self, content) -> dict:
        return {
            "content": content,
            "content_category": self.content_category,
            "entity_range": self.entity_range,
            "max_words": self.max_words,
            "iterations": self.iterations,
        }

    @staticmethod
    def _extract_last_summary(output: list[dict]):
        try:
            return output[-1]["denser_summary"]
        except IndexError:
            raise Exception("Invalid format returned from the model : No output.")
        except KeyError:
            raise Exception(
                "Invalid format returned from the model : No denser_summary key."
            )
        except Exception as e:
            raise Exception(f"Invalid format returned from the model : {e}")

    @traceable
    def generate_summary(self, content: str) -> str:
        """
        Generate a dense summary of the given content.
        """

        return self.chain.invoke(self._get_input(content))


# example
if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    model = ChatOpenAI(model="gpt-4o")
    cod = ChainOfDensity(model=model)

    parser = RSSFeedParser("https://sociaty.io/feed")
    parser.parse()

    for entry in parser.get_entries():
        print(f"\033[92m{entry.title}\033[0m")

        summary = cod.generate_summary(entry.markdown_content)

        print(f"\033[93m{summary}\033[0m")

        print("\n\n")
