from langchain_core.language_models import BaseChatModel
from langchain import hub
from langchain.output_parsers.json import SimpleJsonOutputParser
from langsmith import traceable


class ChainOfDensity:
    """
    A class that wraps the Chain of Density (CoD) prompting technique to iteratively forces the model to pack more detail into a fixed length summary.
    """

    def __init__(
        self,
        model: BaseChatModel,
        content_category: str = "Article",
        entity_range: str = "1-3",
        max_words: int = 80,
        iterations: int = 3,
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
    import requests
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    url = "https://raw.githubusercontent.com/SuperMuel/Syncademic/main/README.md"
    content = requests.get(url).text

    model = ChatOpenAI(model="gpt-4o")

    cod = ChainOfDensity(model=model)

    summary = cod.generate_summary(content)

    print(summary)
