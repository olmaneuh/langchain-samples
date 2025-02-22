from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


USER_INFO: str = (
    "Jordan Rivers, 29, is an urban planner passionate about sustainable city design. "
    "Outside work, they enjoy photography, local history, and digital art, "
    "blending creativity with a vision for greener communities."
)

TODO_LIST_TEMPLATE: str = (
    "Given the following information about a fictional user:\n{user_info}\n"
    "Create a todo list for a regular weekday."
)


def create_todo_list(user_info: str) -> str:
    """
    LangChain code.
    Generates a todo list based on the provided user information.
    """

    # create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["user_info"], template=TODO_LIST_TEMPLATE
    )

    # instance of the llm that will be used
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # define a chain to generate the todo list
    chain = prompt_template | llm | StrOutputParser()

    # return the generated todo list as a string.
    return chain.invoke(input={"user_info": user_info})


def main() -> None:
    """Main function to run the application."""

    # load environment variables
    load_dotenv()

    # create todo list
    todo_list = create_todo_list(USER_INFO)

    # print the todo list created
    print(todo_list)


if __name__ == "__main__":
    main()
