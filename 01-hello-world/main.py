from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
load_dotenv()

def main():
    print("Hello from hello-world!")
    information = """
    Frog (カエル, Kaeru) é um guerreiro de Guardia em 600 A.D., cujo verdadeiro nome é Glenn. Ele era escudeiro de Cyrus, um "Cavaleiro da Tavola Quadrada" ("Cavaleiro de Guardia" na versão DS). Glenn testemunhou o assassinato de Cyrus por Magus e foi transformado em um sapo antropomórfico. Frog dedica sua vida a proteger a rainha Leene e deseja vingar Cyrus, matando Magus. Ele é o verdadeiro portador da Masamune, uma antiga e lendária espada, maior fraqueza de Magus. Frog pode posteriormente dar descanso ao fantasma de Cyrus, e tem a oportunidade tanto de vencer Magus quanto de se aliar a ele. Se Frog combate e vence Magus, ele se tornará humano novamente ao fim do jogo. Frog usa uma espada larga e pode aprender a usar magias de água.
    """

    summary_template = """
        Given the {information} about the character I want you to create:
        1. A short summary
        2. two interesting facts about them, do not add more than two facts
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)
if __name__ == "__main__":
    main()
