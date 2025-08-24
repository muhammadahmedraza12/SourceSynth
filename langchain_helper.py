from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
# from secret_key import openapi_key

# import os
# os.environ["GOOGLE_API_KEY"] = openapi_key

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)

def generate_restaurant_name_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest only a fancy name, nothing else."
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name} restaurant. Return only a comma-separated list."
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"],
        verbose=False
    )

    response = chain({'cuisine': cuisine})
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_items("Pakistani"))
    

# return{
    # 'restaurant_name':"Kababjees",
    # 'menu_items':'chicken kabab, chicken tikka, chicken curry'
    # }

# Run
# result = chain({"cuisine": "Pakistani"})

# result = {
#     "restaurant_name": "Kababjees",
#     "menu_items": chain({"cuisine": "Pakistani"})["menu_items"]
# }


# final_output = f"{result['restaurant_name']}: {result['menu_items']}"
# print(final_output)


