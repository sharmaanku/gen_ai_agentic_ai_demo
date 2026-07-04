"""
Mini AI Agent with Two Tools
"""

# -----------------------
# Tool 1
# -----------------------

def calculator(exp):
    return eval(exp)


# -----------------------
# Tool 2
# -----------------------

def capital(country):

    capitals = {
        "india": "New Delhi",
        "france": "Paris",
        "japan": "Tokyo",
        "usa": "Washington DC"
    }

    return capitals.get(country.lower(), "Unknown")


# -----------------------
# Fake LLM
# -----------------------

def llm(question, observation=None):

    if observation is None:

        print("\nThought:")

        if "capital" in question.lower():

            print("Need capital information.")

            print("\nAction:")
            print("Capital Tool")

            country = question.split()[-1].replace("?", "")

            return {
                "action": "capital",
                "input": country
            }

        else:

            print("Need calculator.")

            print("\nAction:")
            print("Calculator")

            expression = question.replace("What is", "").replace("?", "").strip()

            return {
                "action": "calculator",
                "input": expression
            }

    else:

        print("\nObservation:")
        print(observation)

        print("\nThought:")
        print("Now I have enough information.")

        return {
            "final": observation
        }


# -----------------------
# ReAct Loop
# -----------------------

def react(question):

    print("=" * 60)
    print("Question:", question)

    response = llm(question)

    if response["action"] == "calculator":
        observation = calculator(response["input"])

    elif response["action"] == "capital":
        observation = capital(response["input"])

    final = llm(question, observation)

    print("\nFinal Answer:")
    print(final["final"])

    print("=" * 60)


# -----------------------
# Demo
# -----------------------

react("What is 15 * 7?")

react("What is the capital of India?")
