import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client


def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) "
        "Your task is to extract the ontology of terms mentioned in the given context. "
        "These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms must only include object, entity, location, organization, person, condition, documents, service, concept, date.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms and the relation between them, like the following: \n"
        "{\n"
        "edges: [\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_1_type": "object|entity|location|organization|person|condition|documents|concept|date",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "node_2_type": "object|entity|location|organization|person|condition|documents|concept|date",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   },\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_1_type": "object|entity|location|organization|person|condition|documents|concept|date",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "node_2_type": "object|entity|location|organization|person|condition|documents|concept|date",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
        "}"
    )

    USER_PROMPT = f"#Context: ```{input}``` \n\n #Output: "

    # response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    response, _ = client.chat(model_name=model, messages=[{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": USER_PROMPT}])

    try:
        result = json.loads(response)['edges']
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result



def communitySummaryPrompt(nodes: str, edges: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    SYS_PROMPT = """
        Your task is to generate a comprehensive summary of a knowledge graph community. 
        The context of the input includes nodes and relationships between the nodes. 
        Analyze and create a summary of the community's overall structure, how its entities are related to each other, and significant keypoints associated with its entities.

        # Example Input
        
        Context:

        Entities:
        ABILA CITY PARK
        POK RALLY
        POK
        POKRALLY
        CENTRAL BULLETIN

        Relationships (node1,node2,relation,weight):
        ABILA CITY PARK,POK RALLY,Abila City Park is the location of the POK rally,1
        ABILA CITY PARK,POK,POK is holding a rally in Abila City Park,2
        ABILA CITY PARK,POKRALLY,The POKRally is taking place at Abila City Park,1
        ABILA CITY PARK,CENTRAL BULLETIN,Central Bulletin is reporting on the POK rally taking place in Abila City Park,1.5

        Output:

        The community revolves around the Abila City Park, which is the location of the POK rally. The park has relationships with POK, POKRALLY, and Central Bulletin, all
        of which are associated with the rally event.

    """
    
    USER_PROMPT = """

        # Real Data

        Use the following context for your answer. Do not make anything up in your answer. Try to analyze and summarize the knowledge exists between these entities in one long paragraph.

        Context:

        Entities:
        {nodes}

        Relationships (node1,node2,relation,weight):
        {edges}

        Output:
    """.format(nodes=nodes, edges=edges)

    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)

    return response