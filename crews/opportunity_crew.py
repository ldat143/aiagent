import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
#from tools.custom_tools import SerperDevTool

from tools.custom_tools import DistanceCalculatorTool, PopulationDataTool

load_dotenv()

# Configurar el modelo LLM
llm = LLM(
    model='gemini/gemini-2.0-flash-lite',
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0,
    verbose=True,
)

# Herramientas
serper = SerperDevTool()
distance_tool = DistanceCalculatorTool()
population_tool = PopulationDataTool()

# Agentes
dealership_info_agent = Agent(
    role="Dealership Info Gatherer",
    goal="Collect key information about the target car dealership {dealership} with a zipcode {zipcode}.",
    backstory="Expert researcher skilled at finding dealership details.",
    llm=llm,
    verbose=True
)

opportunities_researcher = Agent(
    role="Opportunities Researcher",
    goal="Using the dealership's {dealership} info, identify cities within 100 miles with population > 1000.",
    backstory="Market research specialist for geographic business expansion.",
    llm=llm,
    verbose=True
)

data_organizer = Agent(
    role="Data Organizer",
    goal="Structure opportunity data into a readable table.",
    backstory="Data analyst experienced in formatting and cleaning information.",
    llm=llm,
    verbose=True
)

results_supervisor = Agent(
    role="Results Supervisor",
    goal="Validate all opportunity data and present final structured output.",
    backstory="Quality assurance expert reviewing all opportunity recommendations.",
    llm=llm,
    verbose=True
)

# Tareas
dealership_info_task = Task(
    description="Find the OEM, address, and coordinates of the dealership {dealership} with zipcode {zipcode}.",
    expected_output="Dictionary with OEM, address, and coordinates.",
    agent=dealership_info_agent,
    tools=[serper]
)

opportunities_research_task = Task(
    description=(
        "Using the OEM, address, state and location of the dealership {dealership}, look for nearby cities and/or metro areas "
        "within a 100-mile radius and with population greater than 1000 people. "
        "Use SerperDevTool to find city names, their states and addresses. "
        "Use PopulationDataTool to verify each city has a population > 1000. "
        "Use DistanceCalculatorTool for distances in miles. "
        "Return a list of valid cities with their City/Town Name, Distance from dealer, and State."
    ),
    expected_output="A list of dictionaries: Nearby City/Town Name, Distance From Dealer, State.",
    agent=opportunities_researcher,
    tools=[serper, distance_tool, population_tool]
)

data_organization_task = Task(
    description="Take the list of nearby cities and organize the data into a markdown table.",
    expected_output="Markdown table: City | Distance | State.",
    agent=data_organizer
)

supervision_task = Task(
    description="Review the dealership info and nearby city suggestions, and present the results clearly.",
    expected_output="Final structured summary with dealership and opportunity info.",
    agent=results_supervisor
)

# Crew
opportunity_crew = Crew(
    agents=[dealership_info_agent, opportunities_researcher, data_organizer, results_supervisor],
    tasks=[dealership_info_task, opportunities_research_task, data_organization_task, supervision_task],
    process=Process.sequential,
    verbose=True
)

# Función para ejecutar desde main.py
def run_opportunity_crew(**kwargs):
    return opportunity_crew.kickoff(inputs=kwargs)
